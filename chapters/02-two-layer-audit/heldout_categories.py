"""Held-out-category transfer and few-shot adaptation on nine unseen AD categories.

The six v3.1 AD categories are the only AD categories with prior PatchCore /
CLIP scores.  Nine additional MVTec AD categories were scored for this study
with the same cached PatchCore / CLIP recipe.  They are used as:
  * zero-label category transfer: train on the six seen categories, test on nine
  * few-shot adaptation: add 1/2/5/10 labeled anomaly corrections per new category
Only the 20 calibration normals of the new category are used for normalization
and thresholding; half are used to fit, half only for the operating threshold.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
from common import (
    AD_CATS6,
    AD_HELDOUT9,
    auc,
    average_precision,
    balanced_accuracy,
    load_embedding,
    write_csv,
    write_json,
)
from features import attach_features
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = [0, 1, 2, 3, 4]
BUDGETS = [0, 1, 2, 5, 10]


def fit_logistic(rows, keys, C=0.1):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    y = np.array([r["truth_bin"] for r in rows])
    if len(set(y)) < 2:
        return None
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced", C=C)).fit(X, y)


def scores_logistic(model, rows, keys):
    if model is None:
        return np.array([r["z_patch"] for r in rows], float)
    X = np.array([[r[k] for k in keys] for r in rows], float)
    return model.predict_proba(X)[:, 1]


def threshold_scores(cal, s):
    """Per-category q95 thresholds from the threshold half of calibration normals."""
    by = defaultdict(list)
    for r, v in zip(cal, s):
        by[r["category"]].append(v)
    return {k: float(np.quantile(v, 0.95)) for k, v in by.items()}


def eval_with_threshold(rows, s, thresholds, y):
    pred = np.array([int(v > thresholds[r["category"]]) for r, v in zip(rows, s)])
    return {
        "n": len(rows), "anomalies": int(y.sum()),
        "auc": auc(y, s), "average_precision": average_precision(y, s),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "recall_at_cal_q95": float(pred[y == 1].mean()) if (y == 1).any() else None,
        "fpr_at_cal_q95": float(pred[y == 0].mean()) if (y == 0).any() else None,
        "wrong_release": int(((pred == 0) & (y == 1)).sum()),
        "false_hold": int(((pred == 1) & (y == 0)).sum()),
    }


def split_calibration(cal):
    """Deterministic 10 fit / 10 threshold split per category."""
    by = defaultdict(list)
    for r in cal:
        by[r["category"]].append(r)
    fit, threshold = [], []
    for cat, rs in by.items():
        rs = sorted(rs, key=lambda r: r["sha256"])
        threshold += rs[:10]
        fit += rs[10:]
    return fit, threshold


def prototype_scores(train_normals, train_anomalies, target, emb):
    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)
    N = [emb[r["sha256"]] for r in train_normals if r["sha256"] in emb]
    A = [emb[r["sha256"]] for r in train_anomalies if r["sha256"] in emb]
    if not N:
        return np.array([r["z_patch"] for r in target], float)
    mu0 = unit(np.mean([unit(x) for x in N], axis=0))
    mu1 = unit(np.mean([unit(x) for x in A], axis=0)) if A else None
    out = []
    for r in target:
        v = emb.get(r["sha256"])
        if v is None:
            out.append(r["z_patch"])
            continue
        v = unit(v)
        if mu1 is None:
            out.append(float(1 - v @ mu0))
        else:
            out.append(float(v @ mu1 - v @ mu0))
    return np.array(out, float)


def knn_scores(train_normals, train_anomalies, target, emb, k=5):
    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)
    refs = [(emb[r["sha256"]], 0) for r in train_normals if r["sha256"] in emb]
    refs += [(emb[r["sha256"]], 1) for r in train_anomalies if r["sha256"] in emb]
    if not refs:
        return np.array([r["z_patch"] for r in target], float)
    R = np.stack([unit(x) for x, _ in refs])
    ry = np.array([y for _, y in refs])
    has_anomaly = bool((ry == 1).any())
    out = []
    for r in target:
        v = emb.get(r["sha256"])
        if v is None:
            out.append(r["z_patch"])
            continue
        sim = R @ unit(v)
        ix = np.argsort(-sim)[:k]
        out.append(float(ry[ix].mean()) if has_anomaly else float(1 - sim.max()))
    return np.array(out, float)


def nca_scores(train_normals, train_anomalies, target, emb):
    """Metric learning: PCA on CLIP embeddings, NCA, then 1-NN anomaly fraction."""
    def unit(v):
        return v / (np.linalg.norm(v) + 1e-9)
    train = train_normals + train_anomalies
    train = [r for r in train if r["sha256"] in emb]
    target = [r for r in target if r["sha256"] in emb]
    if len(train) < 8 or len({r["truth_bin"] for r in train}) < 2 or not target:
        return None
    X = np.stack([unit(emb[r["sha256"]]) for r in train])
    y = np.array([r["truth_bin"] for r in train])
    pca = PCA(n_components=min(32, X.shape[0] - 1, X.shape[1]), random_state=0)
    Xp = pca.fit_transform(X)
    try:
        nca = NeighborhoodComponentsAnalysis(n_components=min(8, Xp.shape[1]),
                                             max_iter=50, random_state=0)
        Xn = nca.fit_transform(Xp, y)
    except Exception:
        return None
    T = pca.transform(np.stack([unit(emb[r["sha256"]]) for r in target]))
    try:
        T = nca.transform(T)
    except Exception:
        return None
    out = []
    for t in T:
        d = ((Xn - t) ** 2).sum(1)
        j = int(np.argmin(d))
        out.append(float(y[j]))
    return np.array(out, float)


def main():
    rows, _ = attach_features(include_heldout=True)
    held = [r for r in rows if r["category"] in AD_HELDOUT9 and r["dataset"] == "mvtec_ad"]
    ad6 = [r for r in rows if r["category"] in AD_CATS6 and r["dataset"] == "mvtec_ad"]
    emb = load_embedding("clip")
    train6 = [r for r in ad6 if r["role"] == "router_train"]
    cal6 = [r for r in ad6 if r["role"] == "calibration"]
    cal_all = [r for r in held if r["role"] == "calibration"]
    cal_fit, cal_thr = split_calibration(cal_all)
    support = [r for r in held if r["role"] == "support_anomaly"]
    test = [r for r in held if r["role"] in ("test_good", "test_anomaly")]
    y_test = np.array([r["truth_bin"] for r in test])
    out = []
    per_cat = []

    # Global models trained on the six seen categories.
    keys_z = ["z_patch", "z_clip"]
    keys_raw = ["patch_score", "clip_score"]
    model_z = fit_logistic(train6, keys_z, C=0.5)
    model_raw = fit_logistic(train6, keys_raw, C=0.5)
    for name, model, keys in [("global_logistic_z", model_z, keys_z),
                              ("global_logistic_raw", model_raw, keys_raw)]:
        s = scores_logistic(model, test, keys)
        # zero-label threshold: use all 20 calibration normals
        cs = scores_logistic(model, cal_all, keys)
        th = threshold_scores(cal_all, cs)
        m = eval_with_threshold(test, s, th, y_test)
        out.append({"method": name, "budget": 0, "seed": None, **m})
        for cat in AD_HELDOUT9:
            ix = np.array([r["category"] == cat for r in test])
            if len(set(y_test[ix])) > 1:
                per_cat.append({"method": name, "category": cat,
                                "auc": auc(y_test[ix], s[ix]), "n": int(ix.sum())})
    # Raw patch threshold selected on AD6 calibration normals, no new calibration.
    cal6s = np.array([r["patch_score"] for r in cal6])
    t_raw = float(np.quantile(cal6s, 0.95))
    s = np.array([r["patch_score"] for r in test])
    pred = (s > t_raw).astype(int)
    out.append({"method": "raw_patch_ad6_q95", "budget": 0, "seed": None,
                "n": len(test), "anomalies": int(y_test.sum()), "auc": auc(y_test, s),
                "average_precision": average_precision(y_test, s),
                "balanced_accuracy": balanced_accuracy(y_test, pred),
                "recall_at_cal_q95": float(pred[y_test == 1].mean()),
                "fpr_at_cal_q95": float(pred[y_test == 0].mean()),
                "wrong_release": int(((pred == 0) & (y_test == 1)).sum()),
                "false_hold": int(((pred == 1) & (y_test == 0)).sum())})
    # Per-category conformal threshold using only the new category's normals (no labels).
    s = np.array([r["z_patch"] for r in test])
    cs = np.array([r["z_patch"] for r in cal_all])
    th = threshold_scores(cal_all, cs)
    out.append({"method": "per_category_conformal_patch", "budget": 0, "seed": None,
                **eval_with_threshold(test, s, th, y_test)})

    # Few-shot adaptation per category and budget.
    by_cat_support = defaultdict(list)
    for r in support:
        by_cat_support[r["category"]].append(r)
    by_cat_calfit = defaultdict(list)
    for r in cal_fit:
        by_cat_calfit[r["category"]].append(r)
    by_cat_calthr = defaultdict(list)
    for r in cal_thr:
        by_cat_calthr[r["category"]].append(r)
    for budget in BUDGETS:
        for seed in SEEDS[:3] if budget < 10 else SEEDS[:1]:
            rng = np.random.default_rng(seed)
            pred_rows = []
            # one concatenated evaluation for this seed/budget
            for cat in AD_HELDOUT9:
                cat_sup = list(by_cat_support[cat])
                if budget:
                    rng.shuffle(cat_sup)
                    cat_sup = cat_sup[:budget]
                else:
                    cat_sup = []
                fitn = by_cat_calfit[cat]
                thrn = by_cat_calthr[cat]
                cat_test = [r for r in test if r["category"] == cat]
                # logistic on calibrated scores
                labels = fitn + cat_sup
                model = fit_logistic(labels, keys_z, C=0.1) if cat_sup else None
                s = scores_logistic(model, cat_test, keys_z)
                cs = scores_logistic(model, thrn, keys_z)
                th = threshold_scores(thrn, cs)
                # prototype / kNN / NCA on CLIP embeddings
                sp = prototype_scores(fitn, cat_sup, cat_test, emb)
                cp = prototype_scores(fitn, cat_sup, thrn, emb)
                thp = threshold_scores(thrn, cp)
                sk = knn_scores(fitn, cat_sup, cat_test, emb)
                ck = knn_scores(fitn, cat_sup, thrn, emb)
                thk = threshold_scores(thrn, ck)
                sn = nca_scores(fitn, cat_sup, cat_test, emb)
                cn = nca_scores(fitn, cat_sup, thrn, emb) if sn is not None else None
                thn = threshold_scores(thrn, cn) if cn is not None else None
                for method, ss, tt in [
                    ("fewshot_logistic_z", s, th),
                    ("prototype_clip", sp, thp),
                    ("knn_clip", sk, thk),
                ] + ([("nca_clip", sn, thn)] if sn is not None else []):
                    yy = np.array([r["truth_bin"] for r in cat_test])
                    m = eval_with_threshold(cat_test, ss, tt, yy)
                    pred_rows.append((method, cat, m, ss, yy))
            # pool across categories for each method
            for method in sorted({x[0] for x in pred_rows}):
                sub = [x for x in pred_rows if x[0] == method]
                ss = np.concatenate([x[3] for x in sub])
                yy = np.concatenate([x[4] for x in sub])
                # pooled threshold metrics are already category-specific; recompute counts
                wr = sum(x[2]["wrong_release"] for x in sub)
                fh = sum(x[2]["false_hold"] for x in sub)
                out.append({"method": method, "budget": budget, "seed": seed,
                            "n": len(yy), "anomalies": int(yy.sum()),
                            "auc": auc(yy, ss), "average_precision": average_precision(yy, ss),
                            "balanced_accuracy": float(np.mean([x[2]["balanced_accuracy"] for x in sub])),
                            "recall_at_cal_q95": float(np.mean([x[2]["recall_at_cal_q95"] for x in sub])),
                            "fpr_at_cal_q95": float(np.mean([x[2]["fpr_at_cal_q95"] for x in sub])),
                            "wrong_release": wr, "false_hold": fh})
                for x in sub:
                    per_cat.append({"method": method, "category": x[1], "budget": budget,
                                    "seed": seed, "auc": x[2]["auc"], "n": x[2]["n"]})
    write_csv("heldout_adaptation_raw.csv", out)
    write_csv("heldout_per_category.csv", per_cat)
    # summary over seeds
    groups = defaultdict(list)
    for r in out:
        groups[(r["method"], r["budget"])].append(r)
    summary = []
    for (method, budget), rs in sorted(groups.items()):
        row = {"method": method, "budget": budget, "n_seeds": len(rs)}
        for k in ["auc", "average_precision", "balanced_accuracy", "recall_at_cal_q95",
                  "fpr_at_cal_q95", "wrong_release", "false_hold"]:
            vals = np.array([r[k] for r in rs if r[k] is not None], float)
            row[k + "_mean"] = float(vals.mean()) if len(vals) else None
            row[k + "_std"] = float(vals.std()) if len(vals) else None
        summary.append(row)
    write_csv("heldout_adaptation_summary.csv", summary)
    write_json("heldout_metadata.json", {
        "new_categories": AD_HELDOUT9,
        "n_new_images_scored": len(held),
        "train_categories": AD_CATS6,
        "support": "train EXCEPTION images (10/category), disjoint from test 15 anomalies",
        "test": "10 good + 15 anomalies per category (225 total)",
        "calibration": "20 normals/category; zero-label policies use all 20 for the threshold, "
                       "few-shot policies use 10 fit / 10 threshold",
        "transfer_control": "global model sees only six seen categories; no new-category labels",
        "historical_exposure": "the nine categories were not scored by v3.1 local PatchCore/CLIP, "
                                "but results/mvtec_ad_surface.json already lists all 15 AD categories "
                                "in a prior surface experiment; all 225 held-out test images overlap "
                                "that prior run exactly. This is a v3.1-pipeline holdout, not an "
                                "organizationally blind holdout.",
        "fewshot": "1/2/5/10 labeled anomaly corrections per new category; 3 seeds",
        "metric_learning": "PCA32 + NCA(8) on CLIP RN50 embeddings, 1-NN; CPU only",
        "caveat": "MVTec AD categories are distinct objects; image-level split, no physical unit IDs",
    })
    print(json.dumps({"rows": len(out), "summary": len(summary)}, indent=2))


if __name__ == "__main__":
    main()
