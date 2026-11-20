"""Calibration, conformal selective classification, and escalation frontiers.

Key distinction enforced here:
  * HIGH anomaly score -> HOLD (defect alarm)
  * LOW anomaly score  -> RELEASE
  * ambiguous          -> HUMAN
A case sent to HUMAN is never counted as correct or autonomously settled.
"""
from __future__ import annotations

import itertools
import json
from functools import partial

import numpy as np
from common import (
    AD_HELDOUT9,
    auc,
    average_precision,
    clopper_pearson,
    write_csv,
    write_json,
)
from features import attach_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit_logistic(rows, keys, C=0.5):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    y = np.array([r["truth_bin"] for r in rows])
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced", C=C))
    model.fit(X, y)
    return model


def predict(model, rows, keys):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    return model.predict_proba(X)[:, 1]


def prior_logistic(rows, all_cats):
    def feat(r):
        return [r["z_patch"], r["z_clip"]] + [int(r["category"] == c) for c in all_cats]
    X = np.array([feat(r) for r in rows], float)
    y = np.array([r["truth_bin"] for r in rows])
    model = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    model.fit(X, y)
    return (lambda rs: model.predict_proba(np.array([feat(r) for r in rs], float))[:, 1])


def patch_z_scores(rows):
    return {r["sha256"]: r["z_patch"] for r in rows}


def logistic_scores(model, keys, rows):
    return {r["sha256"]: float(s) for r, s in zip(rows, predict(model, rows, keys))}


def prior_scores(prior_fn, rows):
    return {r["sha256"]: float(s) for r, s in zip(rows, prior_fn(rows))}


def mahalanobis_scores(rows, cal):
    """Per-category robust Mahalanobis distance in calibrated z space."""
    out = {}
    for g in sorted({r["cal_group"] for r in rows}):
        c = [r for r in cal if r["cal_group"] == g and r["z_patch"] is not None]
        if len(c) < 5:
            continue
        X = np.array([[r["z_patch"], r["z_clip"]] for r in c], float)
        mu = X.mean(0)
        cov = np.cov(X.T) + np.eye(2) * 0.05
        inv = np.linalg.inv(cov)
        for r in rows:
            if r["cal_group"] == g:
                d = np.array([r["z_patch"], r["z_clip"]], float) - mu
                out[r["sha256"]] = float(d @ inv @ d)
    return out


def conformal_p(score, ref):
    ref = np.sort(np.asarray(ref, float))
    return (len(ref) - np.searchsorted(ref, score, side="left") + 1.0) / (len(ref) + 1.0)


def add_pvalues(rows, scores, cal_rows):
    """Per-category upper-tail conformal p-value, using only GOOD calibration rows."""
    ref = {}
    for r in cal_rows:
        if r["sha256"] in scores:
            ref.setdefault(r["cal_group"], []).append(scores[r["sha256"]])
    out = []
    for r in rows:
        if r["sha256"] not in scores:
            continue
        p = conformal_p(scores[r["sha256"]], ref.get(r["cal_group"], []))
        out.append((r, p))
    return out


def policy_counts(pairs, alpha_release, alpha_hold):
    c = {"release": 0, "hold": 0, "human": 0, "wrong_release": 0,
         "false_hold": 0, "true_release": 0, "anomalies_held": 0,
         "anomalies": 0, "good": 0}
    for r, p in pairs:
        truth = r["truth_bin"]
        c["anomalies" if truth else "good"] += 1
        if p > alpha_release:
            c["release"] += 1
            if truth:
                c["wrong_release"] += 1
            else:
                c["true_release"] += 1
        elif p < alpha_hold:
            c["hold"] += 1
            if truth:
                c["anomalies_held"] += 1
            else:
                c["false_hold"] += 1
        else:
            c["human"] += 1
    n = sum(v for k, v in c.items() if k in ("release", "hold", "human"))
    c["n"] = n
    c["autonomous_coverage"] = (c["release"] + c["hold"]) / n if n else None
    c["human_rate"] = c["human"] / n if n else None
    c["wrong_release_rate"] = c["wrong_release"] / c["anomalies"] if c["anomalies"] else None
    c["false_hold_rate"] = c["false_hold"] / c["good"] if c["good"] else None
    c["wrong_release_upper95"] = clopper_pearson(c["wrong_release"], c["anomalies"])[1]
    return c


def main():
    rows, _ = attach_features(include_heldout=True)
    # Existing (already-scored) development categories only for this script.
    existing = [r for r in rows if r.get("role") in ("router_train", "dev_audit", "calibration")
                and r["category"] not in AD_HELDOUT9]
    methods = {}

    for fam in ("mvtec_ad", "mvtec_loco"):
        tr = [r for r in existing if r["dataset"] == fam and r["role"] == "router_train"]
        cal = [r for r in existing if r["dataset"] == fam and r["role"] == "calibration"]
        if fam == "mvtec_ad":
            keysets = {
                "patch_conformal": None,
                "patch_clip_logistic": ["z_patch", "z_clip"],
                "oneclass_mahalanobis": None,
            }
        else:
            keysets = {
                "patch_conformal": None,
                "patch_clip_logistic": ["z_patch", "z_clip"],
                "oneclass_mahalanobis": None,
                "l1l2_semantic": ["z_patch", "z_clip", "z_sem"],
                "l1l2_full": ["z_patch", "z_clip", "z_sem", "z_fused", "z_aclip"],
                "l2_only_semantic": ["z_sem"],
            }
        for name, keys in keysets.items():
            if name == "patch_conformal":
                score_fn = patch_z_scores
            elif name == "oneclass_mahalanobis":
                score_fn = None
            else:
                model = fit_logistic(tr, keys, C=0.5)
                score_fn = partial(logistic_scores, model, keys)
            methods[(fam, name)] = {"keys": keys, "score_fn": score_fn, "cal": cal, "train_n": len(tr)}

    # Global prior baseline across both families (only categories seen in training).
    tr_all = [r for r in existing if r["role"] == "router_train"]
    cats = sorted({r["category"] for r in tr_all})
    prior_fn = prior_logistic(tr_all, cats)
    methods[("all", "prior_logistic")] = {
        "keys": None, "score_fn": partial(prior_scores, prior_fn),
        "cal": [r for r in existing if r["role"] == "calibration"], "train_n": len(tr_all)}

    eval_sets = {
        "mvtec_ad": [r for r in existing if r["dataset"] == "mvtec_ad" and r["role"] == "dev_audit"],
        "mvtec_loco": [r for r in existing if r["dataset"] == "mvtec_loco" and r["role"] == "dev_audit"],
        "all": [r for r in existing if r["role"] == "dev_audit"],
    }
    grid_rows = []
    rc_rows = []
    method_auc = []
    for (fam, name), spec in sorted(methods.items()):
        ev = eval_sets[fam]
        cal = spec["cal"]
        if name == "oneclass_mahalanobis":
            all_for_scores = ev + cal
            scores = mahalanobis_scores(all_for_scores, cal)
        else:
            scores = spec["score_fn"](ev + cal)
        pairs_eval = add_pvalues(ev, scores, cal)
        if not pairs_eval:
            continue
        y = np.array([r["truth_bin"] for r, _ in pairs_eval])
        p = np.array([p for _, p in pairs_eval], float)
        # For AUC, lower p == more anomalous
        s = -p
        method_auc.append({"family": fam, "method": name, "n": len(y),
                           "anomalies": int(y.sum()), "auc_neg_p": auc(y, s),
                           "average_precision_neg_p": average_precision(y, s),
                           "train_n": spec["train_n"]})
        # standard risk-coverage: alpha_hold=0 (release or human)
        for alpha in [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
            c = policy_counts(pairs_eval, alpha, 0.0)
            c.update(family=fam, method=name, alpha_release=alpha, alpha_hold=0.0)
            rc_rows.append(c)
        for ar, ah in itertools.product([0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                                        [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
            if ah >= ar:
                continue
            c = policy_counts(pairs_eval, ar, ah)
            c.update(family=fam, method=name, alpha_release=ar, alpha_hold=ah)
            grid_rows.append(c)
    # Complement baselines: never release, always release, hold everything.
    for fam, ev in eval_sets.items():
        n = len(ev)
        anom = sum(r["truth_bin"] for r in ev)
        good = n - anom
        grid_rows += [
            {"family": fam, "method": "always_release", "alpha_release": None, "alpha_hold": None,
             "release": n, "hold": 0, "human": 0, "wrong_release": anom, "false_hold": 0,
             "true_release": good, "anomalies_held": 0, "anomalies": anom, "good": good,
             "n": n, "autonomous_coverage": 1.0, "human_rate": 0.0,
             "wrong_release_rate": 1.0 if anom else None, "false_hold_rate": 0.0,
             "wrong_release_upper95": clopper_pearson(anom, anom)[1] if anom else None},
            {"family": fam, "method": "always_human", "alpha_release": None, "alpha_hold": None,
             "release": 0, "hold": 0, "human": n, "wrong_release": 0, "false_hold": 0,
             "true_release": 0, "anomalies_held": 0, "anomalies": anom, "good": good,
             "n": n, "autonomous_coverage": 0.0, "human_rate": 1.0,
             "wrong_release_rate": 0.0 if anom else None, "false_hold_rate": 0.0,
             "wrong_release_upper95": clopper_pearson(0, anom)[1] if anom else None},
            {"family": fam, "method": "hold_all", "alpha_release": None, "alpha_hold": None,
             "release": 0, "hold": n, "human": 0, "wrong_release": 0, "false_hold": good,
             "true_release": 0, "anomalies_held": anom, "anomalies": anom, "good": good,
             "n": n, "autonomous_coverage": 1.0, "human_rate": 0.0,
             "wrong_release_rate": 0.0 if anom else None, "false_hold_rate": 1.0,
             "wrong_release_upper95": clopper_pearson(0, anom)[1] if anom else None},
        ]
    write_csv("selective_grid.csv", grid_rows)
    write_csv("risk_coverage.csv", rc_rows)
    write_csv("method_auc.csv", method_auc)

    # Choose operating points: max autonomous coverage subject to safety constraints.
    constraints = {
        "zero_wrong_release": lambda r: r["wrong_release"] == 0,
        "upper95_wr_le_1pct": lambda r: r["wrong_release_upper95"] <= 0.01,
        "wr_rate_le_5pct": lambda r: r["wrong_release_rate"] <= 0.05,
        "wr_rate_le_10pct": lambda r: r["wrong_release_rate"] <= 0.10,
    }
    op_rows = []
    for fam in ("mvtec_ad", "mvtec_loco", "all"):
        for name in sorted({r["method"] for r in grid_rows if r["family"] == fam}):
            cand = [r for r in grid_rows if r["family"] == fam and r["method"] == name]
            for cname, fn in constraints.items():
                ok = [r for r in cand if fn(r)]
                if not ok:
                    continue
                best = max(ok, key=lambda r: (r["autonomous_coverage"], -r["human_rate"],
                                              -r["wrong_release"]))
                op_rows.append({**best, "constraint": cname})
    write_csv("operating_points.csv", op_rows)
    write_json("selective_metadata.json", {
        "scope": "existing v3.1 development categories; dev_audit is exposed development data",
        "calibration": "per-category GOOD calibration role only; conformal upper-tail p-value",
        "policy": "release if p>alpha_release; hold if p<alpha_hold; otherwise HUMAN",
        "unresolved": "HUMAN is never counted as autonomous or correct",
        "grid": {"alpha_release": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                 "alpha_hold": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]},
        "safety_upper": "one-sided 95% Clopper-Pearson upper bound on wrong-release proportion",
        "methods": {f"{k[0]}:{k[1]}": {"keys": v["keys"], "train_n": v["train_n"]}
                    for k, v in methods.items()},
        "caveat": "Conformal validity requires exchangeable calibration/test; MVTec categories and "
                  "the prior development exposure violate that assumption, so empirical results are descriptive.",
    })
    print(json.dumps({"grid_rows": len(grid_rows), "operating_points": len(op_rows),
                      "method_auc": len(method_auc)}, indent=2))


if __name__ == "__main__":
    main()
