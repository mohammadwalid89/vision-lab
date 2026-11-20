"""Prior-unseen evaluation on the nine new AD categories.

The 225 held-out test images overlap exactly with an earlier surface experiment
(`results/mvtec_ad_surface.json`).  This script uses only images that were NOT
in that prior run: 10 calibration normals for testing, 10 other calibration
normals for thresholding, and the 10 train-exception anomalies per category.
This is the cleanest available prior-unseen check.
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
    write_csv,
    write_json,
)
from features import attach_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def fit(rows, keys, C=0.5):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    y = np.array([r["truth_bin"] for r in rows])
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced",
                                            C=C)).fit(X, y)


def score(model, rows, keys):
    X = np.array([[r[k] for k in keys] for r in rows], float)
    return model.predict_proba(X)[:, 1]


def eval_threshold(rows, s, thresholds, y):
    pred = np.array([int(v > thresholds[r["category"]]) for r, v in zip(rows, s)])
    return {
        "n": len(rows), "anomalies": int(y.sum()),
        "auc": auc(y, s), "average_precision": average_precision(y, s),
        "balanced_accuracy": balanced_accuracy(y, pred),
        "recall_at_q95": float(pred[y == 1].mean()) if (y == 1).any() else None,
        "fpr_at_q95": float(pred[y == 0].mean()) if (y == 0).any() else None,
        "wrong_release": int(((pred == 0) & (y == 1)).sum()),
        "false_hold": int(((pred == 1) & (y == 0)).sum()),
    }


def thresholds(cal, s):
    by = defaultdict(list)
    for r, v in zip(cal, s):
        by[r["category"]].append(v)
    return {k: float(np.quantile(v, 0.95)) for k, v in by.items()}


def main():
    rows, _ = attach_features(include_heldout=True)
    held = [r for r in rows if r["category"] in AD_HELDOUT9 and r["dataset"] == "mvtec_ad"]
    ad6 = [r for r in rows if r["category"] in AD_CATS6 and r["dataset"] == "mvtec_ad"]
    train6 = [r for r in ad6 if r["role"] == "router_train"]
    cal6 = [r for r in ad6 if r["role"] == "calibration"]
    # Prior-unseen sets per category.
    cal_by = defaultdict(list)
    sup_by = defaultdict(list)
    for r in held:
        if r["role"] == "calibration":
            cal_by[r["category"]].append(r)
        elif r["role"] == "support_anomaly":
            sup_by[r["category"]].append(r)
    good, bad, threshold_rows = [], [], []
    for cat in AD_HELDOUT9:
        cal = sorted(cal_by[cat], key=lambda r: r["sha256"])
        sup = sorted(sup_by[cat], key=lambda r: r["sha256"])
        threshold_rows += cal[:10]
        good += cal[10:]
        bad += sup
    test = good + bad
    y = np.array([r["truth_bin"] for r in test])
    out = []
    # Raw AD6 threshold
    t_raw = float(np.quantile([r["patch_score"] for r in cal6], 0.95))
    s = np.array([r["patch_score"] for r in test])
    pred = (s > t_raw).astype(int)
    out.append({"method": "raw_patch_ad6_q95", "n": len(test), "anomalies": int(y.sum()),
                "auc": auc(y, s), "average_precision": average_precision(y, s),
                "balanced_accuracy": balanced_accuracy(y, pred),
                "recall_at_q95": float(pred[y == 1].mean()),
                "fpr_at_q95": float(pred[y == 0].mean()),
                "wrong_release": int(((pred == 0) & (y == 1)).sum()),
                "false_hold": int(((pred == 1) & (y == 0)).sum())})
    # Per-category conformal
    s = np.array([r["z_patch"] for r in test])
    cs = np.array([r["z_patch"] for r in threshold_rows])
    th = thresholds(threshold_rows, cs)
    out.append({"method": "per_category_conformal_patch", **eval_threshold(test, s, th, y)})
    # Global models trained on AD6
    for name, keys in [("global_logistic_z", ["z_patch", "z_clip"]),
                       ("global_logistic_raw", ["patch_score", "clip_score"])]:
        model = fit(train6, keys)
        s = score(model, test, keys)
        cs = score(model, threshold_rows, keys)
        th = thresholds(threshold_rows, cs)
        out.append({"method": name, **eval_threshold(test, s, th, y)})
    write_csv("prior_unseen_eval.csv", out)
    write_json("prior_unseen_metadata.json", {
        "purpose": "evaluate on images not present in the prior surface run",
        "prior_run": "results/mvtec_ad_surface.json / results/runs/afa...json",
        "test": "90 good calibration normals + 90 train-exception anomalies (10+10 per category)",
        "threshold": "the other 10 calibration normals per category",
        "not_used": "the 225 test images that overlap the prior surface run",
        "caveat": "still MVTec collection; no physical unit IDs; 10 normals per threshold",
    })
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
