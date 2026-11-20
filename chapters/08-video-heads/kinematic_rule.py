#!/usr/bin/env python3
"""Check whether a hand-written motion rule on kinematic features can separate KTH classes.

The rule is written from the speed/cadence/amplitude thresholds below without
looking at labels. Before compiling it into weights, this script measures whether
the features carry the labels at all:

  features  background-subtracted silhouette features per clip: height variation,
            vertical bounce amplitude, horizontal speed, step cadence (all relative
            to body height)
  diag      per-class feature statistics, a 5-fold subject-grouped logistic probe on
            the features, and the accuracy of the declared rule
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
from kth_common import list_clips, metrics, subject_folds  # noqa: E402

FPS = 25.0
SPEC = {"speed_run": 0.260, "cad_run": 2.70, "amp_run": 0.090,
        "speed_walk": 0.130, "cad_walk": 2.20}
FEATS = ["h_cv", "amp", "speed", "cadence"]


def clip_features(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32))
    cap.release()
    if len(frames) < 8:
        return None
    F = np.stack(frames)
    fg = np.abs(F - np.median(F, axis=0)) > 18
    T, H, W = fg.shape
    area = fg.reshape(T, -1).sum(1)
    top_all = np.where(fg.any(2), np.arange(H)[None, :], H).min(1)
    bottom_all = np.where(fg.any(2), np.arange(H)[None, :], -1).max(1)
    left_all = np.where(fg.any(1), np.arange(W)[None, :], W).min(1)
    right_all = np.where(fg.any(1), np.arange(W)[None, :], -1).max(1)
    keep = area >= 20
    if keep.sum() < 6:
        return None
    h = (bottom_all - top_all + 1)[keep].astype(float)
    cx = ((left_all + right_all) / 2)[keep]
    cy = ((top_all + bottom_all) / 2)[keep]
    top = top_all[keep].astype(float)
    mean_h = h.mean()
    h_cv = float(h.std() / max(mean_h, 1e-6))
    amp = float((np.quantile(top, 0.95) - np.quantile(top, 0.05)) / max(mean_h, 1e-6))
    speed = float((np.abs(np.diff(cx)).mean() + np.abs(np.diff(cy)).mean()) / max(mean_h, 1e-6))

    # cadence: best autocorrelation lag of the detrended top-of-silhouette signal on the
    # longest contiguous run of kept frames
    idx = np.flatnonzero(keep)
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    run = max(runs, key=len)
    cadence = 0.0
    if len(run) >= 16:
        seg = top[np.isin(idx, run)]
        seg = seg - np.convolve(seg, np.ones(5) / 5, mode="same")
        lo, hi = max(2, int(round(FPS / 3.5))), min(len(seg) // 2, int(round(FPS / 0.5)))
        best_r, best_lag = -1.0, 0
        for lag in range(lo, max(lo + 1, hi)):
            a, b = seg[:-lag], seg[lag:]
            if a.std() < 1e-6 or b.std() < 1e-6:
                continue
            r = float(np.corrcoef(a, b)[0, 1])
            if r > best_r:
                best_r, best_lag = r, lag
        if best_r >= 0.30 and best_lag > 0:
            cadence = FPS / best_lag
    return [h_cv, amp, speed, float(cadence)]


def declared_rule(X: np.ndarray):
    """running (0) beats walking (1) beats jogging (2); returns labels and the in-domain mask."""
    s, c, a = X[:, 2], X[:, 3], X[:, 1]
    run = (s >= SPEC["speed_run"]) | ((c >= SPEC["cad_run"]) & (a >= SPEC["amp_run"]))
    walk = (s <= SPEC["speed_walk"]) & (c <= SPEC["cad_walk"])
    in_domain = ~((s <= SPEC["speed_walk"]) & (c >= SPEC["cad_run"]))
    return np.where(run, 0, np.where(walk, 1, 2)), in_domain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--clips-dir", type=Path, default=Path(os.environ.get("KTH_CLIPS_DIR", "kth")))
    parser.add_argument("--out", type=Path, default=Path("results/kinematic_rule.json"))
    args = parser.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rows = list_clips(args.clips_dir)
    feats, labels, subjects = [], [], []
    for r in rows:
        f = clip_features(r["path"])
        if f is not None:
            feats.append(f)
            labels.append(r["label"])
            subjects.append(r["subject"])
    X, y, subj = np.asarray(feats, float), np.array(labels), np.array(subjects)

    oof = np.zeros(len(y), int)
    for fold in subject_folds(subj):
        test = np.isin(subj, fold)
        scaler = StandardScaler().fit(X[~test])
        clf = LogisticRegression(C=1, max_iter=2000).fit(scaler.transform(X[~test]), y[~test])
        oof[test] = clf.predict(scaler.transform(X[test]))
    rule_pred, in_domain = declared_rule(X)
    result = {
        "n": int(len(X)), "features": FEATS, "spec": SPEC,
        "class_means": {c: dict(zip(FEATS, np.round(X[y == c].mean(0), 4).tolist())) for c in (0, 1, 2)},
        "logreg_probe_cv": metrics(y, oof, subj),
        "spec_on_all_rows": {"acc": float((rule_pred == y).mean()),
                             "domain_violations": int((~in_domain).sum())},
        "cadence_median_hz": float(np.median(X[:, 3])),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print("probe accuracy", round(result["logreg_probe_cv"]["accuracy"], 4),
          "| declared rule accuracy", round(result["spec_on_all_rows"]["acc"], 4),
          "| median cadence Hz", result["cadence_median_hz"])


if __name__ == "__main__":
    main()
