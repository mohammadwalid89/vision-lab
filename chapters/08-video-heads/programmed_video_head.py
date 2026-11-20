#!/usr/bin/env python3
"""Closed-form linear head on frozen R3D-18 features, against trained and random heads.

Features are the 512-d pooled outputs of the frozen Kinetics-400 R3D-18 on 16
frames spread over each KTH clip. Each head has 512 x 3 + 3 = 1,539 parameters.
Five-fold subject-grouped out-of-fold predictions on all 300 clips:

  LDA      class means and pooled covariance with a fixed ridge of 1.0; weights are
           computed in closed form, no optimizer
  TRAINED  logistic regression on the same labels (gradient based)
  PROTO    nearest class centroid
  RANDOM   a fixed random head, no labels

Stages:
  features  run the frozen R3D-18 over the cached clips (temporal_lowpass.py cache)
  heads     fit and compare the four heads from the saved features
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from kth_common import (  # noqa: E402
    bootstrap_subject_delta,
    build_model,
    list_clips,
    metrics,
    pooled_features,
    subject_folds,
)

torch.set_num_threads(1)
RIDGE = 1.0


def fit_heads(F: np.ndarray, y: np.ndarray, subj: np.ndarray) -> dict[str, np.ndarray]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n = len(y)
    out = {k: np.zeros(n, int) for k in ("proto", "lda", "trained")}
    rng = np.random.default_rng(0)
    W_rand = rng.normal(0, 1 / np.sqrt(F.shape[1]), (3, F.shape[1]))
    b_rand = rng.normal(0, 0.1, 3)
    out["random"] = (F @ W_rand.T + b_rand).argmax(1)
    for fold in subject_folds(subj):
        test = np.isin(subj, fold)
        train = ~test
        mu, sd = F[train].mean(0), F[train].std(0) + 1e-6
        Ztr, Zte = (F[train] - mu) / sd, (F[test] - mu) / sd
        cents = np.stack([Ztr[y[train] == c].mean(0) for c in (0, 1, 2)])
        out["proto"][test] = (((Zte[:, None, :] - cents[None, :, :]) ** 2).sum(2)).argmin(1)
        # LDA: pooled within-class covariance plus a fixed ridge
        centred = Ztr - cents[y[train]]
        S = centred.T @ centred / max(len(Ztr) - 3, 1) + RIDGE * np.eye(Ztr.shape[1])
        S_inv = np.linalg.pinv(S)
        W = cents @ S_inv
        bias = -0.5 * np.einsum("cd,dc->c", cents @ S_inv, cents.T)
        out["lda"][test] = (Zte @ W.T + bias).argmax(1)
        scaler = StandardScaler().fit(Ztr)
        clf = LogisticRegression(C=1, max_iter=3000).fit(scaler.transform(Ztr), y[train])
        out["trained"][test] = clf.predict(scaler.transform(Zte))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["features", "heads"])
    parser.add_argument("--clips-dir", type=Path, default=Path(os.environ.get("KTH_CLIPS_DIR", "kth")))
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--r3d-weights", type=Path, default=None)
    parser.add_argument("--features", type=Path, default=None,
                        help="npz with F, y, subj, ids (default: <cache-dir>/r3d_features.npz)")
    parser.add_argument("--out", type=Path, default=Path("results/programmed_head.json"))
    args = parser.parse_args()

    feature_file = args.features or args.cache_dir / "r3d_features.npz"
    if args.stage == "features":
        rows = list_clips(args.clips_dir)
        X = np.load(args.cache_dir / "frac100.npz")["X"]
        F = pooled_features(build_model(args.r3d_weights), X)
        np.savez_compressed(feature_file, F=F, y=np.array([r["label"] for r in rows]),
                            subj=np.array([r["subject"] for r in rows]),
                            ids=np.array([r["name"] for r in rows]))
        print("r3d features", F.shape)
        return

    z = np.load(feature_file)
    F, y, subj = z["F"], z["y"], z["subj"]
    pred = fit_heads(F, y, subj)
    correct = {k: (v == y).astype(float) for k, v in pred.items()}
    result = {
        "split": "5-fold subject-grouped CV", "n": int(len(y)), "feature_dim": int(F.shape[1]),
        "head_parameters": int(F.shape[1] * 3 + 3), "ridge": RIDGE,
        "heads": {name: {**metrics(y, pred[name], subj),
                         "vs_trained": None if name == "trained"
                         else bootstrap_subject_delta(subj, correct[name], correct["trained"])}
                  for name in ("proto", "trained", "lda", "random")},
        "lda_vs_trained_cases": {"lda_right": int(correct["lda"].sum()),
                                 "trained_right": int(correct["trained"].sum()),
                                 "lda_only_correct": int(((correct["lda"] == 1) & (correct["trained"] == 0)).sum()),
                                 "trained_only_correct": int(((correct["lda"] == 0) & (correct["trained"] == 1)).sum())},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for name, m in result["heads"].items():
        print(f"{name:8s} acc={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f} vs_trained={m['vs_trained']}")


if __name__ == "__main__":
    main()
