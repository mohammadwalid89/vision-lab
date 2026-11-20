#!/usr/bin/env python3
"""Temporal low-pass edit of R3D-18 kernels on KTH walking / jogging / running.

Every Conv3d with a temporal kernel of 3 is rewritten as W'[:, :, t] = sum_j M[t, j] W[:, :, j],
where M is a 3x3 matrix along time. Arms:
  ORIG       unedited model
  LP001/02/10  low-pass M with a = 0.001 / 0.02 / 0.10
  PLACEBO02  random rotation of the time axis with ||M - I||_F matched to LP02
  SHARP02    high-pass M with a = 0.02
Each arm's 512-d pooled features go to the same logistic-regression probe under
5-fold subject-grouped cross-validation, so every clip is scored by a probe that
never saw its subject.

Stages (run in this order):
  cache      decode 16-frame clips over spans 1.0 / 0.5 / 0.25 of each clip to .npz
  preflight  measure how far the edit moves features at three doses (dead-actuator gate)
  run        the paired arms above
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
import torch.nn as nn  # noqa: E402
from kth_common import (  # noqa: E402
    bootstrap_subject_delta,
    build_model,
    list_clips,
    metrics,
    pooled_features,
    sample_frames,
    subject_folds,
)

torch.set_num_threads(1)

ARMS = {"ORIG": None, "LP001": (0.001, "lp", 0), "LP02": (0.02, "lp", 0), "LP10": (0.10, "lp", 0),
        "PLACEBO02": (0.02, "placebo", 7), "SHARP02": (0.02, "sharp", 0)}


def temporal_matrix(a: float, kind: str = "lp", seed: int = 0) -> np.ndarray:
    low_pass = np.array([[1 - 2 * a, a, 0.0], [a, 1 - 2 * a, a], [0.0, a, 1 - 2 * a]], np.float32)
    if kind == "lp":
        return low_pass
    if kind == "sharp":
        s = abs(a)
        return np.array([[1 + 2 * s, -s, 0.0], [-s, 1 + 2 * s, -s], [0.0, -s, 1 + 2 * s]], np.float32)
    # placebo: rotation about a random axis, angle chosen by bisection so that
    # ||R - I||_F equals ||low_pass - I||_F
    from scipy.linalg import expm

    rng = np.random.default_rng(seed)
    v = rng.standard_normal(3)
    v /= np.linalg.norm(v)
    target = np.linalg.norm(low_pass - np.eye(3))
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], np.float32)
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if np.linalg.norm(expm(mid * K) - np.eye(3)) < target:
            lo = mid
        else:
            hi = mid
    return expm((lo + hi) / 2 * K).astype(np.float32)


def apply_temporal_edit(model, M: np.ndarray) -> dict:
    """Edit every temporal-kernel-3 Conv3d in place; return exact accounting."""
    Mt = torch.as_tensor(M, dtype=torch.float32)
    stats = {"tensors": 0, "entries_touched": 0, "delta_l2": 0.0, "base_l2": 0.0, "max_rel": 0.0}
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, mod in modules.items():
            if isinstance(mod, nn.Conv3d) and mod.weight.shape[2] == 3:
                w = mod.weight.detach().clone()
                new = torch.einsum("tj,oijhw->oithw", Mt, w)
                d = new - w
                stats["tensors"] += 1
                stats["entries_touched"] += int((d != 0).sum())
                stats["delta_l2"] += float(d.norm())
                stats["base_l2"] += float(w.norm())
                stats["max_rel"] = max(stats["max_rel"], float(d.norm() / w.norm().clamp_min(1e-12)))
                mod.weight.copy_(new)
    return stats


def probe(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=3000).fit(scaler.transform(Xtr), ytr)
    return clf.predict(scaler.transform(Xte))


def stage_cache(rows, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for frac in (1.0, 0.5, 0.25):
        out = cache_dir / f"frac{int(frac * 100)}.npz"
        if out.exists():
            print("exists", out.name)
            continue
        X = np.stack([sample_frames(r["path"], frac=frac) for r in rows])
        np.savez_compressed(out, X=X)
        print("cached", out.name, X.shape, flush=True)


def stage_preflight(cache_dir: Path, weights: Path | None, n_clips: int) -> dict:
    X = np.load(cache_dir / "frac100.npz")["X"][:n_clips]
    f0 = pooled_features(build_model(weights), X)
    doses = {}
    for a in (0.02, 0.10, 0.25):
        model = build_model(weights)
        stats = apply_temporal_edit(model, temporal_matrix(a, "lp"))
        shift = np.linalg.norm(pooled_features(model, X) - f0, axis=1)
        doses[str(a)] = {"weight_stats": stats, "feature_shift_mean": float(shift.mean()),
                         "feature_shift_min": float(shift.min()),
                         "feature_rel_shift": float(shift.mean() / np.linalg.norm(f0, axis=1).mean())}
        print("dose", a, doses[str(a)], flush=True)
    vals = [doses[str(a)]["feature_rel_shift"] for a in (0.02, 0.10, 0.25)]
    monotone = bool(vals[0] <= vals[1] <= vals[2])
    return {"doses": doses,
            "gate": {"monotone": monotone, "rel_shift_at_0.25": vals[2],
                     "passed": bool(vals[2] >= 0.005 and monotone)}}


def stage_run(rows, cache_dir: Path, weights: Path | None) -> dict:
    X = np.load(cache_dir / "frac100.npz")["X"]
    y = np.array([r["label"] for r in rows])
    subj = np.array([r["subject"] for r in rows])
    folds = subject_folds(subj)
    result = {"split": "5-fold subject-grouped CV (folds fixed before the run)",
              "window": "full-clip span (frac=1.0)", "arms": {}}
    base_correct = None
    for arm, spec in ARMS.items():
        model = build_model(weights)
        weight_edit = None
        if spec is not None:
            a, kind, seed = spec
            M = temporal_matrix(a, kind, seed)
            weight_edit = {**apply_temporal_edit(model, M), "matrix": M.tolist()}
        F = pooled_features(model, X)
        oof = np.zeros(len(rows), int)
        for fold in folds:
            test = np.isin(subj, fold)
            oof[test] = probe(F[~test], y[~test], F[test])
        m = metrics(y, oof, subj)
        correct = (oof == y).astype(float)
        if arm == "ORIG":
            base_correct = correct
        vs_orig = None if arm == "ORIG" else bootstrap_subject_delta(subj, correct, base_correct)
        result["arms"][arm] = {"weight_edit": weight_edit, "probe": m, "vs_orig": vs_orig}
        print(arm, "oof_acc", round(m["accuracy"], 4), "f1", round(m["macro_f1"], 4),
              "delta", None if vs_orig is None else round(vs_orig["delta"], 4), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("stage", choices=["cache", "preflight", "run"])
    parser.add_argument("--clips-dir", type=Path, default=Path(os.environ.get("KTH_CLIPS_DIR", "kth")),
                        help="folder with the extracted KTH .avi files")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--r3d-weights", type=Path, default=None,
                        help="local torchvision r3d_18 Kinetics-400 state dict; default downloads via torchvision")
    parser.add_argument("--preflight-clips", type=int, default=16)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = list_clips(args.clips_dir)
    if not rows:
        raise SystemExit("no KTH clips found under --clips-dir")
    if args.stage == "cache":
        stage_cache(rows, args.cache_dir)
        return
    if args.stage == "preflight":
        result = stage_preflight(args.cache_dir, args.r3d_weights, args.preflight_clips)
    else:
        result = stage_run(rows, args.cache_dir, args.r3d_weights)
    out = args.out or Path(f"results/lowpass_{args.stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
