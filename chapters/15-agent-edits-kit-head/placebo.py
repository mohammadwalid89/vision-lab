"""Placebo for the agent: random edits of the same kind, same 50-call budget.

Each draw rescales 40 random weights and adds a small random direction; a draw is kept
if it raises AUC on edit_val (the agent's own feedback set).
"""
import argparse
from pathlib import Path

import numpy as np

BUDGET = 50
SEED = 7


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", required=True, help="sandbox_<category> folder")
    ap.add_argument("--out", required=True, help="output .npz")
    a = ap.parse_args()

    sb = Path(a.sandbox)
    v = np.load(sb / "edit_val.npz")
    X, y = v["X"].astype(np.float64), v["y"]
    c = np.load(sb / "start_head.npz")
    w, b = c["coef"].copy(), float(c["intercept"])
    best = auc(X @ w + b, y)
    rng = np.random.default_rng(SEED)
    for _ in range(BUDGET):
        w2 = w.copy()
        k = rng.choice(len(w), 40, replace=False)
        w2[k] *= np.exp(rng.normal(0, .5, 40))
        w2 += rng.normal(0, 1, len(w)) * np.linalg.norm(w) / np.sqrt(len(w)) * .1
        score = auc(X @ w2 + b, y)
        if score > best:
            best, w = score, w2
    np.savez_compressed(a.out, coef=w, intercept=np.float64(b))
    print("placebo best edit_val auc", round(best, 4))


if __name__ == "__main__":
    main()
