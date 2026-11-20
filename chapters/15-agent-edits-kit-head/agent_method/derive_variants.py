"""Step 1 of the agent's final edit: locate the product variants among normal kits.

The agent found that juice-bottle normals form three tight clusters in the plane of
the top two principal components. This script reproduces that clustering (k-means,
k=3, 20 random restarts, seed 1) and saves the plane and the centres for build_head.py.
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", required=True, help="folder holding edit_fit.npz")
    ap.add_argument("--out", required=True, help="output .npz (mu, Vt2, C)")
    a = ap.parse_args()

    f = np.load(Path(a.sandbox) / "edit_fit.npz")
    X, roles = f["X"].astype(float), f["roles"]
    N = X[roles == "fit_normal"]
    mu = N.mean(0)
    _, _, Vt = np.linalg.svd(N - mu, full_matrices=False)
    pn = (N - mu) @ Vt[:2].T

    rng = np.random.default_rng(1)
    best = None
    for _ in range(20):
        C = pn[rng.choice(len(pn), 3, replace=False)]
        for _ in range(50):
            assign = np.argmin(((pn[:, None] - C[None]) ** 2).sum(-1), 1)
            C = np.array([pn[assign == j].mean(0) for j in range(3)])
        cost = ((pn - C[assign]) ** 2).sum()
        if best is None or cost < best[0]:
            best = (cost, C)
    np.savez(a.out, C=best[1], Vt2=Vt[:2], mu=mu)


if __name__ == "__main__":
    main()
