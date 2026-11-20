"""Step 2 of the agent's final edit: write the head's weights.

Takes the starting logistic head and adds one dense direction built from three
labelled fit-set faults (rows 364, 381, 372 of edit_fit.npz), after removing from it
everything that normal kits vary along: the variant plane, the variant-centre
differences and the top-k within-variant normal principal components. Fit-set faults
named in --protect are projected out so they are not pushed down. The intercept is
shifted so the mean score of fit normals is unchanged.

The agent's final head used: --alpha 2.0 --protect 387,379
"""
import argparse
from pathlib import Path

import numpy as np

K_PCS = 20
FAULTS_A = (364, 381)   # one mismatch type; averaged
FAULT_B = 372           # the mirror type


def unit(u):
    return u / np.linalg.norm(u)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sandbox", required=True, help="folder with edit_fit.npz and start_head.npz")
    ap.add_argument("--variants", required=True, help="output of derive_variants.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--protect", default="", help="comma-separated fit rows to protect")
    a = ap.parse_args()

    sb = Path(a.sandbox)
    f = np.load(sb / "edit_fit.npz")
    h = np.load(sb / "start_head.npz")
    X, roles = f["X"].astype(float), f["roles"]
    w0, b0 = h["coef"].copy(), float(h["intercept"])
    q = np.load(a.variants)
    mu, P, C = q["mu"], q["Vt2"], q["C"]

    def assign(A):
        proj = (A - mu) @ P.T
        return np.argmin(((proj[:, None] - C[None]) ** 2).sum(-1), 1)

    is_normal = roles == "fit_normal"
    N = X[is_normal]
    va = assign(N)
    M = np.array([N[va == k].mean(0) for k in range(3)])
    R = N - M[va]
    _, _, Vt = np.linalg.svd(R, full_matrices=False)

    def deproj(u, k):
        Q, _ = np.linalg.qr(np.vstack([P, M[0] - M[1], M[2] - M[1], Vt[:k]]).T)
        return u - Q @ (Q.T @ u)

    m12 = (M[1] + M[2]) / 2
    ra = (X[list(FAULTS_A)] - m12).mean(0)
    rb = X[FAULT_B] - m12
    aa = unit(deproj(ra, K_PCS))
    u = deproj(rb, K_PCS)
    u = u - (u @ aa) * aa
    protect = [int(t) for t in a.protect.split(",") if t]
    if protect:
        E = np.array([deproj(X[i] - M[assign(X[[i]])[0]], K_PCS) for i in protect]).T
        Q, _ = np.linalg.qr(E)
        u = u - Q @ (Q.T @ u)

    w = w0 + a.alpha * u
    b = b0 - ((N @ w).mean() - (N @ w0).mean())
    np.savez(a.out, coef=w, intercept=np.float64(b))
    print(f"relative L2 change {np.linalg.norm(w - w0) / np.linalg.norm(w0):.3f}")


if __name__ == "__main__":
    main()
