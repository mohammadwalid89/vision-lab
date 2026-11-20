"""Closed-form Gaussian heads, the family the agents converged on in the v2 run.

The agents' final heads were written from class means and one pooled within-class
covariance, fitted on the fit and feedback images together:

    ncm     w = mu1 - mu0
    gda     w = P (mu1 - mu0),  P = D inv((N-1) S + tr(S) I)            (empirical-Bayes precision)
    shrink  w = P (mu1 - mu0),  P = inv((1-lam) S + (1+lam) delta I),   delta = tr(S)/D

with b = -w . (mu0 + mu1) / 2 (equal priors). `lam` was chosen per task by the agent
(cross-validation over the feedback images). This is a classical shrinkage discriminant, not
a new method: the agents' contribution was finding, tuning and safeguarding it per task.

The code below is the agent's own head function from its task scripts (cv.py), reduced to
what builds a final head. Usage:

    python gaussian_head.py TASK_DIR --kind shrink --lam 0.1 --out final.npz
"""
import argparse

import numpy as np


def gauss_head(X, y, lam=0.0, kind="shrink"):
    mu0, mu1 = X[y == 0].mean(0), X[y == 1].mean(0)
    N, D = X.shape
    if kind == "ncm":
        w = mu1 - mu0
    else:
        Xc = X - np.where(y[:, None] == 1, mu1, mu0)
        Sig = Xc.T @ Xc / (N - 2)
        if kind == "gda":
            P = D * np.linalg.inv((N - 1) * Sig + np.trace(Sig) * np.eye(D))
        else:
            delta = np.trace(Sig) / D
            P = np.linalg.inv((1 - lam) * Sig + (lam + 1) * delta * np.eye(D))
        w = P @ (mu1 - mu0)
    b = -w @ (mu0 + mu1) / 2
    return w, b


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("task_dir", help="folder with fit.npz and feedback.npz")
    ap.add_argument("--kind", choices=["ncm", "gda", "shrink"], default="shrink")
    ap.add_argument("--lam", type=float, default=0.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    F, V = np.load(f"{a.task_dir}/fit.npz"), np.load(f"{a.task_dir}/feedback.npz")
    X, y = np.vstack([F["X"], V["X"]]), np.concatenate([F["y"], V["y"]])
    w, b = gauss_head(X, y, a.lam, a.kind)
    np.savez(a.out, coef=w, intercept=np.float64(b))


if __name__ == "__main__":
    main()
