"""Ground-truth effect calls: significance AND minimum effect ("Weight-Space Reading" rule).

A summary of Kocabay (2026), not checked against the paper, says ground truth for "did behaviour change"
was Welch p < 0.01 *plus* a minimum effect size, checked against six no-op controls, and
that readers still predicted change for 53-88 % of edits that changed nothing.

We use the rule in two places:
1. **Effect calls** reported next to every signed gate: an arm-vs-A difference is called
   UP / DOWN only if p < alpha AND |effect| >= min_effect; otherwise SAME.
2. **No-op integrity**: every function-preserving no-op control must be called SAME on
   every metric (and have max |score change| <= tol).  A no-op called UP/DOWN means the
   harness, not the model, is producing effects -> attempt is INCONCLUSIVE.

Test choice.  Welch's unequal-variance t-test assumes independent samples.  DEV metrics
for an arm and A are computed on the *same* source groups, so the default here is a
two-sided **paired sign-flip permutation test on per-source differences** (the scorer's
independence unit is the source group).  ``welch`` is provided for genuinely unpaired
comparisons (e.g. independent seeds/draws, or the WSR setting).  Seeds are not
independent datasets: three seeds on the same DEV sources do not triple the sample size.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

ALPHA = 0.01
# Declared minimum effects (absolute, in the metric's own units).  Rate metrics use the
# signed clean tolerance (.02); the marker flip rate uses gate 5's .05 flip advantage.
# These are declared values; change them only in the locked plan, before any DEV score.
MIN_EFFECT = {
    "marker_flip_rate": 0.05,
    "recall": 0.02,
    "FPR": 0.02,
    "conflict_FPR": 0.02,
    "clean_score": 0.10,   # logits; used for score-level no-op checks and diagnostics
}


def welch(a, b) -> dict:
    """Welch two-sample t-test (two-sided).  Uses scipy if present, else a t-sf series."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("need >= 2 observations per group")
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    diff = float(a.mean() - b.mean())
    se2 = va + vb
    if se2 == 0:
        return dict(effect=diff, t=math.inf if diff else 0.0, df=float(len(a) + len(b) - 2),
                    p=0.0 if diff else 1.0, test="welch")
    t = diff / math.sqrt(se2)
    df = se2 ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    try:
        from scipy import stats
        p = float(2 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover - scipy is in the project venv
        p = float(math.erfc(abs(t) / math.sqrt(2)))  # normal approximation
    return dict(effect=diff, t=float(t), df=float(df), p=p, test="welch")


def paired_signflip(diffs, n_perm: int = 20000, seed: int = 0) -> dict:
    """Two-sided paired sign-flip permutation test of mean(diffs) == 0.

    Exact enumeration when the number of non-zero differences is <= 16.
    """
    d = np.asarray(diffs, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n == 0:
        raise ValueError("no paired units")
    obs = abs(d.mean())
    nz = d[d != 0]
    if len(nz) == 0:
        return dict(effect=0.0, p=1.0, n=n, test="paired_signflip", exact=True)
    if len(nz) <= 16:
        signs = ((np.arange(2 ** len(nz))[:, None] >> np.arange(len(nz))) & 1) * 2 - 1
        stats_ = np.abs((signs * nz).sum(1)) / n
        p = float(np.mean(stats_ >= obs - 1e-15))
        return dict(effect=float(d.mean()), p=p, n=n, test="paired_signflip", exact=True)
    rng = np.random.default_rng(seed)
    hits = 0
    done = 0
    batch = 2000
    while done < n_perm:
        m = min(batch, n_perm - done)
        signs = rng.choice((-1.0, 1.0), size=(m, len(nz)))
        hits += int((np.abs(signs @ nz) / n >= obs - 1e-15).sum())
        done += m
    p = (hits + 1) / (n_perm + 1)
    return dict(effect=float(d.mean()), p=float(p), n=n, test="paired_signflip", exact=False)


def call(effect: float, p: float, min_effect: float, alpha: float = ALPHA) -> str:
    """UP / DOWN only when significant AND at least the minimum effect; else SAME."""
    if p < alpha and abs(effect) >= min_effect - 1e-12:
        return "UP" if effect > 0 else "DOWN"
    return "SAME"


def paired_metric_calls(arm: Mapping, ref: Mapping, alpha: float = ALPHA,
                        min_effect: Mapping[str, float] = MIN_EFFECT, seed: int = 0) -> dict:
    """Effect calls arm - ref on per-source paired units (outputs of metrics.paired_units).

    Metrics: marker_flip_rate (all sources), recall (defect sources' clean view),
    FPR (good sources' clean view), conflict_FPR (good sources' marker view),
    clean_score (mean clean logit change).
    """
    if arm["groups"] != ref["groups"]:
        raise ValueError("arm and reference must cover identical source groups")
    lab = ref["label"]
    pieces = {
        "marker_flip_rate": arm["flip"] - ref["flip"],
        "recall": (arm["clean_pred"] - ref["clean_pred"])[lab == 1],
        "FPR": (arm["clean_pred"] - ref["clean_pred"])[lab == 0],
        "conflict_FPR": (arm["conflict"] - ref["conflict"])[lab == 0],
        "clean_score": arm["clean_score"] - ref["clean_score"],
    }
    out = {}
    for k, d in pieces.items():
        if len(d) == 0:
            continue
        t = paired_signflip(d, seed=seed)
        out[k] = dict(effect=t["effect"], p=t["p"], n=t["n"], min_effect=min_effect[k],
                      call=call(t["effect"], t["p"], min_effect[k], alpha))
    return out


def noop_false_positive_rate(calls_per_noop: Mapping[str, Mapping[str, Mapping]]) -> dict:
    """Fraction of (no-op, metric) cells called UP/DOWN.  Must be 0 for a valid harness."""
    cells = [(n, m, c["call"]) for n, mm in calls_per_noop.items() for m, c in mm.items()]
    bad = [(n, m) for n, m, c in cells if c != "SAME"]
    return dict(n_cells=len(cells), n_false_positive=len(bad),
                rate=(len(bad) / len(cells)) if cells else 0.0, offenders=bad)
