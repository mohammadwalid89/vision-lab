"""Computationally equivalent vectorized selector for Phase 2 preregistered experiment.

All score inputs, folds, objective, threshold grids, selection/tie order and fresh-data seal
remain in the original p2_policy.py. Only candidate threshold scoring is vectorized.
Regression against the unmodified selector: see out/p2_selector_regression.json.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import p2_policy as original


def grid_metrics(A, idx, ah):
    """Candidate release outcomes (one row per AR) on the given selection subset."""
    y = A["y"][idx]
    gate = A["gate"][idx]
    rel = gate[None, :] & (A["s_rel"][idx][None, :] > original.AR[:, None])
    hold = gate[None, :] & (A["s_hold"][idx][None, :] < ah) & ~rel
    g = y == 0
    a = ~g
    ng, nd = max(int(g.sum()), 1), max(int(a.sum()), 1)
    esc = np.count_nonzero(rel[:, a], axis=1)
    released_goods = np.count_nonzero(rel[:, g], axis=1)
    held_goods = np.count_nonzero(hold[:, g], axis=1)
    held_defects = np.count_nonzero(hold[:, a], axis=1)
    review_goods = int(g.sum()) - released_goods - held_goods
    review_defects = int(a.sum()) - esc - held_defects
    C = original.COST
    er, fr, hg, hd = esc / nd, held_goods / ng, review_goods / ng, review_defects / nd
    costs = 1000 * (C["pi"] * (er * C["ce"] + hd * C["ch"]) + (1 - C["pi"]) * (fr * C["cf"] + hg * C["ch"]))
    return esc, costs


def fast_select(A, idx, per_category):
    cats = np.unique(A["c"][idx])
    best = None
    if per_category:
        grouped = {c: idx[A["c"][idx] == c] for c in cats}
        for ah in original.AH:
            arm = {}
            for c, ii in grouped.items():
                esc, cost = grid_metrics(A, ii, ah)
                candidates = [
                    (float(cost[j]), -float(ar), float(ar)) for j, ar in enumerate(original.AR) if esc[j] == 0
                ]
                arm[str(c)] = min(candidates)[2] if candidates else 1.0
            arr = np.array([arm.get(str(c), 1.0) for c in A["c"]])
            o = original.outcome(A, original.decisions(A, arr, ah), idx)
            key = (o["escapes"] > 0, o["usd_per_1000"])
            pol = {"ar": arm, "ah": float(ah)}
            if best is None or key < best[0]:
                best = (key, pol)
    else:
        for ah in original.AH:
            esc, cost = grid_metrics(A, idx, ah)
            for j, ar in enumerate(original.AR):
                key = (bool(esc[j] > 0), float(cost[j]))
                if best is None or key < best[0]:
                    best = (key, {"ar": float(ar), "ah": float(ah)})
    return best[1]


def regression():
    from pathlib import Path

    cases = []
    for dataset in ("real_ad", "real_loco", "synthetic_loco"):
        if dataset == "real_ad":
            D = original.Data(fresh=False)
            A = D.arrays("ad", "wrn")
        elif dataset == "real_loco":
            D = original.Data(fresh=False)
            A = D.arrays("loco", "wrn_comp")
        else:
            rng = np.random.default_rng(54)
            n = 302
            categories = np.array(["breakfast_box", "pushpins", "screw_bag"])
            A = {
                "c": categories[np.arange(n) % 3],
                "y": rng.binomial(1, 0.44, n),
                "gate": rng.random(n) > 0.08,
                "s_rel": rng.uniform(0, 1, n),
                "s_hold": rng.uniform(0, 1, n),
            }
            A["stratum"] = np.where(A["y"] == 0, "good", "logical_anomalies")
        tests = list(original.splits(A, 8))
        for per in (False, True):
            for idx, _ in tests:
                old = original.select(A, idx, per)
                new = fast_select(A, idx, per)
                if old != new:
                    raise AssertionError({"dataset": dataset, "per_category": per, "old": old, "new": new})
                cases.append({"dataset": dataset, "per_category": per, "equal": True})
    res = {
        "compared": len(cases),
        "all_identical": True,
        "details": {
            "real_ad": sum(x["dataset"] == "real_ad" for x in cases),
            "synthetic_loco": sum(x["dataset"] == "synthetic_loco" for x in cases),
            "real_loco": sum(x["dataset"] == "real_loco" for x in cases),
        },
    }
    Path("out/p2_selector_regression.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--regression":
        print(regression())
    else:
        assert __import__("pathlib").Path("out/p2_selector_regression.json").exists(), "Run regression first"
        original.select = fast_select
        original.main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
