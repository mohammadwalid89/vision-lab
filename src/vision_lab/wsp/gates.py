"""Signed R1/R2 gate arithmetic, carried forward unchanged from R1/R2.

| Gate | per-seed condition |
| 1a | relative flip reduction vs A >= .50 (zero A flip -> zero improvement)          |
| 1b | absolute flip reduction - 2 x strongest non-negative placebo reduction >= 0    |
| 2  | Recall(cand) - Recall(A) >= -.02                                               |
| 3  | FPR(A) - FPR(cand) >= -.02                                                     |
| 4  | relative marked-good FP reduction >= .30 (zero baseline -> zero improvement)   |
| 5  | vs EACH warm/fresh matched-k retrainer: flip advantage >= .05, AUROC diff >= -.01 |
| 6  | strict flip improvement in all three seeds                                     |
| 7  | OWN: |AUROC - AUROC(B_k)| <= .02 and strictly lower flip than B_k              |

Aggregation: worst seed per gate; means never rescue a failing seed (``evaluate``).
Gate families: CORE = 1a,1b,2,3,4,6 (every slot); G5 only for a retraining-superiority
claim; G7 only where the plan declares it applicable, never as a substitute for CORE.
The arithmetic matches R1 ``dev_bars.seed_bars`` (``V1``/``V2``/``OWN``); tests pin it.
"""
from __future__ import annotations

from typing import Mapping, Sequence

TOL = 1e-12

BARS = {
    "1a_marker_relative_reduction": 0.50,
    "1b_placebo_margin": 0.0,
    "2_recall_margin": -0.02,
    "3_FPR_margin": -0.02,
    "4_conflict_relative_reduction": 0.30,
    "6_same_direction_all_seeds": 1.0,
    "5_retrain_flip_advantage": 0.05,
    "5_retrain_auroc_margin": -0.01,
    "7_own_auroc_distance_margin": 0.0,   # .02 - |AUROC - AUROC_B| >= 0
    "7_own_lower_flip_all_seeds": 1.0,
}
CORE = ("1a_marker_relative_reduction", "1b_placebo_margin", "2_recall_margin",
        "3_FPR_margin", "4_conflict_relative_reduction", "6_same_direction_all_seeds")
G5 = ("5_retrain_flip_advantage", "5_retrain_auroc_margin")
G7 = ("7_own_auroc_distance_margin", "7_own_lower_flip_all_seeds", "2_recall_margin", "3_FPR_margin")


def relative_reduction(baseline: float, value: float) -> float:
    return 0.0 if baseline == 0 else (baseline - value) / baseline


def best_placebo_reduction(a: Mapping, placebos: Sequence[Mapping]) -> float:
    """Strongest NON-NEGATIVE absolute flip reduction across the declared placebo grid."""
    return max([0.0] + [a["marker_flip_rate"] - p["marker_flip_rate"] for p in placebos])


def seed_values(cand: Mapping, a: Mapping, placebos: Sequence[Mapping] = (),
                retrainers: Mapping[str, Mapping] | None = None, b: Mapping | None = None,
                families: Sequence[str] = ("CORE",)) -> dict:
    """Oriented gate values for one seed (higher is better; compare with BARS)."""
    v: dict = {}
    red = a["marker_flip_rate"] - cand["marker_flip_rate"]
    if "CORE" in families or "G7" in families:
        v["2_recall_margin"] = cand["recall"] - a["recall"]
        v["3_FPR_margin"] = a["FPR"] - cand["FPR"]
    if "CORE" in families:
        v["1a_marker_relative_reduction"] = relative_reduction(a["marker_flip_rate"], cand["marker_flip_rate"])
        v["1b_placebo_margin"] = red - 2 * best_placebo_reduction(a, placebos)
        v["4_conflict_relative_reduction"] = relative_reduction(a["conflict_set_FP"], cand["conflict_set_FP"])
        v["6_same_direction_all_seeds"] = float(red > 0)
    if "G5" in families:
        if not retrainers:
            raise ValueError("gate 5 needs the warm and fresh matched-k retrainers")
        # against EACH retrainer: take the worst
        v["5_retrain_flip_advantage"] = min(r["marker_flip_rate"] - cand["marker_flip_rate"] for r in retrainers.values())
        v["5_retrain_auroc_margin"] = min(cand["AUROC"] - r["AUROC"] for r in retrainers.values())
    if "G7" in families:
        if b is None:
            raise ValueError("gate 7 needs B_k")
        v["7_own_auroc_distance_margin"] = 0.02 - abs(cand["AUROC"] - b["AUROC"])
        v["7_own_lower_flip_all_seeds"] = float(cand["marker_flip_rate"] < b["marker_flip_rate"])
    return v


def impossibility_flags(a: Mapping, placebos: Sequence[Mapping] = (),
                        retrainers: Mapping[str, Mapping] | None = None) -> dict:
    """Bars that cannot be met by construction (report; never weaken after seeing them)."""
    best = best_placebo_reduction(a, placebos)
    flags = dict(best_placebo_reduction=best,
                 gate_1b_impossible=bool(2 * best > a["marker_flip_rate"] + TOL))
    if retrainers:
        flags["gate_5_impossible_vs"] = sorted(k for k, r in retrainers.items()
                                               if r["marker_flip_rate"] < BARS["5_retrain_flip_advantage"] - TOL)
    return flags


def evaluate(per_seed_values: Mapping[str, Mapping[str, float]]) -> dict:
    """Worst-seed aggregation per gate.

    Every seed must carry the same gates: a gate missing from one seed raises ``ValueError`` instead of being
    dropped, because dropping it could turn a failing gate into ``pass_all``.  So do no seeds and no known gate.
    """
    seeds = sorted(per_seed_values)
    if not seeds:
        raise ValueError("no seeds to evaluate")
    present = set().union(*[set(per_seed_values[s]) for s in seeds])
    missing = {s: sorted(present - set(per_seed_values[s])) for s in seeds}
    missing = {s: m for s, m in missing.items() if m}
    if missing:
        raise ValueError(f"gates missing from some seeds: {missing}")
    names = [n for n in BARS if n in present]
    if not names:
        raise ValueError("no known gate in the per-seed values")
    per_seed = {s: {n: dict(value=per_seed_values[s][n],
                            verdict="PASS" if per_seed_values[s][n] >= BARS[n] - TOL else "FAIL")
                    for n in names} for s in seeds}
    worst = {n: min(per_seed_values[s][n] for s in seeds) for n in names}
    bars = {n: "PASS" if worst[n] >= BARS[n] - TOL else "FAIL" for n in names}
    return dict(per_seed=per_seed, worst=worst, bars=bars,
                pass_all=all(v == "PASS" for v in bars.values()), gates=names)
