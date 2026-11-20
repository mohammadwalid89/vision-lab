import numpy as np
import pytest

from vision_lab.wsp import gates as G


def _m(rng):
    return dict(marker_flip_rate=float(rng.uniform(0, .8)), recall=float(rng.uniform(.5, .9)),
                FPR=float(rng.uniform(0, .5)), conflict_set_FP=int(rng.integers(0, 2000)),
                AUROC=float(rng.uniform(.7, .95)))


def test_parity_with_r1_dev_bars(r1_bars):
    rng = np.random.default_rng(0)
    name = {"1a_marker_relative_reduction": "marker_relative_reduction_min",
            "1b_placebo_margin": "placebo_margin_min", "2_recall_margin": "recall_margin_min",
            "3_FPR_margin": "FPR_margin_min", "4_conflict_relative_reduction": "conflict_relative_reduction_min",
            "6_same_direction_all_seeds": "same_direction_all_seeds",
            "5_retrain_flip_advantage": "retrain_flip_advantage_min", "5_retrain_auroc_margin": "retrain_auroc_margin_min",
            "7_own_auroc_distance_margin": "own_auroc_distance_margin_min", "7_own_lower_flip_all_seeds": "own_lower_flip_all_seeds"}
    for _ in range(200):
        a, c, b, r = _m(rng), _m(rng), _m(rng), _m(rng)
        pl = [_m(rng) for _ in range(3)]
        best = max([0.] + [a["marker_flip_rate"] - p["marker_flip_rate"] for p in pl])
        for exp, fams in (("V1", ("CORE",)), ("V2", ("CORE", "G5")), ("OWN", ("G7",))):
            ref = r1_bars.seed_bars(c, a, b, r, best, exp)
            ours = G.seed_values(c, a, pl, {"r": r}, b, fams)
            for k, v in ours.items():
                assert v == pytest.approx(ref[name[k]]), (exp, k)


def test_relative_reduction_zero_baseline():
    assert G.relative_reduction(0, 0.3) == 0.0


def test_gate5_takes_worst_retrainer():
    a = dict(marker_flip_rate=.7, recall=.7, FPR=.1, conflict_set_FP=100, AUROC=.85)
    c = dict(marker_flip_rate=.1, recall=.7, FPR=.1, conflict_set_FP=10, AUROC=.85)
    retr = {"warm": dict(marker_flip_rate=.3, AUROC=.80), "fresh": dict(marker_flip_rate=.12, AUROC=.87)}
    v = G.seed_values(c, a, [], retr, None, ("CORE", "G5"))
    assert v["5_retrain_flip_advantage"] == pytest.approx(.02)
    assert v["5_retrain_auroc_margin"] == pytest.approx(-.02)


def test_worst_seed_decides_and_means_never_rescue():
    vals = {"1": {"2_recall_margin": .05}, "2": {"2_recall_margin": .05}, "3": {"2_recall_margin": -.021}}
    ev = G.evaluate(vals)
    assert ev["bars"]["2_recall_margin"] == "FAIL" and not ev["pass_all"]
    assert ev["per_seed"]["1"]["2_recall_margin"]["verdict"] == "PASS"


def test_impossibility_flags():
    a = dict(marker_flip_rate=.2)
    f = G.impossibility_flags(a, [dict(marker_flip_rate=.05)], {"warm": dict(marker_flip_rate=.04)})
    assert f["gate_1b_impossible"] and f["gate_5_impossible_vs"] == ["warm"]
    f2 = G.impossibility_flags(a, [dict(marker_flip_rate=.25)])          # negative gain -> 0
    assert f2["best_placebo_reduction"] == 0.0 and not f2["gate_1b_impossible"]
