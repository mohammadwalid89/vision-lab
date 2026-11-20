import numpy as np
import pytest

from vision_lab.wsp import significance as S


def test_welch_matches_scipy():
    from scipy import stats
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 30), rng.normal(0.5, 2, 17)
    ours = S.welch(a, b)
    ref = stats.ttest_ind(a, b, equal_var=False)
    assert ours["t"] == pytest.approx(ref.statistic) and ours["p"] == pytest.approx(ref.pvalue)


def test_signflip_exact_known_answer():
    assert S.paired_signflip([1, 1, 1, 1])["p"] == pytest.approx(2 / 16)
    assert S.paired_signflip([0, 0, 0])["p"] == 1.0


def test_signflip_monte_carlo_close_to_exact():
    rng = np.random.default_rng(1)
    d = rng.normal(0.3, 1, 40)
    p = S.paired_signflip(d, n_perm=20000, seed=3)["p"]
    from scipy import stats
    assert abs(p - stats.ttest_1samp(d, 0).pvalue) < 0.05


def test_call_requires_significance_and_min_effect():
    assert S.call(0.10, 0.001, 0.05) == "UP"
    assert S.call(-0.10, 0.001, 0.05) == "DOWN"
    assert S.call(0.01, 1e-9, 0.05) == "SAME"      # significant but tiny
    assert S.call(0.30, 0.05, 0.05) == "SAME"      # large but not significant


def _units(n, flip, clean_pred, label, score):
    return dict(groups=[f"g{i}" for i in range(n)], label=label, flip=flip, clean_pred=clean_pred,
                conflict=np.where(label == 0, clean_pred, np.nan), clean_score=score, marker_score=score)


def test_identical_arms_are_same_everywhere():
    rng = np.random.default_rng(2)
    n = 200
    lab = (rng.random(n) < .3).astype(int)
    u = _units(n, rng.integers(0, 2, n).astype(float), rng.integers(0, 2, n).astype(float), lab, rng.normal(size=n))
    calls = S.paired_metric_calls(u, u)
    assert all(c["call"] == "SAME" and c["p"] == 1.0 for c in calls.values())


def test_large_flip_reduction_called_down():
    n = 400
    lab = np.zeros(n, dtype=int)
    ref = _units(n, np.ones(n), np.zeros(n), lab, np.zeros(n))
    arm = _units(n, np.r_[np.zeros(300), np.ones(100)], np.zeros(n), lab, np.zeros(n))
    assert S.paired_metric_calls(arm, ref)["marker_flip_rate"]["call"] == "DOWN"


def test_noop_false_positive_rate():
    r = S.noop_false_positive_rate({"a": {"x": {"call": "SAME"}, "y": {"call": "UP"}}, "b": {"x": {"call": "SAME"}}})
    assert r["n_false_positive"] == 1 and r["rate"] == pytest.approx(1 / 3)
