"""vision_lab.metrics.auc gives the same numbers as the five chapter copies it replaced.

The old copies are kept below verbatim (chapter 02 used scikit-learn; chapter 06 used average ranks; chapters 15
and 16 counted pairs and took the arguments as (scores, labels)).
"""
import numpy as np
import pytest

from vision_lab.metrics import auc


def old_pairs_auc(s, y):  # chapters 15 (placebo.py, dev_eval.py) and 16 (suite.py)
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def old_ranks(x):  # chapter 06 (programmed_head.py)
    order = np.argsort(x, kind="mergesort")
    out = np.empty(len(x), float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return out


def old_ranks_auc(y, s):  # chapter 06
    y = np.asarray(y).astype(int)
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = old_ranks(np.asarray(s, float))
    return float((r[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def old_sklearn_auc(y, s):  # chapter 02 (common.py)
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def cases():
    rng = np.random.default_rng(20261122)
    for k in range(300):
        n = int(rng.integers(2, 400))
        y = rng.integers(0, 2, n)
        if y.min() == y.max():
            y[0] = 1 - y[0]
        s = rng.normal(size=n) + 0.8 * y
        if k % 3 == 0:
            s = np.round(s, 1)  # many ties
        if k % 3 == 1:
            s = s.astype(np.float32).astype(np.float64)
        yield y, s


def test_identical_to_the_pair_counting_copies():
    for y, s in cases():
        assert auc(y, s) == old_pairs_auc(s, y)  # bit for bit


def test_same_numbers_as_the_rank_and_sklearn_copies():
    pytest.importorskip("sklearn")
    for y, s in cases():
        new = auc(y, s)
        for old in (old_ranks_auc(y, s), old_sklearn_auc(y, s)):
            assert new == pytest.approx(old, abs=1e-12)
        # Four decimals only for the rank copy: a 2.2e-16 difference from scikit-learn can cross a rounding boundary.
        assert round(new, 4) == round(old_ranks_auc(y, s), 4)


def test_argument_order_is_labels_then_scores():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert auc(y, s) == 1.0
    assert auc(np.array([0, 1, 0, 1]), np.array([0.5, 0.5, 0.2, 0.9])) == 0.875


def test_edge_cases():
    assert np.isnan(auc([1, 1], [0.2, 0.3]))
    assert np.isnan(auc([0, 0], [0.2, 0.3]))
    with pytest.raises(ValueError):
        auc([0, 1], [0.2, float("nan")])
    with pytest.raises(ValueError):
        auc([0, 2], [0.2, 0.3])
    with pytest.raises(ValueError):
        auc([0, 1, 1], [0.2, 0.3])
