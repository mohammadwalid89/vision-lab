"""Metrics shared by the chapters."""

from __future__ import annotations

import numpy as np


def auc(y, s) -> float:
    """Area under the ROC curve for labels ``y`` (1 = positive) and scores ``s``; ties count one half.

    The share of (positive, negative) pairs the score orders correctly, as in the Mann-Whitney U statistic.
    Returns NaN when either class is empty.  Raises ``ValueError`` on NaN scores, which have no order, and on
    labels other than 0 and 1.
    """
    y = np.asarray(y).astype(int).ravel()
    s = np.asarray(s, dtype=float).ravel()
    if y.shape != s.shape:
        raise ValueError(f"{len(y)} labels but {len(s)} scores")
    if np.isnan(s).any():
        raise ValueError("NaN score")
    if not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    pos, neg = s[y == 1], np.sort(s[y == 0])
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    below = np.searchsorted(neg, pos, side="left")
    tied = np.searchsorted(neg, pos, side="right") - below
    return float((below.sum() + 0.5 * tied.sum()) / (len(pos) * len(neg)))
