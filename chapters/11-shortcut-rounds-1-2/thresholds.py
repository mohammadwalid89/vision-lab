"""Operating-point rule used from Round 1 on.

Same rule as `freeze_threshold` in the DEV scorer (maximum balanced accuracy on
clean DEV views, higher threshold wins ties), computed with one sort instead of
one confusion matrix per candidate. The test suite checks the two agree.
"""

import hashlib

import numpy as np

from vision_lab.wsp.legacy.scorer import checked_arrays


def fast_freeze(rows, scores, reference_hash):
    if not rows or any(r['split'] != 'dev' or r['variant'] != 'clean' for r in rows):
        raise PermissionError('Clean DEV only')
    y, s = checked_arrays([r['label'] for r in rows], scores)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if not n1 or not n0:
        raise ValueError('Both classes required')
    order = np.argsort(s, kind='stable')
    ss = s[order]
    yy = y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    removed = np.r_[0, np.cumsum(yy)][starts]
    tp = n1 - removed
    fp = n0 - (starts - removed)
    thresholds = np.r_[ss[starts], np.nextafter(ss[-1], np.inf)]
    ba = np.r_[(tp / n1 + 1 - fp / n0) / 2, 0.5]
    best = max(zip(ba, thresholds))
    threshold = float(best[1])
    return dict(
        threshold=threshold,
        selection='max clean-dev Youden J (equivalent to balanced accuracy); higher threshold tie break',
        reference_weight_hash=reference_hash,
        dev_ids=[r['id'] for r in rows],
        dev_balanced_accuracy=float(best[0]),
        dev_youden_j=float(2 * best[0] - 1),
        dev_scores_sha256=hashlib.sha256(
            np.asarray(s, dtype='<f8').tobytes()
        ).hexdigest(),
        frozen=True,
    )
