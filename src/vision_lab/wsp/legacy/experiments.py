"""Heads, weight-space moves and feature ablations for Rounds 1 and 2.

Everything here works on arrays. Estimators are fitted on TRAIN rows only; the
caller is responsible for that, since arrays carry no split information.
"""

import hashlib

import numpy as np

SEEDS = (20261112, 20261113, 20261114)


def array_hash(x):
    x = np.ascontiguousarray(x)
    return hashlib.sha256(
        str(x.shape).encode() + str(x.dtype).encode() + x.tobytes()
    ).hexdigest()


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z)))


class Standardizer:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-5)
        return self

    def transform(self, x):
        return ((x - self.mean) / self.scale).astype(np.float32)


def train_head(
    x, y, seed, steps=200, lr=0.03, l2=0.001, sample_weight=None, initial=None
):
    """Full-batch weighted logistic regression by gradient descent.

    Head A and head B are trained on the same paired matrix; only the sample
    weights differ (A: the confounded view only, B: both views at half weight),
    so both cost the same number of updates.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or len(y) != len(x) or not np.isin(y, [0, 1]).all():
        raise ValueError('Bad training arrays')
    w = (
        np.ones(len(y))
        if sample_weight is None
        else np.asarray(sample_weight, dtype=np.float64)
    )
    if w.shape != y.shape or np.any(w < 0) or w.sum() <= 0:
        raise ValueError('Bad sample weights')
    w = w / w.sum()
    theta = (
        np.random.default_rng(seed).normal(0, 0.001, x.shape[1] + 1)
        if initial is None
        else np.array(initial, dtype=float, copy=True)
    )
    for _ in range(steps):
        err = (sigmoid(x @ theta[:-1] + theta[-1]) - y) * w
        theta[:-1] -= lr * (x.T @ err + l2 * theta[:-1])
        theta[-1] -= lr * err.sum()
    return theta


def logits(x, theta):
    return np.asarray(x) @ theta[:-1] + theta[-1]


def train_pair(x, y, confounded_weights, seed, steps=200, lr=0.03, l2=0.001):
    initial = np.random.default_rng(seed).normal(0, 0.001, x.shape[1] + 1)
    a = train_head(
        x,
        y,
        seed,
        steps=steps,
        lr=lr,
        l2=l2,
        sample_weight=confounded_weights,
        initial=initial,
    )
    b = train_head(
        x,
        y,
        seed,
        steps=steps,
        lr=lr,
        l2=l2,
        sample_weight=np.full(len(y), 0.5),
        initial=initial,
    )
    # The retrained reference gets the total A+B budget (2 * steps).
    retrained = train_head(
        x,
        y,
        seed + 100000,
        steps=2 * steps,
        lr=lr,
        l2=l2,
        sample_weight=np.full(len(y), 0.5),
    )
    return a, b, retrained


def v2_weights(a, b, alpha):
    if a.shape != b.shape:
        raise ValueError('Same-base head shapes required')
    return b + alpha * (b - a)


def norm_match(x, target):
    x = np.asarray(x, dtype=float)
    n = np.linalg.norm(x)
    if n < 1e-12:
        raise ValueError('Zero direction')
    return x * (np.linalg.norm(target) / n)


def v2_controls(a, b, retrained, seed):
    rng = np.random.default_rng(seed)
    delta = b - a
    if np.linalg.norm(delta) < 1e-12:
        raise ValueError('Zero A/B delta: intervention unidentifiable')
    random_delta = norm_match(rng.normal(size=delta.shape), delta)
    shuffled = rng.permutation(delta)
    return dict(
        A=a.copy(),
        B=b.copy(),
        midpoint=(a + b) / 2,
        random_norm_matched_delta=b + random_delta,
        shuffled_delta=b + shuffled,
        sign_flip=b - delta,
        matched_compute_retrained=retrained.copy(),
    )


def concept_directions(clean, marked, good_clean, seed):
    clean = np.asarray(clean)
    marked = np.asarray(marked)
    good_clean = np.asarray(good_clean)
    if clean.shape != marked.shape or len(clean) < 2:
        raise ValueError('Aligned train pairs required')
    marker = (marked - clean).mean(0)
    if np.linalg.norm(marker) < 1e-8:
        raise ValueError('Marker concept unidentifiable')
    rng = np.random.default_rng(seed)
    random = norm_match(rng.normal(size=marker.shape), marker)
    order = rng.permutation(len(good_clean))
    mid = len(order) // 2
    if not mid or mid == len(order):
        raise ValueError('Need two train-good groups')
    irrelevant = good_clean[order[:mid]].mean(0) - good_clean[order[mid:]].mean(0)
    # Orthogonalise against the marker so the control cannot reuse it.
    irrelevant = irrelevant - marker * (irrelevant @ marker) / (marker @ marker)
    irrelevant = norm_match(irrelevant, marker)
    return dict(marker=marker, random=random, irrelevant=irrelevant, sham=marker.copy())


def ablate(x, direction, dose, center, sham=False):
    x = np.asarray(x)
    d = np.asarray(direction)
    norm = d @ d
    if norm < 1e-12:
        raise ValueError('Zero concept')
    component = np.outer((x - center) @ d / norm, d)
    # The sham computes the same projection but applies zero dose.
    return x - (0.0 if sham else dose) * component


def train_weights(rows):
    if any(r['split'] != 'train' for r in rows):
        raise PermissionError('Train only')
    return np.array([float(r['confounded_train']) for r in rows], dtype=float)


def align_pairs(rows, split):
    if split not in ('train', 'dev'):
        raise PermissionError('Holdout locked')
    groups = {}
    for i, r in enumerate(rows):
        if r['split'] == split and r['variant'] in ('clean', 'marker'):
            g = groups.setdefault(r['group_id'], {})
            if r['variant'] in g:
                raise ValueError('Duplicate pair member')
            g[r['variant']] = i
    if not groups or any(set(g) != {'clean', 'marker'} for g in groups.values()):
        raise ValueError('Incomplete pairs')
    ordered = [groups[k] for k in sorted(groups)]
    return np.array([g['clean'] for g in ordered]), np.array(
        [g['marker'] for g in ordered]
    )
