"""Weight-edit families and no-op controls.

Edit families (after the deletion-mask paper and the "Weight-Space Reading" toolkit: block/row scaling, direction removal, noise):
  delete(W, keep)                   deletion-only (zero entries where keep == False)
  random_deletion_keep(...)         uniformly random deletion set of exactly r entries
  signed_sparse(W, support, delta)  additive delta on a fixed support
  scale_rows / scale_block          multiply rows (output units) or a block by a factor
  remove_direction(W, v)            W <- (I - v v^T/|v|^2) W  (input direction removal, W is (d, h))
  norm_matched_noise(W, ref, rng)   Gaussian noise with ||noise|| == ||ref delta||

No-op controls (six; each must leave the head's function unchanged — the harness checks
max |score change| <= tol and that every effect call is SAME):
  identity            fresh copy, separate code path
  scale_one           scale_block(..., factor=1.0)
  zero_delta          signed_sparse with an all-zero delta on the slot's support
  dtype_roundtrip     float64 -> float32 -> float64 on weights already representable in f32
  unit_permutation    permute hidden units (W1 columns, b1, W2) — changes ~all weights
  tanh_sign_flip      negate (W1[:, j], b1[j], W2[j]) for a set of units; tanh is odd
The last two change many weight values while computing the same function: they are the
cases where a reader/heuristic that looks only at weight diffs predicts a change that does
not happen (the WSR false-positive warning).
"""
from __future__ import annotations

import hashlib

import numpy as np

from .heads import mlp_copy


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def delete(W, keep):
    keep = np.asarray(keep, dtype=bool)
    if keep.shape != np.shape(W):
        raise ValueError("keep mask shape mismatch")
    return np.where(keep, W, 0.0)


def n_deletions(fraction: float, size: int) -> int:
    """r = round(fraction * D), half away from zero (deletion-probe convention)."""
    x = fraction * size
    return int(np.floor(x + 0.5))


def random_deletion_keep(shape, r: int, seed: int) -> np.ndarray:
    size = int(np.prod(shape))
    if not 0 <= r <= size:
        raise ValueError("r out of range")
    idx = np.random.default_rng(seed).choice(size, size=r, replace=False)
    keep = np.ones(size, dtype=bool)
    keep[idx] = False
    return keep.reshape(shape)


def signed_sparse(W, support, delta):
    support = np.asarray(support, dtype=bool)
    delta = np.asarray(delta, dtype=np.float64)
    if support.shape != np.shape(W) or delta.shape != np.shape(W):
        raise ValueError("support and delta must have W's shape")
    out = np.array(W, dtype=np.float64, copy=True)
    out[support] += delta[support]
    return out


def scale_rows(W, rows, factor):
    out = np.array(W, dtype=np.float64, copy=True)
    out[np.asarray(rows)] *= factor
    return out


def scale_block(W, index, factor):
    out = np.array(W, dtype=np.float64, copy=True)
    out[index] = out[index] * factor
    return out


def remove_direction(W, v):
    """Project input direction v out of a first-layer weight matrix W of shape (d, h)."""
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    nv = v @ v
    if nv == 0:
        raise ValueError("zero direction")
    W = np.asarray(W, dtype=np.float64)
    return W - np.outer(v, v @ W) / nv


def norm_matched_noise(W, ref_delta_norm: float, seed: int):
    n = np.random.default_rng(seed).normal(size=np.shape(W))
    nn = np.linalg.norm(n)
    return np.asarray(W, dtype=np.float64) + n * (ref_delta_norm / nn if nn else 0.0)


def noop_identity(p):
    return mlp_copy(p)


def noop_scale_one(p):
    q = mlp_copy(p)
    q["W1"] = scale_block(q["W1"], (slice(None), slice(None)), 1.0)
    return q


def noop_zero_delta(p, support=None):
    q = mlp_copy(p)
    sup = np.ones_like(q["W1"], dtype=bool) if support is None else support
    q["W1"] = signed_sparse(q["W1"], sup, np.zeros_like(q["W1"]))
    return q


def noop_dtype_roundtrip(p):
    """Exact only for weights representable in float32; we first snap A to float32 values.

    Returns (snapped_reference, roundtripped) so the comparison is exact by construction.
    """
    snapped = {k: np.asarray(v, dtype=np.float32).astype(np.float64) for k, v in p.items()}
    rt = {k: np.asarray(np.asarray(v, dtype=np.float32), dtype=np.float64) for k, v in snapped.items()}
    return snapped, rt


def noop_unit_permutation(p, key: str = "noop|perm"):
    q = mlp_copy(p)
    h = q["W1"].shape[1]
    perm = _rng(key).permutation(h)
    if h > 1 and np.array_equal(perm, np.arange(h)):
        perm = np.roll(perm, 1)
    q["W1"], q["b1"], q["W2"] = q["W1"][:, perm], q["b1"][perm], q["W2"][perm]
    return q


def noop_tanh_sign_flip(p, key: str = "noop|flip"):
    q = mlp_copy(p)
    h = q["W1"].shape[1]
    units = _rng(key).choice(h, size=max(1, h // 2), replace=False)
    q["W1"][:, units] *= -1
    q["b1"][units] *= -1
    q["W2"][units] *= -1
    return q


NOOPS = ("identity", "scale_one", "zero_delta", "dtype_roundtrip", "unit_permutation", "tanh_sign_flip")


def noop_suite(p, support=None):
    """{name: (reference_params, noop_params)} for the six function-preserving no-ops."""
    snapped, rt = noop_dtype_roundtrip(p)
    return {
        "identity": (p, noop_identity(p)),
        "scale_one": (p, noop_scale_one(p)),
        "zero_delta": (p, noop_zero_delta(p, support)),
        "dtype_roundtrip": (snapped, rt),
        "unit_permutation": (p, noop_unit_permutation(p)),
        "tanh_sign_flip": (p, noop_tanh_sign_flip(p)),
    }


def weight_diff_stats(p, q) -> dict:
    """How much the *weights* changed (what a weight-diff reader sees)."""
    changed = sum(int(np.sum(np.asarray(p[k]) != np.asarray(q[k]))) for k in p)
    total = sum(int(np.size(p[k])) for k in p)
    l2 = float(np.sqrt(sum(np.sum((np.asarray(p[k]) - np.asarray(q[k])) ** 2) for k in p)))
    return dict(n_changed=changed, n_total=total, frac_changed=changed / total, l2=l2)
