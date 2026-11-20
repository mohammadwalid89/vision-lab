"""Slot 2 — object-patch scoring on the native patch-token grid (14x14 for SigLIP2-B/16).

Primary recipe (declared, no pooling/dilation/layer search):
  coverage_i = SAM object-mask area fraction of patch i (mask BOX-resized to 224 px, then
               averaged over each 16x16 patch); w = coverage / sum(coverage) (uniform if 0)
  z_obj      = sum_i w_i t_i  over last-layer patch tokens t_i (post-layernorm)
  feature    = z_obj ONLY (768-d)          <- ``mode='object_only'`` (primary)
             | [pooled | z_obj] (1536-d)   <- ``mode='pooled_concat'`` (diagnostic only)
Rationale for ``object_only``: the MAP-pooled vector attends over every token, including
the marker's; the 4x4 probe kept it in the feature and was inert (flip -0.2..-0.6 %).
Concatenating it re-opens the shortcut channel the slot is meant to close.

"Scoring" vs "pooling": for a linear head, scoring each object token and taking the
w-weighted mean of the token logits equals the head applied to z_obj (weights sum to 1).
``token_logit_aggregate`` exposes the scoring form; the test suite pins the identity.

Placebo: three within-source random permutations of the 196 coverage
weights (hash namespace ``r3|slot2|perm|<source_id>|<draw>``), applied to the SAME tokens
while they are in memory — the aggregate cannot be reconstructed afterwards from z and w.
The same permutation is reused for every view of a source (clean/marker/nuisance).

Limit: SigLIP tokens mix globally through attention, so object tokens can still
carry marker information.  ``marker_leakage`` measures it on the paired views.
"""
from __future__ import annotations

import hashlib

import numpy as np


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def patch_coverage(mask, grid: int = 14, img: int = 224) -> np.ndarray:
    """Area fraction of the (binary) mask inside each grid patch, flattened row-major."""
    from PIL import Image
    m = np.asarray(mask)
    m = (m > (127 if m.dtype == np.uint8 and m.max() > 1 else 0)).astype(np.uint8) * 255
    a = np.asarray(Image.fromarray(m, mode="L").resize((img, img), Image.Resampling.BOX),
                   dtype=np.float64) / 255.0
    if img % grid:
        raise ValueError("img must be a multiple of grid")
    p = img // grid
    return a.reshape(grid, p, grid, p).mean(axis=(1, 3)).reshape(-1)


def normalise_weights(cov) -> np.ndarray:
    c = np.asarray(cov, dtype=np.float64)
    s = c.sum(axis=-1, keepdims=True)
    uni = np.full_like(c, 1.0 / c.shape[-1])
    return np.where(s > 0, c / np.where(s > 0, s, 1.0), uni)


def aggregate(tokens, w) -> np.ndarray:
    """tokens (N, P, D), w (N, P) -> (N, D)."""
    t = np.asarray(tokens, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if t.ndim != 3 or w.shape != t.shape[:2]:
        raise ValueError("tokens (N,P,D) and weights (N,P) required")
    return np.einsum("np,npd->nd", w, t)


def permutation_for(source_id: str, n: int, draw: int, namespace: str = "r3|slot2|perm|") -> np.ndarray:
    perm = _rng(f"{namespace}{source_id}|{draw}").permutation(n)
    if n > 1 and np.array_equal(perm, np.arange(n)):
        perm = np.roll(perm, 1)
    return perm


def permuted_weights(w, source_ids, draw: int) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    out = np.empty_like(w)
    for i, sid in enumerate(source_ids):
        out[i] = w[i][permutation_for(sid, w.shape[1], draw)]
    return out


def object_feature(pooled, z_obj, mode: str = "object_only") -> np.ndarray:
    if mode == "object_only":
        return np.asarray(z_obj, dtype=np.float64)
    if mode == "pooled_concat":
        return np.concatenate([np.asarray(pooled, dtype=np.float64), np.asarray(z_obj, dtype=np.float64)], 1)
    raise ValueError(mode)


def token_logit_aggregate(tokens, w, theta) -> np.ndarray:
    """Score every token with a linear head theta (D+1), then w-weighted mean of logits."""
    t = np.asarray(tokens, dtype=np.float64)
    logits = t @ theta[:-1] + theta[-1]           # (N, P)
    return np.einsum("np,np->n", np.asarray(w, dtype=np.float64), logits)


def aggregates_with_controls(tokens, w, source_ids, n_draws: int = 3) -> dict:
    """In-memory hook: primary aggregate + the declared permutation placebos."""
    out = {"object": aggregate(tokens, w)}
    for d in range(1, n_draws + 1):
        out[f"perm_{d}"] = aggregate(tokens, permuted_weights(w, source_ids, d))
    out["uniform"] = aggregate(tokens, np.full(np.shape(w), 1.0 / np.shape(w)[1]))
    return out


def marker_leakage(clean_vec, marker_vec) -> np.ndarray:
    """Per-pair relative change ||v_marker - v_clean|| / ||v_clean||."""
    c = np.asarray(clean_vec, dtype=np.float64)
    m = np.asarray(marker_vec, dtype=np.float64)
    return np.linalg.norm(m - c, axis=-1) / np.maximum(np.linalg.norm(c, axis=-1), 1e-12)
