"""Slot 1 — SAM object input masking with hard mean-fill (pipeline hooks).

One primary transform, no search: pixels outside the (already 6-px-dilated, per-source)
R1 SAM object mask are replaced by a constant equal to the encoder's normalisation mean
(``encoder_mean``: SigLIP (0.5,0.5,0.5) -> RGB 128, i.e. ~0 in model input space), at the
image's native resolution, BEFORE the encoder's own resize/normalise.  Consequence, and
the property the tests pin: the encoder input is a function of object pixels only, so
anything placed outside the mask (the planted marker) cannot reach the features.

Why not "image mean": a fill computed from the whole image includes the marker's pixels,
so the fill colour itself differs between the clean and marked view — the shortcut leaks
through every filled pixel.  ``image_mean`` is therefore refused.  ``object_mean`` (mean of
kept pixels) is leak-free and offered for completeness; it is NOT the primary.

Controls built here:
  * ``random_area``  equal-area random mask: the object mask cyclically shifted by a
    hash-seeded offset (exactly equal area and shape; 3 draws; one draw per source shared
    by all of that source's views).  Overlap with the object and with the marker region is
    reported, because a random mask that mostly deletes the object is a weak control.
  * ``padding_only`` deterministic removal of the synthetic outer padding using the known
    canvas geometry (tests whether SAM adds anything when the marker is planted
    outside the original image pixels).
  * ``identity``     no-op transform (sham): must give bit-identical features.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np

SIGLIP_MEAN = (0.5, 0.5, 0.5)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
FILL_MODES = ("encoder_mean", "object_mean")
LEAKY_FILL_MODES = ("image_mean",)


class LeakError(ValueError):
    """A transform whose output depends on pixels it claims to remove."""


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def resize_mask(mask, hw) -> np.ndarray:
    """Nearest-neighbour resize of a binary mask to (H, W)."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("2-D mask required")
    m = m > (127 if m.dtype == np.uint8 and m.max() > 1 else 0)
    H, W = hw
    if m.shape == (H, W):
        return m.copy()
    ys = np.minimum((np.arange(H) + 0.5) * m.shape[0] / H, m.shape[0] - 1).astype(int)
    xs = np.minimum((np.arange(W) + 0.5) * m.shape[1] / W, m.shape[1] - 1).astype(int)
    return m[ys][:, xs]


def fill_value(image, keep, mode: str, encoder_mean: Sequence[float] = SIGLIP_MEAN) -> np.ndarray:
    if mode in LEAKY_FILL_MODES:
        raise LeakError(f"fill mode {mode!r} depends on removed pixels (marker leak)")
    if mode == "encoder_mean":
        return np.round(np.asarray(encoder_mean) * 255.0).astype(np.uint8)
    if mode == "object_mean":
        if not keep.any():
            return np.round(np.asarray(encoder_mean) * 255.0).astype(np.uint8)
        return np.round(image[keep].reshape(-1, image.shape[-1]).mean(0)).astype(np.uint8)
    raise ValueError(f"unknown fill mode {mode!r}")


def mask_fill(image, keep_mask, mode: str = "encoder_mean", encoder_mean=SIGLIP_MEAN) -> np.ndarray:
    """Hard fill of every pixel where keep_mask is False.  image: (H, W, 3) uint8."""
    img = np.asarray(image)
    if img.ndim != 3 or img.dtype != np.uint8:
        raise ValueError("uint8 (H, W, 3) image required")
    keep = resize_mask(keep_mask, img.shape[:2])
    out = img.copy()
    out[~keep] = fill_value(img, keep, mode, encoder_mean)
    return out


def padding_only_keep(hw, content_box) -> np.ndarray:
    """Keep exactly the original-image rectangle (y0, x0, y1, x1) of a padded canvas."""
    H, W = hw
    y0, x0, y1, x1 = content_box
    if not (0 <= y0 < y1 <= H and 0 <= x0 < x1 <= W):
        raise ValueError("content box outside canvas")
    keep = np.zeros((H, W), dtype=bool)
    keep[y0:y1, x0:x1] = True
    return keep


def random_equal_area_keep(keep_mask, key: str) -> tuple[np.ndarray, dict]:
    """Cyclic shift of the object mask by a hash-seeded offset (exact area/shape match)."""
    m = np.asarray(keep_mask, dtype=bool)
    H, W = m.shape
    rng = _rng(key)
    dy, dx = int(rng.integers(0, H)), int(rng.integers(0, W))
    if dy == 0 and dx == 0:
        dy = H // 2
    r = np.roll(m, (dy, dx), axis=(0, 1))
    return r, dict(shift=(dy, dx), area=int(m.sum()), area_random=int(r.sum()))


def overlap_report(random_keep, object_keep, marker_box=None) -> dict:
    """Disclose what a random mask removes: object pixels lost, marker pixels kept."""
    rk = np.asarray(random_keep, dtype=bool)
    ok = resize_mask(object_keep, rk.shape)
    out = dict(object_pixels_removed=float((ok & ~rk).sum() / max(1, ok.sum())),
               jaccard_with_object=float((ok & rk).sum() / max(1, (ok | rk).sum())))
    if marker_box is not None:
        y0, x0, y1, x1 = marker_box
        mk = np.zeros_like(rk)
        mk[y0:y1, x0:x1] = True
        out["marker_pixels_kept"] = float((mk & rk).sum() / max(1, mk.sum()))
    return out


@dataclass(frozen=True)
class InputTransform:
    """A named, deterministic image transform applied before the encoder.

    ``fn(image, meta) -> image``; ``meta`` carries at least ``source_id`` and, depending
    on the transform, ``object_mask`` (per source, any resolution) and ``content_box``.
    """
    name: str
    kind: str          # 'primary' | 'control' | 'reference' | 'noop'
    fn: Callable

    def __call__(self, image, meta: Mapping):
        return self.fn(image, meta)


def _primary(image, meta, mode="encoder_mean", encoder_mean=SIGLIP_MEAN):
    return mask_fill(image, meta["object_mask"], mode, encoder_mean)


def padding_kept_fraction(object_keep, content_box, key: str) -> float:
    """Label-free placebo-bar risk check: fraction of synthetic-padding pixels a random
    equal-area mask KEEPS.  The marker lives in the padding, so 1 - this is roughly the chance
    the placebo removes the marker; if it is > .5, gate 1b (2 x best placebo) is likely
    impossible by construction."""
    ok = np.asarray(object_keep, dtype=bool)
    rk, _ = random_equal_area_keep(ok, key)
    pad = ~padding_only_keep(ok.shape, content_box)
    return float((rk & pad).sum() / max(1, pad.sum()))


def build_transforms(n_random: int = 3, encoder_mean=SIGLIP_MEAN,
                     namespace: str = "r3|slot1|random_area|", padding_sparing: bool = False) -> dict:
    """The declared slot-1 transform family (primary + controls).  No variants.

    ``padding_sparing=True`` adds ``random_area_ps_{d}``: the same shifted mask OR the
    synthetic padding, i.e. it removes random image content but never the padding/marker.
    Used as the slot-1 placebo in the run (declared in the plan before any result was read).
    """
    t = {"identity": InputTransform("identity", "noop", lambda im, meta: np.asarray(im).copy()),
         "sam_meanfill": InputTransform("sam_meanfill", "primary",
                                        lambda im, meta: _primary(im, meta, "encoder_mean", encoder_mean)),
         "padding_only": InputTransform(
             "padding_only", "reference",
             lambda im, meta: mask_fill(im, padding_only_keep(np.shape(im)[:2], meta["content_box"]),
                                        "encoder_mean", encoder_mean))}
    for d in range(1, n_random + 1):
        def fn(im, meta, d=d):
            obj = resize_mask(meta["object_mask"], np.shape(im)[:2])
            rk, _ = random_equal_area_keep(obj, f"{namespace}{meta['source_id']}|{d}")
            return mask_fill(im, rk, "encoder_mean", encoder_mean)
        t[f"random_area_{d}"] = InputTransform(f"random_area_{d}", "control", fn)
        if padding_sparing:
            def fn_ps(im, meta, d=d):
                obj = resize_mask(meta["object_mask"], np.shape(im)[:2])
                rk, _ = random_equal_area_keep(obj, f"{namespace}{meta['source_id']}|{d}")
                rk = rk | ~padding_only_keep(np.shape(im)[:2], meta["content_box"])
                return mask_fill(im, rk, "encoder_mean", encoder_mean)
            t[f"random_area_ps_{d}"] = InputTransform(f"random_area_ps_{d}", "control", fn_ps)
    return t
