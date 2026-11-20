"""Planted-marker synthetic worlds for unit / known-answer / end-to-end tests.

``feature_world``  R1-schema rows + features: x = noise + label*a*u_def + marker*b*u_mark.
                   Historical ("confounded") TRAIN shows defects with the marker and goods
                   without, so a head fit on it learns marker -> defect (the shortcut).
``image_world``    tiny RGB canvases: object disc inside a content box, synthetic padding,
                   a bright marker square in the padding, dark defect spot on the object.
``TinyPatchEncoder`` patch-embedding encoder (optionally one global attention mixing step)
                   returning SigLIP-like dict(pooled, tokens, mean_patch).
"""
from __future__ import annotations

import numpy as np

CATS = ("cat_a", "cat_b", "cat_c")


def feature_world(n_train=600, n_dev=300, d=24, defect_frac_train=0.3, defect_frac_dev=0.3,
                  a=2.0, b=3.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    u_def = np.zeros(d)
    u_def[0] = 1.0
    u_mark = np.zeros(d)
    u_mark[1] = 1.0
    u_bri = np.zeros(d)
    u_bri[2] = 1.0
    rows, X = [], []

    def add(split, src, label, cat, variant, conf, base):
        marker = variant == "marker"
        x = base + label * a * u_def + marker * b * u_mark
        if variant == "brightness":
            x = x + 1.5 * u_bri
        if variant == "background":
            x = x + rng.normal(0, 0.3, d)
        rows.append(dict(id=f"{src}|{variant}", group_id=f"g{src}", source_id=src, split=split, label=int(label),
                         variant=variant, category=cat, confounded_train=float(conf)))
        X.append(x)

    for split, n, frac in (("train", n_train, defect_frac_train), ("dev", n_dev, defect_frac_dev)):
        for i in range(n):
            src = f"{split[0]}{i:05d}"
            label = int(rng.random() < frac)
            cat = CATS[i % len(CATS)]
            base = rng.normal(0, noise, d)
            if split == "train":
                add(split, src, label, cat, "clean", conf=(label == 0), base=base)
                add(split, src, label, cat, "marker", conf=(label == 1), base=base)
            else:
                for v in ("clean", "marker", "brightness", "background"):
                    add(split, src, label, cat, v, conf=0, base=base)
    return rows, np.asarray(X, dtype=np.float64), dict(u_def=u_def, u_mark=u_mark)


def image_world(n=8, size=32, pad=4, marker=3, seed=0):
    """Returns list of dicts(image_clean, image_marker, object_mask, content_box, marker_box, label)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    c = size / 2 - 0.5
    rad = (size - 2 * pad) / 2 - 2
    obj = (yy - c) ** 2 + (xx - c) ** 2 <= rad ** 2
    out = []
    for i in range(n):
        img = np.full((size, size, 3), 40, dtype=np.uint8)               # padding colour
        img[pad:size - pad, pad:size - pad] = rng.integers(90, 110, 3)   # original image area
        img[obj] = rng.integers(150, 200, 3)                             # object
        label = i % 2
        if label:
            dy, dx = rng.integers(-3, 4, 2)
            img[int(c) + dy - 1:int(c) + dy + 2, int(c) + dx - 1:int(c) + dx + 2] = 10
        mk = img.copy()
        mk[0:marker, 0:marker] = 255
        out.append(dict(image_clean=img, image_marker=mk, object_mask=obj.copy(),
                        content_box=(pad, pad, size - pad, size - pad), marker_box=(0, 0, marker, marker),
                        label=label, source_id=f"s{i:03d}"))
    return out


class TinyPatchEncoder:
    """Deterministic patch encoder.  ``attention=False`` -> tokens depend only on their own patch."""

    def __init__(self, size=32, patch=4, dim=16, attention=False, seed=0, mean=(0.5, 0.5, 0.5)):
        rng = np.random.default_rng(seed)
        self.size, self.patch, self.dim, self.attention = size, patch, dim, attention
        self.grid = size // patch
        self.E = rng.normal(0, 0.2, (patch * patch * 3, dim))
        self.Q = rng.normal(0, 0.3, (dim, dim))
        self.K = rng.normal(0, 0.3, (dim, dim))
        self.mean = np.asarray(mean)

    def tokens(self, img):
        x = (np.asarray(img, dtype=np.float64) / 255.0 - self.mean) / 0.5
        g, p = self.grid, self.patch
        patches = x.reshape(g, p, g, p, 3).transpose(0, 2, 1, 3, 4).reshape(g * g, p * p * 3)
        t = patches @ self.E
        if self.attention:
            q, k = t @ self.Q, t @ self.K
            a = q @ k.T / np.sqrt(self.dim)
            a = np.exp(a - a.max(1, keepdims=True))
            a /= a.sum(1, keepdims=True)
            t = t + a @ t
        return t

    def __call__(self, img):
        t = self.tokens(img)
        return dict(tokens=t, pooled=t.mean(0), mean_patch=t.mean(0))
