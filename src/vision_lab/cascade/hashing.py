"""Perceptual hashing and cross-split duplicate checks for fixed image cohorts."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def phash(path: str | Path) -> str:
    """256-bit DCT perceptual hash as a 64-character hex string."""
    with Image.open(path) as image:
        pixels = np.asarray(image.convert("L").resize((64, 64)), dtype=np.float32)
    axis = np.arange(64)
    basis = np.cos(np.pi * (2 * axis[None, :] + 1) * axis[:, None] / 128)
    basis[0] *= 1 / np.sqrt(64)
    basis[1:] *= np.sqrt(2 / 64)
    low = (basis @ pixels @ basis.T)[:16, :16]
    bits = low > np.median(low[1:, :])
    return f"{sum(int(bit) << i for i, bit in enumerate(bits.flat)):064x}"


def assert_disjoint(rows: list[dict]) -> None:
    """Raise if a unit, file hash or perceptual hash appears in two splits."""
    for key in ("unit", "sha256", "phash"):
        seen: dict = {}
        for row in rows:
            previous = seen.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError(f"cross-split {key} duplicate: {row['id']}")
