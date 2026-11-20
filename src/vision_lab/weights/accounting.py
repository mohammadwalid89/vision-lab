"""Hashing and exact accounting of weight edits (what changed, by how much)."""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def sha256_path(path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(a) -> str:
    import numpy as np
    a = np.ascontiguousarray(np.asarray(a))
    return hashlib.sha256(str(a.shape).encode() + str(a.dtype).encode() + a.tobytes()).hexdigest()


def tensor_sha(t) -> str:
    import torch
    a = t.detach().cpu().contiguous()
    if a.dtype == torch.bfloat16:
        a = a.float()
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()


def norm_stats(base: dict, edited: dict) -> dict:
    """Per-tensor and total change between two dicts of tensors with the same keys."""
    out = {"tensors": [], "total_params": 0, "changed_nonzero": 0,
           "delta_l1": 0.0, "delta_l2": 0.0, "delta_max_abs": 0.0, "base_l2": 0.0}
    for k in base:
        b, e = base[k].detach().float(), edited[k].detach().float()
        d = e - b
        nonzero = int((d != 0).sum().item())
        dl2 = float(d.norm().item())
        bl2 = float(b.norm().item())
        max_abs = float(d.abs().max().item()) if d.numel() else 0.0
        out["tensors"].append({
            "name": k, "shape": list(b.shape), "params": int(b.numel()),
            "changed_nonzero": nonzero,
            "changed_frac_of_tensor": round(nonzero / b.numel(), 8),
            "delta_l1": float(d.abs().sum().item()), "delta_l2": dl2,
            "delta_max_abs": max_abs, "base_l2": bl2,
            "rel_l2": dl2 / bl2 if bl2 > 0 else float("inf"),
            "base_sha256": tensor_sha(base[k]), "edited_sha256": tensor_sha(edited[k]),
        })
        out["total_params"] += int(b.numel())
        out["changed_nonzero"] += nonzero
        out["delta_l1"] += float(d.abs().sum().item())
        out["delta_l2"] += dl2
        out["delta_max_abs"] = max(out["delta_max_abs"], max_abs)
        out["base_l2"] += bl2
    out["delta_total_l2"] = math.sqrt(sum(t["delta_l2"] ** 2 for t in out["tensors"]))
    return out


def require_unchanged(path: Path, sha_before: str) -> None:
    if sha256_path(path) != sha_before:
        raise RuntimeError(f"source checkpoint changed: {path.name}")
