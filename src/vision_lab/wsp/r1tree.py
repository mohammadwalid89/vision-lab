"""Read-only adapter for the R1 data tree (chapter 11).  The layout below is what the R3 probes
expected; anything that differs is caught by ``preflight``, which fails closed.

Expected layout under ``wsp_root``:
  r1/INTAKE.json                      {"rows": [{source_id, group_id, split, category, label}, ...]}
  r1/SPLIT_MANIFEST.json              {"groups": [{group_id, split}, ...]}
  r1/cache/siglip2_base/index.json    {"status": "READY", "rows": [{id, group_id, split, variant,
                                       label, category, confounded_train, ...}], "arrays":
                                       {"pooled": {"path", "sha256"}, "mean_patch": {...}, ...}}
  r1/masks/<group_id>_object.png      binary 0/255 object mask per source group (384x384)
  r1/MASK_MANIFEST.json               {name: sha256}, {"files": {name: sha256}} or the R1 form
                                       {"groups": {gid: {"object_sha256": ..., ...}}} (all accepted)
Views: R1 stored its rendered views as
lossless WebP, ``r1/data/<split>/<view id>.webp`` (sha256 in the index rows).  Use
``--views-dir <wsp_root>/r1/data``; ``PngDirViews`` also accepts ``<dir>/<view id>.png`` (dry run)
or ``PluginViews('module:function')`` where function(row) -> uint8 (H, W, 3).  Holdout rows are
dropped before anything else and their ids are never decoded.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np

ACTIVE = ("train", "dev")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class R1Tree:
    def __init__(self, wsp_root):
        self.root = Path(wsp_root)
        self.r1 = self.root / "r1"
        self.cache = self.r1 / "cache/siglip2_base"

    def intake_train(self):
        rows = json.loads((self.r1 / "INTAKE.json").read_text())["rows"]
        split = {g["group_id"]: g["split"] for g in json.loads((self.r1 / "SPLIT_MANIFEST.json").read_text())["groups"]}
        return [r for r in rows if r["split"] == "train" and split.get(r["group_id"]) == "train"]

    def view_rows(self):
        idx = json.loads((self.cache / "index.json").read_text())
        if idx.get("status") != "READY":
            raise RuntimeError("R1 cache index not READY")
        intake = json.loads((self.r1 / "INTAKE.json").read_text())["rows"]
        src_of = {r["group_id"]: r["source_id"] for r in intake if r["split"] in ACTIVE}
        out = []
        for r in idx["rows"]:
            if r["split"] not in ACTIVE:
                continue                                     # holdout: dropped, never decoded
            r = dict(r)
            r.setdefault("source_id", src_of[r["group_id"]])
            r.setdefault("confounded_train", 0.0)
            out.append(r)
        return out, idx

    def cached_array(self, idx, name):
        e = idx["arrays"][name]
        p = self.root / e["path"]
        return p, e["sha256"]

    def mask(self, group_id) -> np.ndarray:
        from PIL import Image
        return np.asarray(Image.open(self.r1 / "masks" / f"{group_id}_object.png").convert("L")) > 127

    def mask_manifest(self) -> dict:
        p = self.r1 / "MASK_MANIFEST.json"
        if not p.exists():
            return {}
        d = json.loads(p.read_text())
        if "groups" in d:    # R1 format: {"groups": {gid: {"object_mask_path", "object_sha256", ...}}}
            return {f"{gid}_object.png": e["object_sha256"] for gid, e in d["groups"].items()}
        return d.get("files", d)


class PngDirViews:
    """Directory of rendered views.  Resolution order for a row: ``<id>.png``, ``<split>/<id>.webp``,
    ``<id>.webp`` (R1's tree is ``r1/data`` with ``<split>/<id>.webp``).  When the row
    carries ``sha256`` (the real R1 index does) the file is hashed on every read and a mismatch
    raises: a view that is not byte-identical to the one R1 recorded is never encoded."""
    PATTERNS = ("{id}.png", "{split}/{id}.webp", "{id}.webp")

    def __init__(self, directory, pattern=None):
        self.dir = Path(directory)
        self.patterns = (pattern,) if pattern else self.PATTERNS

    def path(self, row):
        cands = [self.dir / p.format(**row) for p in self.patterns]
        for c in cands:
            if c.exists():
                return c
        return cands[0]

    def exists(self, row):
        return self.path(row).exists()

    def __call__(self, row):
        from PIL import Image
        p = self.path(row)
        if row.get("sha256") and sha256_file(p) != row["sha256"]:
            raise RuntimeError(f"view {row['id']}: sha256 differs from the R1 index; refusing to encode")
        return np.asarray(Image.open(p).convert("RGB"))


class PluginViews:
    def __init__(self, spec: str):
        mod, fn = spec.split(":")
        self.fn = getattr(importlib.import_module(mod), fn)

    def exists(self, row):
        return True

    def __call__(self, row):
        a = np.asarray(self.fn(row))
        if a.dtype != np.uint8 or a.ndim != 3:
            raise ValueError("view plugin must return uint8 (H, W, 3)")
        return a
