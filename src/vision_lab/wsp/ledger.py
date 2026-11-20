"""Attempt ledger and resource guard.

Attempt ledger: one immutable recipe per slot; attempts are
slot x k in {8, 32, 128}; every attempt contains all three seeds and its complete controls;
no fallbacks; k = 32 alone decides each slot's primary conclusion.  Registering a second
recipe for a slot, an undeclared k, or re-running a completed attempt with different
inputs raises.  Resume is allowed only for an identical (slot, k, recipe, inputs) hash.

Resource guard: free-space floor (4 GiB), R3 footprint cap (5 GB, counted on the declared
output directory), RSS ceiling, and CPU affinity <= 3 cores.  Violations raise
``ResourceStop`` -> the attempt is INFRA_BLOCKED / INCONCLUSIVE, never a scientific negative.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
from dataclasses import dataclass, field
from pathlib import Path

DECLARED_K = (8, 32, 128)
PRIMARY_K = 32


class LedgerError(RuntimeError):
    pass


class ResourceStop(RuntimeError):
    pass


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


@dataclass
class AttemptLedger:
    path: Path | None = None
    declared_k: tuple = DECLARED_K
    recipes: dict = field(default_factory=dict)     # slot -> recipe sha
    attempts: dict = field(default_factory=dict)    # "slot|k" -> dict(status, inputs_sha)

    def register_recipe(self, slot: str, recipe: dict) -> str:
        h = _sha(recipe)
        if slot in self.recipes and self.recipes[slot] != h:
            raise LedgerError(f"slot {slot!r} already has an immutable recipe; fallbacks are not allowed")
        self.recipes[slot] = h
        self._save()
        return h

    def begin(self, slot: str, k: int, inputs: dict) -> str:
        if slot not in self.recipes:
            raise LedgerError("register the recipe before any attempt")
        if k not in self.declared_k:
            raise LedgerError(f"k={k} not declared")
        key = f"{slot}|{k}"
        ih = _sha(dict(recipe=self.recipes[slot], inputs=inputs))
        prev = self.attempts.get(key)
        if prev and prev["inputs_sha"] != ih:
            raise LedgerError(f"{key} exists with different inputs; replacement attempts are not allowed")
        if prev and prev["status"] in ("PASS", "NULL_OR_NEGATIVE"):
            raise LedgerError(f"{key} already completed")
        self.attempts[key] = dict(status="RUNNING", inputs_sha=ih)
        self._save()
        return ih

    def finish(self, slot: str, k: int, status: str) -> None:
        if status not in ("PASS", "NULL_OR_NEGATIVE", "INCONCLUSIVE", "INFRA_BLOCKED"):
            raise LedgerError(status)
        self.attempts[f"{slot}|{k}"]["status"] = status
        self._save()

    def primary_conclusion(self, slot: str) -> str | None:
        a = self.attempts.get(f"{slot}|{PRIMARY_K}")
        return None if a is None else a["status"]

    def _save(self):
        if self.path is not None:
            Path(self.path).write_text(json.dumps(dict(recipes=self.recipes, attempts=self.attempts), indent=2))

    @classmethod
    def load(cls, path):
        p = Path(path)
        if not p.exists():
            return cls(path=p)
        d = json.loads(p.read_text())
        return cls(path=p, recipes=d["recipes"], attempts=d["attempts"])


def dir_bytes(path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


@dataclass
class ResourceGuard:
    out_dir: Path
    free_floor_bytes: int = 4 * 2**30
    footprint_cap_bytes: int = 5 * 10**9
    rss_ceiling_bytes: int = 8 * 2**30
    max_cores: int = 3

    def check(self, tag: str = "", pending_bytes: int = 0) -> dict:
        du = shutil.disk_usage(self.out_dir if Path(self.out_dir).exists() else "/")
        if du.free - pending_bytes < self.free_floor_bytes:
            raise ResourceStop(f"{tag}: free space would fall below the floor")
        used = dir_bytes(self.out_dir) if Path(self.out_dir).exists() else 0
        if used + pending_bytes > self.footprint_cap_bytes:
            raise ResourceStop(f"{tag}: R3 footprint cap exceeded ({used + pending_bytes} B)")
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        if rss >= self.rss_ceiling_bytes:
            raise ResourceStop(f"{tag}: peak RSS {rss} B over ceiling")
        cores = len(os.sched_getaffinity(0))
        if cores > self.max_cores:
            raise ResourceStop(f"{tag}: affinity {cores} cores > {self.max_cores}; run under taskset")
        return dict(tag=tag, free_bytes=du.free, footprint_bytes=used, peak_rss_bytes=rss, cores=cores)
