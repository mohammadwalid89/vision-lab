"""Committed k-list ordering (copied rule from R2 ``r2_methods.ordered_sources``).

Sources are interleaved good/defect, categories round-robin in sha256('r2|seed|cat')
order, sources within a category in sha256('r2|seed|source_id') order.  The k-list for a
seed is the first k items, so k=8 ⊂ k=32 ⊂ k=128 (prefix property, verified by tests and
by the R3 probe on the real intake).
"""
from __future__ import annotations

import hashlib
from typing import Sequence


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def ordered_sources(rows: Sequence[dict], seed: int, min_per_label: int = 64) -> list:
    ordered = []
    for label in (0, 1):
        cats = sorted({r["category"] for r in rows if r["label"] == label},
                      key=lambda c: _h(f"r2|{seed}|{c}"))
        queues = {c: sorted([r for r in rows if r["label"] == label and r["category"] == c],
                            key=lambda r: _h(f'r2|{seed}|{r["source_id"]}'))
                  for c in cats}
        seq = []
        while any(queues.values()):
            for c in cats:
                if queues[c]:
                    seq.append(queues[c].pop(0))
        ordered.append(seq)
    if min(map(len, ordered)) < min_per_label:
        raise ValueError("insufficient TRAIN capacity")
    return [r for pair in zip(*ordered) for r in pair]


def k_lists(rows: Sequence[dict], seeds: Sequence[int], ks: Sequence[int], min_per_label: int = 64) -> dict:
    """{seed: {k: [source_id, ...]}} with the prefix property checked."""
    out = {}
    for s in seeds:
        seq = [r["source_id"] for r in ordered_sources(rows, s, min_per_label)]
        out[s] = {k: seq[:k] for k in sorted(ks)}
        ks_sorted = sorted(ks)
        for small, big in zip(ks_sorted, ks_sorted[1:]):
            if out[s][big][:small] != out[s][small]:
                raise AssertionError("k lists are not prefixes")
    return out
