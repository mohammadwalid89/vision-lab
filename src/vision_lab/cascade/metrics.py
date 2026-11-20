"""Release-decision metrics with explicit denominators.

A case is a dict with the keys: id, truth ("OK" or "EXCEPTION"), verdict ("OK",
"EXCEPTION" or None when no automatic decision was made), correct (bool or
None), state ("resolved" or "queued"), resolved_by, latency_ms and neurons.
"""
from __future__ import annotations

import math
import statistics


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    """Wilson score interval for k successes out of n."""
    if not n:
        return [0, 1]
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(max(0, mid - half), 4), round(min(1, mid + half), 4)]


def summarize(rows: list[dict]) -> dict:
    """Coverage, wrong automatic accepts and latency for one run."""
    n = len(rows)
    resolved = [r for r in rows if r["verdict"] is not None]
    correct = sum(r["correct"] is True for r in resolved)
    wrong_accept = sum(r["truth"] == "EXCEPTION" and r["verdict"] == "OK" for r in rows)
    exceptions = sum(r["truth"] == "EXCEPTION" for r in rows)
    caught = sum(r["truth"] == "EXCEPTION" and r["verdict"] == "EXCEPTION" for r in rows)
    latency = sorted(r["latency_ms"] for r in rows)
    by_layer: dict = {}
    for r in rows:
        by_layer[r["resolved_by"]] = by_layer.get(r["resolved_by"], 0) + 1
    return {
        "n": n,
        "resolved": len(resolved),
        "coverage": round(len(resolved) / n, 4) if n else 0,
        "coverage_ci95": wilson(len(resolved), n),
        "accuracy_resolved": round(correct / len(resolved), 4) if resolved else None,
        "accuracy_ci95": wilson(correct, len(resolved)) if resolved else None,
        "wrong_auto_accept": wrong_accept,
        "wrong_auto_accept_rate_ci95": wilson(wrong_accept, n),
        "wrong_auto_accept_per_10k": round(10000 * wrong_accept / n, 2) if n else 0,
        "exception_recall": round(caught / exceptions, 4) if exceptions else None,
        "exception_recall_ci95": wilson(caught, exceptions) if exceptions else None,
        "queued": sum(r["state"] == "queued" for r in rows),
        "recapture_requested": sum(r["state"] == "recapture_requested" for r in rows),
        "resolved_by": by_layer,
        "latency_p50_ms": round(statistics.median(latency), 2) if n else 0,
        "latency_p95_ms": latency[min(n - 1, math.ceil(0.95 * n) - 1)] if n else 0,
        "neurons_total": round(sum(r["neurons"] for r in rows), 5),
        "label": "measured_public_subset_public_test",
    }
