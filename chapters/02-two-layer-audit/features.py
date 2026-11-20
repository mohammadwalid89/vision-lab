"""Feature assembly shared by the audit scripts.

Per-category calibration statistics use only the GOOD calibration role of the same
category, so no defect label sets a threshold. Defect labels are used only where
an experiment explicitly simulates operator corrections on the training pool.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
from common import ROOT, load_heldout_scores, load_kit_scores, load_local_scores

SCORE_KEYS = {"patch_score": "patch", "clip_score": "clip", "semantic": "sem",
              "fused": "fused", "anomalyclip": "aclip"}


def _median_mad(values) -> tuple[float, float]:
    v = np.asarray(values, float)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med))) * 1.4826
    if mad < 1e-9:
        mad = float(np.std(v))
    if mad < 1e-9:
        mad = 1.0
    return med, mad


def attach_features(include_heldout: bool = True):
    """Return (rows, stats): score rows with robust z-scores and recorded CPU costs."""
    rows = load_local_scores()
    if include_heldout:
        rows += load_heldout_scores()
    kit = load_kit_scores()
    for r in rows:
        k = kit.get(r.get("sha256"), {})
        r["semantic"] = k.get("semantic")
        r["fused"] = k.get("fused")
        r["anomalyclip"] = k.get("anomalyclip")
        # CPU seconds recorded by the earlier scoring runs, not re-measured here
        l1 = 0.0
        for field in ("preprocess_cost", "patch_cost", "clip_cost"):
            l1 += float((r.get(field) or {}).get("cpu", 0.0))
        if l1 == 0.0:
            l1 = float((r.get("score_cost") or {}).get("cpu", np.nan))
        r["l1_cpu"] = l1
        r["l2_sem_cpu"] = np.nan
        r["l2_full_cpu"] = np.nan
    costs = load_kit_costs()
    for r in rows:
        c = costs.get(r.get("sha256"))
        if c:
            r["l2_sem_cpu"] = c["semantic"]
            r["l2_full_cpu"] = c["full"]
    stats = {}
    for key in SCORE_KEYS:
        groups: dict = {}
        for r in rows:
            if r.get("role") == "calibration" and r.get(key) is not None:
                groups.setdefault((r["dataset"], r["category"]), []).append(r[key])
        for g, vals in groups.items():
            stats[(g, key)] = _median_mad(vals)
    for r in rows:
        for key, short in SCORE_KEYS.items():
            st = stats.get(((r["dataset"], r["category"]), key))
            value = r.get(key)
            r["z_" + short] = (float(value) - st[0]) / st[1] if (st and value is not None) else np.nan
        r["cal_group"] = (r["dataset"], r["category"])
    return rows, stats


def load_kit_costs() -> dict:
    """Observed CPU seconds per image for the semantic and full second layers."""
    out: dict = {}
    for variant, name in (("semantic", "semantic"), ("fused", "full")):
        pattern = str(ROOT / f"results/v31/local_kit_{variant}/mvtec_loco/*_scores.jsonl")
        for path in glob.glob(pattern):
            for line in Path(path).read_text().splitlines():
                r = json.loads(line)
                c = r.get("cost", {}).get("cpu", 0.0)
                c += r.get("clip_cost", {}).get("cpu", 0.0)
                c += r.get("refinement_physical_cost", {}).get("cpu", 0.0)
                out.setdefault(r["sha256"], {})[name] = float(c)
    return out
