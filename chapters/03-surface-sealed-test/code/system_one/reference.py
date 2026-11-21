"""Calibration references, evidence table and fixed data splits.

Built once from the recorded research rows (`features.attach_features`, published as
chapter 02's features.py) plus the live gate embeddings, and frozen to state/reference.json.

Splits (fixed by SHA hash, independent of labels' values):
  seen categories (6 AD + 3 LOCO):  router_train -> correction pool (simulated corrections)
                                     dev_audit    -> 50% validation (promotion gate) / 50% audit (never used by the gate)
  new categories (9 AD):             test images -> 50% validation / 50% pool; support anomalies -> pool
Calibration normals are never used for learning or evaluation; they are the conformal reference.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import config

CHAPTER02 = Path(__file__).resolve().parents[3] / "02-two-layer-audit"


def _bucket(sha: str, salt: str) -> float:
    return int(hashlib.sha256((salt + sha).encode()).hexdigest()[:8], 16) / 2**32


def conformal_p(score: float, ref_sorted: np.ndarray) -> float:
    """Upper-tail conformal p-value: (#ref >= s + 1)/(n + 1)."""
    n = len(ref_sorted)
    return float((n - np.searchsorted(ref_sorted, score, side="left") + 1) / (n + 1))


def slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def _mad(v):
    v = np.asarray(v, float)
    med = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - med)))
    return med, (mad if mad > 1e-9 else (float(v.std()) or 1.0))


def build(gate_npz=None, gate_margin=1.5):
    sys.path.insert(0, str(CHAPTER02))
    from features import attach_features

    rows, _ = attach_features(include_heldout=True)
    cats = {}
    for c in config.CATEGORIES:
        cal = [r for r in rows if r["category"] == c and r["role"] == "calibration"]
        e = {"family": config.family(c), "n_cal": len(cal), "cal_patch": sorted(float(r["patch_score"]) for r in cal)}
        e["patch_med"], e["patch_mad"] = _mad(e["cal_patch"])
        if c in config.LOCO:
            pairs = [[float(r["patch_score"]), float(r["semantic"])] for r in cal if r.get("semantic") is not None]
            e["cal_pairs"] = pairs  # paired (patch, semantic novelty) per calibration image
            e["sem_med"], e["sem_mad"] = _mad([q for _, q in pairs])
        cats[c] = e
    table = []
    for r in rows:
        c = r["category"]
        if c not in cats or r["role"] in ("calibration", "fit"):
            continue
        if c in config.HELD9:
            if r["role"] == "support_anomaly":
                split = "pool"
            elif r["role"] in ("test_good", "test_anomaly"):
                split = "validation" if _bucket(r["sha256"], "val:") < 0.5 else "pool"
            else:
                continue
        else:
            if r["role"] == "router_train":
                split = "pool"
            elif r["role"] == "dev_audit":
                split = "validation" if _bucket(r["sha256"], "val:") < 0.5 else "audit"
            else:
                continue
        e = cats[c]
        item = {
            "sha256": r["sha256"],
            "category": c,
            "family": e["family"],
            "split": split,
            "role": r["role"],
            "stratum": r["stratum"],
            "y": int(r["truth_bin"]),
            "source": f"{r['dataset']}/{r['source']}",
            "patch": float(r["patch_score"]),
            "z_patch": (float(r["patch_score"]) - e["patch_med"]) / e["patch_mad"],
        }
        if c in config.LOCO:
            item["sem"] = float(r["semantic"])
            item["z_sem"] = (item["sem"] - e["sem_med"]) / e["sem_mad"]
        table.append(item)
    gate = {}
    if gate_npz is not None:
        z = np.load(gate_npz)
        for c in config.CATEGORIES:
            cal = sorted(
                (r for r in rows if r["category"] == c and r["role"] == "calibration"), key=lambda r: r["sha256"]
            )
            V = np.stack([z[r["sha256"]] for r in cal if r["sha256"] in z.files])
            mu = V.mean(0)
            mu /= np.linalg.norm(mu)
            d = 1 - V @ mu
            gate[c] = {
                "centroid": mu.round(6).tolist(),
                "radius": float(gate_margin * d.max()),
                "cal_max_distance": float(d.max()),
                "n": int(len(V)),
            }
    return {
        "schema": "system-one-reference-1",
        "categories": cats,
        "table": table,
        "gate": gate,
        "split_rule": __doc__.split("Splits")[1].strip(),
    }


class Reference:
    def __init__(self, data: dict):
        self.data = data
        self.cats = data["categories"]
        self.table = data["table"]
        self.by_sha = {r["sha256"]: r for r in self.table}

    @classmethod
    def load(cls, path=None):
        return cls(json.loads((path or config.STATE / "reference.json").read_text()))

    def z_patch(self, c, patch):
        e = self.cats[c]
        return (patch - e["patch_med"]) / e["patch_mad"]

    def z_sem(self, c, sem):
        e = self.cats[c]
        return (sem - e["sem_med"]) / e["sem_mad"]

    def p1(self, c, patch, extra=()):
        ref = np.sort(np.concatenate([self.cats[c]["cal_patch"], list(extra)]))
        return conformal_p(patch, ref)

    def calibration_pairs(self, c, extra=()):
        """(z_patch, z_sem) for each LOCO calibration normal, plus promoted confirmed-good corrections."""
        e = self.cats[c]
        P = np.asarray(list(e["cal_pairs"]) + [[x[0], x[1]] for x in extra if x[1] is not None], float)
        return (P[:, 0] - e["patch_med"]) / e["patch_mad"], (P[:, 1] - e["sem_med"]) / e["sem_mad"]
