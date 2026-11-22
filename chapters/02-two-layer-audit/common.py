"""Loading, metric and output helpers shared by the audit scripts.

Inputs are cached specialist outputs from the earlier v3.1 pipeline plus the
held-out AD scores written by score_heldout_ad.py. Labels are dataset folder
labels: every "feedback" experiment here simulates operator corrections with
that oracle, there is no recorded correction stream.

Environment:
  VISION_V31_ROOT  tree holding results/v31/... score tables and data/public/...
  VISION_OUT       output directory (default ./out); metrics go to $VISION_OUT/metrics
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from vision_lab.metrics import auc  # noqa: F401  (the audit scripts import it from here)

ROOT = Path(os.environ.get("VISION_V31_ROOT", "."))
RES = Path(os.environ.get("VISION_OUT", "out"))
MET = RES / "metrics"

AD_CATS6 = ["bottle", "cable", "hazelnut", "pill", "tile", "transistor"]
LOCO_CATS = ["breakfast_box", "pushpins", "screw_bag"]
AD_HELDOUT9 = ["capsule", "carpet", "grid", "leather", "metal_nut",
               "screw", "toothbrush", "wood", "zipper"]


def write_json(name: str, obj) -> Path:
    MET.mkdir(parents=True, exist_ok=True)
    path = MET / name
    path.write_text(json.dumps(obj, indent=2) + "\n")
    return path


def write_csv(name: str, rows: list[dict]) -> Path:
    MET.mkdir(parents=True, exist_ok=True)
    path = MET / name
    if not rows:
        path.write_text("")
        return path
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def load_local_scores(include_heldout: bool = False) -> list[dict]:
    """One deduplicated row per image from the existing v3.1 local score tables."""
    rows = {}
    for path in sorted(glob.glob(str(ROOT / "results/v31/local/*/*_scores.jsonl"))):
        dataset = Path(path).parent.name
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            r["dataset"] = dataset
            r["truth_bin"] = int(r["truth"] == "not_ok")
            r["is_anomaly"] = int(r["truth"] == "not_ok")
            r["source_path"] = str(ROOT / "data/public" / dataset / r["source"])
            rows[r["sha256"]] = r
    if include_heldout:
        for r in load_heldout_scores():
            r["dataset"] = "mvtec_ad"
            rows.setdefault(r["sha256"], r)
    return list(rows.values())


def load_heldout_scores() -> list[dict]:
    path = RES / "heldout_ad_scores.jsonl"
    if not path.exists():
        # An empty list here would mark held-out images as fresh (chapter 03) and leave the nine
        # held-out categories without calibration, so a missing file is an error.
        raise FileNotFoundError(f"{path} is missing; run score_heldout_ad.py first, or set VISION_OUT to its folder")
    out = []
    for line in path.read_text().splitlines():
        r = json.loads(line)
        r["truth"] = "not_ok" if r["truth"] == "not_ok" else "ok"
        r["truth_bin"] = int(r["truth"] == "not_ok")
        r["is_anomaly"] = int(r["truth"] == "not_ok")
        out.append(r)
    return out


def load_embedding(name: str) -> dict:
    path = RES / f"{name}_embeddings.npz"
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float32) for k in z.files}


def load_kit_scores() -> dict:
    """Local-kit and AnomalyCLIP scores for LOCO, keyed by image SHA."""
    out: dict = {}
    for fam in ("fused", "semantic", "normal"):
        pattern = str(ROOT / f"results/v31/local_kit_{fam}/mvtec_loco/*_scores.jsonl")
        for path in glob.glob(pattern):
            for line in Path(path).read_text().splitlines():
                r = json.loads(line)
                v = r.get("novelty_distance")
                if v is not None:
                    out.setdefault(r["sha256"], {})[fam] = float(v)
    for path in glob.glob(str(ROOT / "results/v31/anomalyclip/mvtec_loco/*.jsonl")):
        for line in Path(path).read_text().splitlines():
            r = json.loads(line)
            v = r.get("anomalyclip_score")
            if v is not None and r.get("sha256"):
                out.setdefault(r["sha256"], {})["anomalyclip"] = float(v)
    return out


def balanced_accuracy(y, pred) -> float:
    y = np.asarray(y, int)
    pred = np.asarray(pred, int)
    vals = []
    for c in (0, 1):
        m = y == c
        if m.sum():
            vals.append((pred[m] == c).mean())
    return float(np.mean(vals)) if vals else float("nan")


def average_precision(y, s) -> float:
    from sklearn.metrics import average_precision_score
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def bootstrap_ci(y, s, metric: str = "auc", n_boot: int = 1000, seed: int = 0, groups=None) -> list[float]:
    """Percentile bootstrap. With groups, resample groups first, then take all their rows."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    rng = np.random.default_rng(seed)
    score = auc if metric == "auc" else average_precision
    vals = []
    if groups is None:
        n = len(y)
        for _ in range(n_boot):
            ix = rng.integers(0, n, n)
            if len(np.unique(y[ix])) < 2:
                continue
            vals.append(score(y[ix], s[ix]))
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        for _ in range(n_boot):
            gs = rng.choice(uniq, len(uniq), replace=True)
            ix = np.concatenate([np.where(groups == g)[0] for g in gs])
            if len(np.unique(y[ix])) < 2:
                continue
            vals.append(score(y[ix], s[ix]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> list[float]:
    from scipy.stats import beta
    if n == 0:
        return [float("nan"), float("nan")]
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]
