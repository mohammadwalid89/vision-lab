"""Paths, categories and fixed policy constants for the System One application.

Everything that defines behaviour lives here so a run is fully described by this
file plus the model artifacts it points to (hashed into the manifest)."""

from __future__ import annotations

import os
from pathlib import Path

# VISION_V31_ROOT: the v3.1 build tree (cascade/ package, results/v31/ model artifacts, model_cache*/),
# the same tree chapter 02's loaders read.
VISION = Path(os.environ.get("VISION_V31_ROOT", "."))
# VISION_OUT: chapter 02's output folder; its score_heldout_ad.py writes the held-out PatchCore banks
# to patchcore/ there.
HELDOUT = Path(os.environ.get("VISION_OUT", "out"))
APP = Path(__file__).resolve().parents[1]
STATE = Path(os.environ.get("SYSTEM_ONE_STATE", APP / "state"))

AD6 = ["bottle", "cable", "hazelnut", "pill", "tile", "transistor"]
HELD9 = ["capsule", "carpet", "grid", "leather", "metal_nut", "screw", "toothbrush", "wood", "zipper"]
LOCO = ["breakfast_box", "pushpins", "screw_bag"]
KIT_CATS = ["breakfast_box", "screw_bag", "pushpins"]  # order used by the v3.1 kit model indicators
CATEGORIES = AD6 + HELD9 + LOCO


def family(category: str) -> str:
    return "loco" if category in LOCO else "ad"


def patchcore_path(category: str) -> Path:
    if category in LOCO:
        return VISION / f"results/v31/local/mvtec_loco/{category}_patchcore.pt"
    if category in AD6:
        return VISION / f"results/v31/local/mvtec_ad/{category}_patchcore.pt"
    return HELDOUT / f"patchcore/{category}_patchcore.pt"


SEMANTIC_MODEL = VISION / "results/v31/local_kit_semantic/mvtec_loco/{category}_L15.joblib"
CLIP_RN50 = VISION / "model_cache_v3/torch/hub/clip/RN50.pt"
RESNET18 = VISION / "model_cache/torch/hub/checkpoints/resnet18-f37072fd.pth"

# Decision policy of the v001 application.
# Release thresholds: selected out of sample. They are not a guarantee.
# Hold thresholds: 0.005, i.e. hold automatically only when the part is more anomalous than
# essentially every normal reference.  On calibrated scores the hold threshold is roughly the
# false-hold rate on good parts, so it must sit well below the production defect rate.
POLICY = {
    "ad": {"mode": "l1", "alpha_release": 0.95, "alpha_hold": 0.005},
    "loco": {"mode": "gate", "alpha_release": 0.30, "alpha_hold": 0.005},
}

TORCH_THREADS = int(os.environ.get("SYSTEM_ONE_THREADS", "2"))
