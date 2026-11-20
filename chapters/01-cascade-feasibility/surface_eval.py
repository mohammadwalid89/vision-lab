"""Surface / pack-completeness arm: ResNet-18 features, calibrated logistic head, good-unit memory.

Per category and seed, the head scores the test images. A case is released as
EXCEPTION when the head probability is >= T_NOK, released as OK when it is < T_OK and
the nearest-good distance stays below the 98th percentile of the calibration
normals, and otherwise goes to a person. Wrong automatic accepts are labelled
exceptions released as OK.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from vision_lab.cascade.metrics import summarize, wilson

SEEDS = (11, 23, 47)
T_OK, T_NOK = 0.2, 0.8
ANOMALY_QUANTILE = 0.98
RESNET18_FILE = "checkpoints/resnet18-f37072fd.pth"


def embed(manifest: dict, root: Path):
    import torch
    from torchvision.models import ResNet18_Weights, resnet18

    torch.set_num_threads(2)
    weights = ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    transform = weights.transforms()
    vectors, latencies = [], []
    for row in manifest["rows"]:
        begin = time.perf_counter()
        with Image.open(root / row["path"]) as image:
            tensor = transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            vector = backbone(tensor).flatten().numpy()
        vectors.append(vector / max(np.linalg.norm(vector), 1e-9))
        latencies.append(round((time.perf_counter() - begin) * 1000, 2))
    return np.stack(vectors), latencies


def decide(detector: float, anomaly_score: float, t_a: float):
    if detector >= T_NOK:
        return "EXCEPTION"
    if detector < T_OK and anomaly_score < t_a:
        return "OK"
    return None


def run_seed(manifest: dict, vectors, latencies, seed: int) -> dict:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.linear_model import LogisticRegression

    rows = manifest["rows"]
    by_category = defaultdict(list)
    for index, row in enumerate(rows):
        by_category[row["category"]].append(index)
    cases, per_category = [], {}
    for category in manifest["categories"]:
        ids = by_category[category]
        train = [i for i in ids if rows[i]["split"] == "train"]
        calibration = [i for i in ids if rows[i]["split"] == "calibration"]
        test = [i for i in ids if rows[i]["split"] == "test"]
        labels = np.array([int(rows[i]["truth"] == "EXCEPTION") for i in train])
        rng = np.random.default_rng(seed)
        selected = np.concatenate([
            rng.choice(np.flatnonzero(labels == label), size=int(sum(labels == label) * 0.8), replace=False)
            for label in (0, 1)])
        head = CalibratedClassifierCV(LogisticRegression(max_iter=500, random_state=seed),
                                      method="sigmoid", cv=3)
        head.fit(vectors[np.array(train)[selected]], labels[selected])
        probabilities = head.predict_proba(vectors[test])[:, 1]
        good = vectors[[i for i in train if rows[i]["truth"] == "OK"]]
        anomaly = 1 - np.max(vectors[ids] @ good.T, axis=1)
        score_by_id = dict(zip(ids, anomaly))
        t_a = float(np.quantile([score_by_id[i] for i in calibration], ANOMALY_QUANTILE))
        false_alarm = sum(score_by_id[i] > t_a for i in calibration)
        category_cases = []
        for i, probability in zip(test, probabilities):
            verdict = decide(float(probability), float(score_by_id[i]), t_a)
            case = {"id": rows[i]["id"], "truth": rows[i]["truth"], "category": category,
                    "state": "resolved" if verdict else "queued", "verdict": verdict,
                    "resolved_by": "L1" if verdict else "HUMAN", "latency_ms": latencies[i],
                    "neurons": 0, "correct": None if verdict is None else verdict == rows[i]["truth"]}
            cases.append(case)
            category_cases.append(case)
        per_category[category] = {
            "metrics": summarize(category_cases), "t_a": t_a,
            "calibration_false_alarm": int(false_alarm), "calibration_good_n": len(calibration),
            "calibration_false_alarm_wilson95": wilson(int(false_alarm), len(calibration)),
        }
    return {"seed": seed, "metrics": summarize(cases), "categories": per_category}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory holding manifest.json and the images (from build_manifest.py)")
    parser.add_argument("--out", type=Path, required=True, help="summary JSON to write")
    parser.add_argument("--torch-home", type=Path, default=os.environ.get("TORCH_HOME"),
                        help="torch cache holding the ImageNet ResNet-18 weights (default: $TORCH_HOME)")
    args = parser.parse_args()
    if args.torch_home:
        os.environ["TORCH_HOME"] = str(args.torch_home)
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

    manifest = json.loads((args.data_root / "manifest.json").read_text())
    vectors, latencies = embed(manifest, args.data_root)
    import torch

    weights_file = Path(torch.hub.get_dir()) / RESNET18_FILE
    backbone_sha = hashlib.sha256(weights_file.read_bytes()).hexdigest()
    runs = [run_seed(manifest, vectors, latencies, seed) for seed in SEEDS]
    summary = {
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "categories": manifest["categories"], "backbone_sha256": backbone_sha, "runs": runs,
        "limitation": ("experimental fusion; source scene paths and perceptual hashes disjoint, "
                       "physical unit identity unavailable"),
    }
    args.out.write_text(json.dumps(summary, indent=2))
    for run in runs:
        m = run["metrics"]
        print(f"seed {run['seed']}: n={m['n']} resolved={m['resolved']} wrong_auto_accept={m['wrong_auto_accept']}")


if __name__ == "__main__":
    main()
