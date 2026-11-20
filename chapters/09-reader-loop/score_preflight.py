#!/usr/bin/env python3
"""Score the preflight probe output as a box detector (IoU 0.5, score >= 0.25).

Mirage's object names are mapped to the six target classes with the frozen synonym
table, boxes are converted from its 0-1000 coordinates to pixels, and each predicted
box is matched greedily (highest score first) to an unmatched same-class VisDrone box.

The original run recorded the totals only; this script recomputes them from the probe
file with the frozen mapping and a greedy matcher (see README, Status).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mirage_common import (
    CLASSES,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    detections_from_record,
    iou,
    parse_gt,
)
from PIL import Image


def score_image(pool_dir: Path, image_id: str, probe: dict) -> dict:
    with Image.open(pool_dir / "images" / f"{image_id}.jpg") as image:
        width, height = image.size
    record = {"width": width, "height": height, "output": probe["output"]}
    detections = sorted(detections_from_record(record, CONF_THRESHOLD), key=lambda d: -d["score"])
    gts = parse_gt(pool_dir / "annotations" / f"{image_id}.txt")
    matched = set()
    tp = 0
    for det in detections:
        best, best_iou = None, 0.0
        for gi, (cls, box) in enumerate(gts):
            if gi in matched or cls != det["class"]:
                continue
            value = iou(det["xyxy"], box)
            if value > best_iou:
                best, best_iou = gi, value
        if best is not None and best_iou >= IOU_THRESHOLD:
            matched.add(best)
            tp += 1
    return {"gt_objects": len(gts), "mirage_target_boxes": len(detections), "tp": tp,
            "fp": len(detections) - tp, "fn": len(gts) - tp}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--probe", type=Path, required=True, help="output of preflight_probe.py")
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("results/preflight_boxes.json"))
    args = parser.parse_args()

    probes = json.loads(args.probe.read_text())
    per_image = {i: score_image(args.pool_dir, i, p) for i, p in probes.items() if p["output"]}
    total = {k: sum(v[k] for v in per_image.values()) for k in ("gt_objects", "mirage_target_boxes", "tp", "fp", "fn")}
    result = {"classes": CLASSES, "conf": CONF_THRESHOLD, "iou": IOU_THRESHOLD,
              "per_image": per_image, "total": total}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(total))


if __name__ == "__main__":
    main()
