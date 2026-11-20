#!/usr/bin/env python3
"""Run Mirage on a few VisDrone images and store its structured output.

This is the preflight that decided the task: can Mirage's object records serve a
box-detection benchmark on aerial images? One CPU call takes minutes, so pass few ids.
Requires the `cactus` package (Mirage) and its weights.
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from mirage_common import VISDRONE_TO_TARGET

KEYWORDS = ["person", "pedestrian", "car", "van", "truck", "bus", "motorcycle", "motorbike",
            "bicycle", "tricycle"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ids", nargs="+", help="VisDrone image ids (file stems)")
    parser.add_argument("--pool-dir", type=Path, required=True,
                        help="VisDrone train folder with images/ and annotations/")
    parser.add_argument("--out", type=Path, default=Path("results/preflight_probe.json"))
    args = parser.parse_args()

    import cactus

    mirage = cactus.Mirage()
    out = {}
    for image_id in args.ids:
        image = args.pool_dir / "images" / f"{image_id}.jpg"
        gt = []
        for line in (args.pool_dir / "annotations" / f"{image_id}.txt").read_text().splitlines():
            p = [int(float(x)) for x in line.strip().split(",")[:8]]
            if len(p) >= 6 and p[4] != 0 and p[5] in VISDRONE_TO_TARGET:
                gt.append(VISDRONE_TO_TARGET[p[5]])
        start = time.time()
        try:
            result = mirage.describe(str(image), keywords=KEYWORDS,
                                     sections=["objects", "relations", "text"],
                                     fields=["bbox", "score"], conf=0.05, max_objects=32)
            error = None
        except Exception as exc:
            result, error = None, repr(exc)
        seconds = round(time.time() - start, 1)
        out[image_id] = {"seconds": seconds, "gt_counts": {c: gt.count(c) for c in set(gt)},
                         "n_gt": len(gt), "error": error, "output": result}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1))
        n_objects = None if result is None else len(result.get("objects", []))
        print(image_id, "gt", len(gt), "mirage_objs", n_objects, "s", seconds, "err", error, flush=True)
    print("rss_mb", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


if __name__ == "__main__":
    main()
