"""Run surface_eval on a generated toy cohort (random-texture images, no real data).

Checks that the pipeline runs end to end and that the manifest/summary shapes hold.
The numbers it prints say nothing about defect detection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import surface_eval
from PIL import Image

from vision_lab.cascade.hashing import phash

SPLITS = (("train", "OK", 40), ("train", "EXCEPTION", 10), ("calibration", "OK", 20),
          ("test", "OK", 10), ("test", "EXCEPTION", 15))


def make_manifest(root: Path, categories=("cat_a", "cat_b"), seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for category in categories:
        for split, truth, count in SPLITS:
            for k in range(count):
                name = f"{category}/{split}/{truth.lower()}_{k:03d}.png"
                pixels = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
                if truth == "EXCEPTION":
                    pixels[16:48, 16:48] = 255
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(pixels).save(path)
                rows.append({"id": name.replace("/", "-"), "unit": name, "category": category,
                             "defect_type": "toy", "split": split, "truth": truth, "path": name,
                             "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "phash": phash(path)})
    manifest = {"dataset": "synthetic", "revision": "none", "categories": list(categories), "rows": rows}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        manifest = make_manifest(root, seed=args.seed)
        vectors, latencies = surface_eval.embed(manifest, root)
        run = surface_eval.run_seed(manifest, vectors, latencies, seed=11)
    metrics = run["metrics"]
    print(f"toy cohort: n={metrics['n']} resolved={metrics['resolved']} "
          f"wrong_auto_accept={metrics['wrong_auto_accept']}")


if __name__ == "__main__":
    main()
