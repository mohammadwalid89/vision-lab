#!/usr/bin/env python3
"""Freeze the dev/train image ids and presence labels for the Mirage experiment.

The scored question after the box preflight failed is image-level multi-label
presence: a class is present when at least one non-ignored ground-truth object of
that class exists. Paths in the manifest are relative to --pool-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mirage_common import (
    CLASSES,
    CONF_THRESHOLD,
    SPLIT_SALT,
    SYNONYMS,
    VISDRONE_TO_TARGET,
    group_of,
    gt_presence,
    image_ids,
    make_split,
    sha256_path,
    sha256_text,
)
from PIL import Image


def rows(pool_dir: Path, ids: list[str], split: str) -> list[dict]:
    out = []
    for i in ids:
        image = pool_dir / "images" / f"{i}.jpg"
        annotation = pool_dir / "annotations" / f"{i}.txt"
        presence, n_gt = gt_presence(pool_dir, i)
        with Image.open(image) as im:
            width, height = im.size
        out.append({"id": i, "split": split, "scene_group": group_of(i),
                    "image_path": f"images/{i}.jpg", "annotation_path": f"annotations/{i}.txt",
                    "image_sha256": sha256_path(image), "annotation_sha256": sha256_path(annotation),
                    "width": width, "height": height, "n_gt": n_gt, "gt_presence": presence})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool-dir", type=Path, required=True,
                        help="VisDrone train folder with images/ and annotations/")
    parser.add_argument("--out", type=Path, default=Path("results/manifest.json"))
    args = parser.parse_args()

    dev, train, groups = make_split(args.pool_dir)
    pool = image_ids(args.pool_dir)
    manifest = {
        "task": "image_level_multilabel_presence",
        "classes": CLASSES,
        "visdrone_to_target": {str(k): v for k, v in VISDRONE_TO_TARGET.items()},
        "synonym_map": SYNONYMS,
        "confidence_threshold": CONF_THRESHOLD,
        "pool": {"n": len(pool), "ids_sha256": sha256_text("\n".join(pool)), "n_scene_groups": len(groups)},
        "split": {"salt": SPLIT_SALT, "dev_n": len(dev), "train_n": len(train),
                  "dev_ids": dev, "train_ids": train,
                  "dev_ids_sha256": sha256_text("\n".join(dev)),
                  "train_ids_sha256": sha256_text("\n".join(train)),
                  "unused_n": len(set(pool) - set(dev) - set(train))},
        "rows": rows(args.pool_dir, dev, "dev") + rows(args.pool_dir, train, "train"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    print("dev", len(dev), "train", len(train), "unused", manifest["split"]["unused_n"])
    for split in ("dev", "train"):
        print(split, "presence_positives",
              {c: sum(r["gt_presence"][c] for r in manifest["rows"] if r["split"] == split) for c in CLASSES})


if __name__ == "__main__":
    main()
