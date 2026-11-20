#!/usr/bin/env python3
"""Run Mirage on the frozen dev and train ids and cache one JSON record per image.

Resumable: images with an existing record are skipped. One CPU thread.
Requires the `cactus` package (Mirage) and its weights. No comparison arm was run
on top of these records (see README).
"""
from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from mirage_common import CLASSES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, required=True, help="output of make_manifest.py")
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache/mirage"))
    parser.add_argument("--limit", type=int, default=None, help="run only the first N ids")
    args = parser.parse_args()

    import cactus

    manifest = json.loads(args.manifest.read_text())
    ids = (manifest["split"]["dev_ids"] + manifest["split"]["train_ids"])[:args.limit]
    by_id = {r["id"]: r for r in manifest["rows"]}
    mirage = cactus.Mirage()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    options = {"sections": ["objects", "relations", "text"], "fields": ["bbox", "score"],
               "conf": 0.05, "max_objects": 32, "keywords": CLASSES}
    for n, image_id in enumerate(ids, 1):
        cache = args.cache_dir / f"{image_id}.json"
        if cache.is_file():
            print(f"[{n}/{len(ids)}] cached {image_id}", flush=True)
            continue
        row = by_id[image_id]
        start = time.time()
        record = {"id": image_id, "split": row["split"], "image_sha256": row["image_sha256"],
                  "width": row["width"], "height": row["height"],
                  "mirage_version": getattr(cactus, "__version__", "?"), "options": options}
        try:
            record["output"] = mirage.describe(str(args.pool_dir / row["image_path"]), **options)
            record["error"] = None
        except Exception as exc:
            record["output"], record["error"] = None, repr(exc)
        record["seconds"] = round(time.time() - start, 2)
        record["rss_peak_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)
        cache.write_text(json.dumps(record, indent=1))
        n_objects = len((record["output"] or {}).get("objects", []))
        print(f"[{n}/{len(ids)}] {image_id} {record['seconds']}s objs={n_objects} err={record['error']}", flush=True)


if __name__ == "__main__":
    main()
