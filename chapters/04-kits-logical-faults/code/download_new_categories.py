"""Obtain pinned public MVTec LOCO mirror categories for isolated Phase 3.
Does not read/test any evaluation labels or run any models.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage

from huggingface_hub import snapshot_download

D = Path(__file__).resolve().parent
OUT = D / "out"
OUT.mkdir(exist_ok=True)
repo = "zhaolutuan/0423"
rev = "464875d33af84257cac350ebcf2c4210343f85a3"


def main():
    if disk_usage("/").free < 8 * 2**30:
        raise SystemExit("Insufficient free space for Phase3; no download started")
    print("START", datetime.now(timezone.utc).isoformat(), "HF repo", repo, "pinned revision", rev, flush=True)
    try:
        location = snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            revision=rev,
            allow_patterns=["MVTec-LOCO/juice_bottle/**", "MVTec-LOCO/splicing_connectors/**"],
            local_dir=str(D / "raw"),
            max_workers=2,
        )
    except Exception as e:
        print("ERROR", type(e).__name__, str(e)[:900], flush=True)
        raise
    print("FINISHED DATA DOWNLOAD", location, flush=True)
    # metadata only, no test labels/scores analyzed
    cats = {}
    total = 0
    for cat in ("juice_bottle", "splicing_connectors"):
        root = D / "raw/MVTec-LOCO" / cat
        count = {}
        size = 0
        for split in ("train", "validation", "test"):
            files = [
                x for x in (root / split).rglob("*") if x.is_file() and x.suffix.lower() in (".png", ".jpg", ".jpeg")
            ]
            count[split] = len(files)
            size += sum(x.stat().st_size for x in files)
        cats[cat] = {"images": count, "bytes": size, "license_present": (root / "license.txt").exists()}
        total += size
    if total > 3 * 2**30:
        raise RuntimeError("Data footprint exceeded 3 GiB budget")
    (OUT / "download_manifest.json").write_text(
        json.dumps(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "repository": repo,
                "revision": rev,
                "categories": cats,
                "bytes": total,
            },
            indent=2,
        )
    )
    print("VALIDATED MANIFEST", json.dumps(cats), flush=True)
    (OUT / "DOWNLOAD_COMPLETE").write_text(datetime.now(timezone.utc).isoformat())


if __name__ == "__main__":
    main()
