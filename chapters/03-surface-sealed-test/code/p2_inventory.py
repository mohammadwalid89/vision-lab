"""Phase 2 data inventory: which MVTec AD / LOCO images were ever used by the research or Phase 1.

Every image on disk is hashed (sha256).  An image is EXPOSED if its hash appears in any research row
(chapter 02's local, held-out and kit scores) or in the Phase 1 evidence table; otherwise FRESH.  Fresh test
images form the Phase 2 final evaluation set; nothing is selected or tuned on them.

The research rows are read with chapter 02's loaders (VISION_V31_ROOT, VISION_OUT). The Phase 1 evidence
table is state/reference.json of the v001 application (built by system_one/reference.py; the file is a
run output and is not in this repository)."""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "02-two-layer-audit"))
sys.dont_write_bytecode = True
from common import load_heldout_scores, load_kit_scores, load_local_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-root", default=os.environ.get("VISION_DATA_ROOT", "data/public"), help="holds mvtec_ad/ and mvtec_loco/"
    )
    ap.add_argument("--phase1-reference", required=True, help="state/reference.json of the v001 application")
    args = ap.parse_args()
    DATA = Path(args.data_root)
    exposed = {}
    for r in load_local_scores() + load_heldout_scores():
        if r.get("sha256"):
            exposed[r["sha256"]] = r.get("role") or r.get("split") or "research"
    for s in load_kit_scores():
        exposed.setdefault(s, "kit")
    ref = json.load(open(args.phase1_reference))
    for r in ref["table"]:
        exposed.setdefault(r["sha256"], "phase1_" + r["split"])
    rows = []
    for ds in ("mvtec_ad", "mvtec_loco"):
        for p in sorted(glob.glob(str(DATA / ds / "*/*/*/*.png"))):
            rel = Path(p).relative_to(DATA / ds)
            cat, split, typ = rel.parts[0], rel.parts[1], rel.parts[2]
            h = hashlib.sha256(open(p, "rb").read()).hexdigest()
            rows.append(
                {
                    "dataset": ds,
                    "category": cat,
                    "split": split,
                    "type": typ,
                    "path": p,
                    "sha256": h,
                    "exposed": exposed.get(h),
                }
            )
    dup = Counter(r["sha256"] for r in rows)
    for r in rows:
        r["duplicate_bytes"] = dup[r["sha256"]] > 1
    (HERE / "out").mkdir(exist_ok=True)
    json.dump(rows, open(HERE / "out/inventory.json", "w"))
    summ = defaultdict(Counter)
    for r in rows:
        if r["split"] == "train" and r["dataset"] == "mvtec_ad":
            key = "train"
        else:
            key = r["split"]
        summ[(r["dataset"], key, "good" if r["type"] == "good" else "defect")][
            "fresh" if r["exposed"] is None else "exposed"
        ] += 1
    for k, v in sorted(summ.items()):
        print(k, dict(v))
    print("exposed hashes not found on disk:", len(set(exposed) - {r["sha256"] for r in rows}))
    print("byte-duplicate images:", sum(r["duplicate_bytes"] for r in rows))


if __name__ == "__main__":
    main()
