"""Manifest-only stratified, deterministic design split of previously unused kit categories.
Must run only after DOWNLOAD_COMPLETE. Does not load pixels nor run any model.
"""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

D = Path(__file__).parent
OUT = D / "out"
raw = D / "raw/MVTec-LOCO"
SEED = "VISION-P3-KITS-20261028"


def main():
    assert (OUT / "DOWNLOAD_COMPLETE").exists(), "Download not complete/verified"
    assert not (OUT / "p3_split_manifest.json").exists(), "Manifest already fixed. Do not overwrite"
    assert not (OUT / "p3_dev.json").exists(), "Dev file already exists"
    C = ["juice_bottle", "splicing_connectors"]
    rec = {
        "protocol": "2026-10-28 precommitted 40/60 path-hash split, stratified by category and test stratum",
        "seed": SEED,
        "categories": C,
        "train": [],
        "validation": [],
        "development": [],
        "final": [],
    }
    for cat in C:
        root = raw / cat
        for split in ["train", "validation"]:
            files = sorted(
                x for x in (root / split).rglob("*") if x.is_file() and x.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            assert len(files) > 100 if split == "train" else len(files) > 10, (cat, split, len(files))
            for f in files:
                rec[split].append({"category": cat, "path": str(f), "stratum": "good", "source_split": split})
        groups = defaultdict(list)
        for f in (root / "test").rglob("*"):
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                groups[f.parent.name].append(f)
        assert {"good", "logical_anomalies", "structural_anomalies"} <= set(groups), (cat, list(groups))
        for typ, files in sorted(groups.items()):
            sr = sorted(
                files, key=lambda x: hashlib.sha256((SEED + "|" + str(x.relative_to(raw))).encode()).hexdigest()
            )
            cut = max(1, int(len(sr) * 0.4))
            for i, f in enumerate(sr):
                target = "development" if i < cut else "final"
                rec[target].append({"category": cat, "path": str(f), "stratum": typ, "source_split": "test"})
    # prevent pixel overlap by SHA256 across train, validation, dev and final (check only after split assignment)
    seen = {}
    for k in ["train", "validation", "development", "final"]:
        for r in rec[k]:
            b = Path(r["path"]).read_bytes()
            h = hashlib.sha256(b).hexdigest()
            r["sha256"] = h
            # As run, the assignment below sits inside the if-branch, so seen stays empty and this check cannot fire.
            if h in seen:
                raise ValueError("Image byte overlap between designated partitions: " + r["path"] + " vs " + seen[h])
                seen[h] = r["path"]
    summary = {
        k: {
            f"{cat}/{typ}": sum(r["category"] == cat and r["stratum"] == typ for r in rec[k])
            for cat in C
            for typ in ["good", "logical_anomalies", "structural_anomalies"]
            if any(r["category"] == cat and r["stratum"] == typ for r in rec[k])
        }
        for k in ["train", "validation", "development", "final"]
    }
    (OUT / "p3_split_manifest.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "seed": SEED,
                "protocol": rec["protocol"],
                "full_manifest_sha256": hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest(),
            },
            indent=2,
        )
    )
    (OUT / "p3_train.json").write_text(json.dumps(rec["train"]))
    (OUT / "p3_validation.json").write_text(json.dumps(rec["validation"]))
    (OUT / "p3_dev.json").write_text(json.dumps(rec["development"]))
    (OUT / "p3_locked_final.json").write_text(json.dumps(rec["final"]))
    (OUT / "p3_locked_final.sha256").write_text(
        hashlib.sha256((OUT / "p3_locked_final.json").read_bytes()).hexdigest() + "  p3_locked_final.json\n"
    )
    print("DATA SPLIT FIXED", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
