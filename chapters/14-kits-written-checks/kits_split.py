"""Deterministic split for the kits track. The holdout is sealed here.

Scope: MVTec LOCO AD logical anomalies and good images, categories juice_bottle and
splicing_connectors. Structural anomalies are never enumerated.

Normals: train/good -> `fit_normal` (dictionary, memory bank, standardisers);
validation/good -> `calib_normal` (operating point only).
Evaluation images are ordered inside each (category, kind) stratum by
sha256(SEED_STRING|relative_path): first 40% train, next 30% dev, the rest holdout.
Only train and dev rows can be loaded or opened.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

SEED = 20261112
SEED_STRING = "VISION-KITS-PROG-20261112"
CATS = ["juice_bottle", "splicing_connectors"]
IN_SCOPE = {
    "fit_normal": ("train", "good"),
    "calib_normal": ("validation", "good"),
    "eval_good": ("test", "good"),
    "eval_logical": ("test", "logical_anomalies"),
}
EVAL_ROLES = ("eval_good", "eval_logical")
FRACTIONS = {"train": .40, "dev": .30}


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def image_id(relpath):
    return hashlib.sha256(("kid|" + relpath).encode()).hexdigest()[:16]


def enumerate_rows(root):
    root = Path(root)
    rows = []
    for cat in CATS:
        for role, (src, kind) in IN_SCOPE.items():
            d = root / cat / src / kind
            if not d.is_dir():
                raise SystemExit(f"missing source directory: {d}")
            for name in sorted(p.name for p in d.iterdir() if p.is_file()):
                rel = f"{cat}/{src}/{kind}/{name}"
                p = root / rel
                rows.append({"id": image_id(rel), "category": cat, "role": role,
                             "kind": "logical" if role == "eval_logical" else "good",
                             "source_split": src, "relpath": rel,
                             "bytes": p.stat().st_size, "sha256": sha_file(p)})
    return rows


def assign_split(rows):
    buckets = {}
    for r in rows:
        if r["role"] in EVAL_ROLES:
            buckets.setdefault((r["category"], r["kind"]), []).append(r)
        else:
            r["split"] = "train"
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda r: sha_bytes(
            (SEED_STRING + "|" + r["relpath"]).encode()))
        n = len(ordered)
        n_train = int(round(FRACTIONS["train"] * n))
        n_dev = int(round(FRACTIONS["dev"] * n))
        for i, r in enumerate(ordered):
            r["split"] = ("train" if i < n_train
                          else "dev" if i < n_train + n_dev else "holdout")
            r["split_rank"] = i
        if len({r["split"] for r in ordered}) != 3:
            raise SystemExit(f"degenerate allocation for {key}")
    return rows


def load_split(manifest_path, split):
    """Only train and dev are loadable; the holdout branch raises first."""
    if split not in ("train", "dev"):
        raise PermissionError("holdout is sealed")
    m = json.loads(Path(manifest_path).read_text())
    rows = [r for r in m["images"] if r["split"] == split]
    if not rows:
        raise RuntimeError(f"no rows for split {split}")
    return rows


def open_image(row, dataset_root):
    if row["split"] not in ("train", "dev"):
        raise PermissionError("holdout row: refusing to open")
    root = Path(dataset_root).resolve()
    p = (root / row["relpath"]).resolve()
    if not p.is_relative_to(root):
        raise PermissionError(f"path escape: {p}")
    from PIL import Image
    with Image.open(p) as im:
        return im.convert("RGB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-root", default=os.environ.get("LOCO_ROOT"),
                    help="MVTec LOCO root (default: $LOCO_ROOT)")
    ap.add_argument("--out", required=True, help="directory for the split manifest")
    a = ap.parse_args()
    if not a.dataset_root:
        ap.error("--dataset-root or LOCO_ROOT is required")

    rows = assign_split(enumerate_rows(a.dataset_root))
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate ids")
    if len({r["sha256"] for r in rows}) != len(rows):
        raise SystemExit("duplicate image bytes (leak across splits)")

    summary = {}
    for r in rows:
        k = f"{r['category']}/{r['role']}/{r['split']}"
        summary[k] = summary.get(k, 0) + 1
    manifest = {
        "dataset": "MVTec LOCO AD", "license": "CC BY-NC-SA 4.0 (non-commercial)",
        "seed": SEED, "seed_string": SEED_STRING, "fractions": FRACTIONS,
        "summary": summary,
        "counts_by_split": {s: sum(1 for r in rows if r["split"] == s)
                            for s in ("train", "dev", "holdout")},
        "images": [{k: r[k] for k in ("id", "category", "role", "kind", "source_split",
                                     "split", "relpath", "sha256", "bytes")}
                   for r in rows],
    }
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "SPLIT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"summary": summary, "counts_by_split": manifest["counts_by_split"]},
                     indent=1))


if __name__ == "__main__":
    main()
