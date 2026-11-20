"""Download a fixed per-category cohort from a public Hugging Face mirror and write manifest.json.

Per category: up to 40 normal and 10 anomaly training images, 20 separate normal
calibration images, 10 normal and 15 anomaly test images. Later-split rows that
share a file hash or perceptual hash with an earlier split are dropped.
The source datasets are not redistributed; see the repo-level dataset notes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import requests

from vision_lab.cascade.hashing import assert_disjoint, phash

SOURCES = {
    "mvtec_ad": {
        "repo": "foersben/mvtec-ad",
        "source": "https://www.mvtec.com/research-teaching/datasets/mvtec-ad",
        "license": "CC-BY-NC-SA-4.0",
        "attribution": "MVTec Software GmbH; Bergmann et al. (2019, 2021)",
    },
    "visa": {
        "repo": "imaadd05/visa-anomaly-detection",
        "source": "https://arxiv.org/abs/2207.14315",
        "license": "CC-BY-4.0",
        "attribution": "VisA dataset authors; Zou et al. (2022)",
    },
    "mvtec_loco": {
        "repo": "Rafa612/mvtec_loco",
        "source": "https://www.mvtec.com/company/research/datasets/mvtec-loco",
        "license": "CC-BY-NC-SA-4.0",
        "attribution": "MVTec Software GmbH; Bergmann et al., MVTec LOCO AD (2022)",
    },
}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def spread(paths: list[str], count: int) -> list[str]:
    """Pick `count` paths evenly spaced over the sorted list."""
    paths = sorted(paths)
    if len(paths) < count:
        raise ValueError("source category has too few images for the fixed cohort")
    if count == 1:
        return [paths[0]]
    return [paths[(i * (len(paths) - 1)) // (count - 1)] for i in range(count)]


def select(source: str, category: str, files: list[str]) -> list[tuple]:
    def under(prefix: str, suffixes=(".png",)):
        return sorted(n for n in files if n.startswith(prefix) and n.lower().endswith(suffixes))

    if source in ("mvtec_ad", "mvtec_loco"):
        train_good = under(category + "/train/good/")
        test_good = under(category + "/test/good/")
        anomaly = sorted(n for n in under(category + "/test/") if "/good/" not in n)
        good_train = train_good[:40]
        if source == "mvtec_loco":
            good_cal = under(category + "/validation/good/")[:20]
        else:
            good_cal = train_good[40:60]
        good_test = test_good[:10]
    else:
        normal = under(category + "/Data/Images/Normal/", IMAGE_SUFFIXES)
        anomaly = under(category + "/Data/Images/Anomaly/", IMAGE_SUFFIXES)
        good_train = normal[:40]
        good_cal = normal[len(normal) // 2:len(normal) // 2 + 20]
        good_test = normal[-10:]
    anomalies = spread(anomaly, 25)
    cohorts = (("train", "OK", good_train), ("train", "EXCEPTION", anomalies[:10]),
               ("calibration", "OK", good_cal), ("test", "OK", good_test),
               ("test", "EXCEPTION", anomalies[10:]))
    picked = []
    for split, truth, paths in cohorts:
        if len(paths) < (20 if split == "calibration" else 10):
            raise ValueError(f"too few {split} images for {category}")
        picked.extend((category, split, truth, path) for path in paths)
    return picked


def build(source: str, root: Path) -> dict:
    info = SOURCES[source]
    root.mkdir(parents=True, exist_ok=True)
    meta = requests.get("https://huggingface.co/api/datasets/" + info["repo"], timeout=30)
    meta.raise_for_status()
    meta = meta.json()
    revision = meta["sha"]
    files = [entry["rfilename"] for entry in meta["siblings"]]
    categories = sorted({n.split("/")[0] for n in files
                         if "/" in n and n.split("/")[0] not in ("results", "split_csv")})
    selected = [item for category in categories for item in select(source, category, files)]

    rows = []
    for category, split, truth, name in selected:
        destination = root / name
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://huggingface.co/datasets/{info['repo']}/resolve/{revision}/{quote(name)}"
            response = requests.get(url, timeout=90)
            response.raise_for_status()
            destination.write_bytes(response.content)
        if truth == "OK":
            defect_type = "good"
        else:
            defect_type = "anomaly" if source == "visa" else name.split("/")[2]
        rows.append({
            "id": source + "-" + name.replace("/", "-").rsplit(".", 1)[0],
            "unit": source + ":" + name,
            "category": category, "defect_type": defect_type, "split": split, "truth": truth,
            "path": name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "phash": phash(destination),
        })

    kept, seen, dropped = [], {"sha256": {}, "phash": {}}, []
    for row in rows:
        collision = next((k for k in seen if row[k] in seen[k] and seen[k][row[k]] != row["split"]), None)
        if collision:
            dropped.append({"id": row["id"], "reason": "cross-split " + collision})
            continue
        kept.append(row)
        for key in seen:
            seen[key].setdefault(row[key], row["split"])
    assert_disjoint(kept)
    for category in categories:
        test = [r for r in kept if r["category"] == category and r["split"] == "test"]
        if sum(r["truth"] == "OK" for r in test) < 5 or sum(r["truth"] == "EXCEPTION" for r in test) < 10:
            raise ValueError(f"perceptual-hash exclusion leaves too few test scenes in {category}")
    manifest = {
        "dataset": source, "source": info["source"],
        "mirror": "https://huggingface.co/datasets/" + info["repo"],
        "revision": revision, "license": info["license"], "attribution": info["attribution"],
        "split_rule": ("per-category fixed scene paths: up to 40 normal and 10 anomaly training images, "
                       "20 separate normal calibration images, 10 normal and 15 anomaly test images; "
                       "later split rows sharing a source or perceptual hash with an earlier split are "
                       "excluded; original datasets do not provide physical unit IDs"),
        "categories": categories, "dropped_cross_split_duplicates": dropped, "rows": kept,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=sorted(SOURCES))
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory the images and manifest.json are written to")
    args = parser.parse_args()
    manifest = build(args.source, args.data_root)
    print(f"{args.source}: {len(manifest['rows'])} rows, revision {manifest['revision']}")


if __name__ == "__main__":
    main()
