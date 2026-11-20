"""List the public datasets the chapters use, with their licences, and fetch the ones that can be fetched by script.

No dataset is stored in this repository. MVTec AD and MVTec LOCO AD are CC BY-NC-SA 4.0: they must be downloaded
from the publisher by each user and must never be redistributed, so this script only prints where to get them.

    python scripts/fetch_data.py                     # list datasets, licences and target folders
    python scripts/fetch_data.py --get cifar10 eurosat --root data
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    key: str
    name: str
    chapters: str
    licence: str
    redistribute: str
    source: str
    env: str = ""
    torchvision: str = ""  # torchvision.datasets class name, if it can be fetched by script


DATASETS = [
    Dataset("mvtec_ad", "MVTec AD", "01 02 03 06 10",
            "CC BY-NC-SA 4.0 (non-commercial). Cite Bergmann et al., CVPR 2019 and IJCV 2021.",
            "No. Never redistribute images, features or derived images.",
            "https://www.mvtec.com/research-teaching/datasets/mvtec-ad (publisher download form)",
            env="VISION_COHORT_DIR"),
    Dataset("mvtec_loco", "MVTec LOCO AD", "01 02 04 05 10 11-15",
            "CC BY-NC-SA 4.0 (non-commercial). Cite Bergmann et al., IJCV 2022.",
            "No. Never redistribute images, masks or derived images.",
            "https://www.mvtec.com/company/research/datasets/mvtec-loco (publisher download form)",
            env="LOCO_ROOT"),
    Dataset("visa", "VisA (Visual Anomaly)", "01 11 12 13",
            "CC BY 4.0. Cite Zou et al., SPot-the-Difference, ECCV 2022.",
            "Allowed with attribution; not shipped here.",
            "The release archive linked from the paper (arXiv 2207.14315); extract to <root>/visa"),
    Dataset("total_text", "Total-Text", "01",
            "BSD-3-Clause (Ch'ng, Chan, Liu).",
            "Allowed with the licence notice; not shipped here.",
            "https://github.com/cs-chan/Total-Text-Dataset"),
    Dataset("visdrone", "VisDrone2019-DET (train, val)", "07",
            "Academic, non-commercial. The exact terms of the 2019 archives were not verified.",
            "No.",
            "https://github.com/VisDrone/VisDrone-Dataset"),
    Dataset("kth", "KTH Actions (walking, jogging, running)", "08",
            "Research use; no explicit licence text on the publisher page.",
            "No. Clips must not be redistributed.",
            "KTH CVAP recognition-of-human-actions page (three zip files)",
            env="KTH_CLIPS_DIR"),
    Dataset("cifar10", "CIFAR-10", "16",
            "No formal licence; cite Krizhevsky (2009).", "No; fetch by script.",
            "torchvision", torchvision="CIFAR10"),
    Dataset("eurosat", "EuroSAT (RGB)", "16",
            "MIT (dataset release); Sentinel-2 imagery is Copernicus open data. Verify on the release page.",
            "Not needed; fetch by script.", "torchvision", torchvision="EuroSAT"),
    Dataset("gtsrb", "GTSRB", "16",
            "No formal licence; research use, citation requested.", "No; fetch by script.",
            "torchvision", torchvision="GTSRB"),
    Dataset("pets", "Oxford-IIIT Pets", "16",
            "CC BY-SA 4.0. Verify on the VGG page.", "Not needed; fetch by script.",
            "torchvision", torchvision="OxfordIIITPet"),
    Dataset("flowers102", "Oxford Flowers 102", "16 (v1 only; excluded from the counted tasks)",
            "No formal licence; research use. Cite Nilsback and Zisserman (2008). Verify on the VGG page.",
            "Not needed; fetch by script.", "torchvision", torchvision="Flowers102"),
    Dataset("dtd", "Describable Textures Dataset", "16 (features extracted; no task built)",
            "Research use only. Cite Cimpoi et al., CVPR 2014. Verify on the VGG page.",
            "Not needed; fetch by script.", "torchvision", torchvision="DTD"),
]


def list_datasets(root: Path) -> None:
    for d in DATASETS:
        print(f"{d.key:11s} {d.name}  [chapters {d.chapters}]")
        print(f"            licence: {d.licence}")
        print(f"            redistribute: {d.redistribute}")
        how = "python scripts/fetch_data.py --get " + d.key if d.torchvision else d.source
        print(f"            get: {how}")
        print(f"            put in: {root / d.key}" + (f"  (or set {d.env})" if d.env else ""))


def fetch_torchvision(d: Dataset, root: Path) -> None:
    import torchvision.datasets as tvd

    cls = getattr(tvd, d.torchvision)
    target = root / d.key
    target.mkdir(parents=True, exist_ok=True)
    if d.torchvision == "CIFAR10":
        cls(target, train=True, download=True)
        cls(target, train=False, download=True)
    elif d.torchvision in ("GTSRB", "DTD"):
        cls(target, split="train", download=True)
    elif d.torchvision == "Flowers102":
        cls(target, split="test", download=True)
    elif d.torchvision == "OxfordIIITPet":
        cls(target, split="trainval", download=True)
        cls(target, split="test", download=True)
    else:
        cls(target, download=True)
    print(f"{d.key}: downloaded to {target}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("data"), help="where datasets go (never inside git)")
    ap.add_argument("--get", nargs="*", default=None, metavar="KEY", help="datasets to fetch by script")
    args = ap.parse_args()

    by_key = {d.key: d for d in DATASETS}
    if not args.get:
        list_datasets(args.root)
        return
    for key in args.get:
        d = by_key.get(key)
        if d is None:
            raise SystemExit(f"unknown dataset {key!r}; choose from {', '.join(by_key)}")
        if not d.torchvision:
            print(f"{key}: manual download only ({d.licence}). Get it from: {d.source}")
            continue
        fetch_torchvision(d, args.root)


if __name__ == "__main__":
    main()
