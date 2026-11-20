"""Frozen constants and helpers for the Mirage reader-loop preflight."""
from __future__ import annotations

import hashlib
from pathlib import Path

CLASSES = ["person", "bicycle", "car", "truck", "bus", "motorcycle"]

# VisDrone categories: 0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van,
# 6 truck, 7 tricycle, 8 awning-tricycle, 9 bus, 10 motor. Van and tricycle are outside
# the six-class task, as in the detector study (chapter 07).
VISDRONE_TO_TARGET = {1: "person", 2: "person", 3: "bicycle", 4: "car",
                      6: "truck", 9: "bus", 10: "motorcycle"}

# Exact normalised names and simple plurals only. Not mapped on purpose: vehicle,
# transport, van, minivan, tricycle, cart, box.
SYNONYMS = {
    "person": "person", "persons": "person", "people": "person",
    "pedestrian": "person", "pedestrians": "person", "man": "person",
    "woman": "person", "child": "person", "human": "person",
    "bicycle": "bicycle", "bicycles": "bicycle", "bike": "bicycle", "bikes": "bicycle",
    "car": "car", "cars": "car", "automobile": "car", "automobiles": "car",
    "sedan": "car", "sedans": "car", "hatchback": "car", "suv": "car", "taxi": "car",
    "truck": "truck", "trucks": "truck", "lorry": "truck", "lorries": "truck",
    "pickup": "truck", "pickup truck": "truck",
    "bus": "bus", "buses": "bus", "coach": "bus", "coaches": "bus",
    "motorcycle": "motorcycle", "motorcycles": "motorcycle", "motorbike": "motorcycle",
    "motorbikes": "motorcycle", "motor": "motorcycle", "motors": "motorcycle",
    "scooter": "motorcycle", "scooters": "motorcycle", "moped": "motorcycle",
}
CONF_THRESHOLD = 0.25   # presence threshold
IOU_THRESHOLD = 0.5
DEV_TARGET = 24
TRAIN_TARGET = 12
SPLIT_SALT = "mirage-mechinterv-20261106"
# Images whose Mirage output was inspected during the preflight; never in dev or train.
PROBED_IDS = {"0000209_00609_d_0000221", "0000157_00451_d_0000082", "0000126_11844_d_0000130"}


def sha256_path(p: Path) -> str:
    digest = hashlib.sha256()
    with open(p, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def image_ids(pool_dir: Path) -> list[str]:
    return sorted(p.stem for p in (Path(pool_dir) / "images").glob("*.jpg"))


def group_of(image_id: str) -> str:
    """Scene group: the first underscore-separated field of a VisDrone file name."""
    return image_id.split("_")[0]


PROBED_GROUPS = {group_of(i) for i in PROBED_IDS}


def parse_gt(annotation_path: Path) -> list[tuple[str, list[int]]]:
    """(target class, xyxy) for the six-class task; ignored regions and other classes dropped."""
    out = []
    for line in Path(annotation_path).read_text().splitlines():
        if not line.strip():
            continue
        left, top, w, h, score, cat = [int(float(x)) for x in line.strip().split(",")[:6]]
        if score == 0 or cat not in VISDRONE_TO_TARGET:
            continue
        out.append((VISDRONE_TO_TARGET[cat], [left, top, left + w, top + h]))
    return out


def gt_presence(pool_dir: Path, image_id: str):
    gts = parse_gt(Path(pool_dir) / "annotations" / f"{image_id}.txt")
    return {c: int(any(gc == c for gc, _ in gts)) for c in CLASSES}, len(gts)


def make_split(pool_dir: Path):
    """Scene-group-disjoint dev/train split.

    Groups are ordered by sha256(salt|group) and filled greedily (dev first, then
    train); groups touched by the preflight are skipped.
    """
    ids = image_ids(pool_dir)
    groups: dict[str, list[str]] = {}
    for i in ids:
        groups.setdefault(group_of(i), []).append(i)
    ordered = sorted(groups, key=lambda g: hashlib.sha256(f"{SPLIT_SALT}|{g}".encode()).hexdigest())
    dev, train = [], []
    for g in ordered:
        members = sorted(groups[g])
        if g in PROBED_GROUPS:
            continue
        if len(dev) + len(members) <= DEV_TARGET:
            dev.extend(members)
        elif len(train) + len(members) <= TRAIN_TARGET:
            train.extend(members)
    return sorted(dev), sorted(train), sorted(groups)


def map_name(name: str):
    return SYNONYMS.get(" ".join(str(name).strip().lower().split()))


def box_xyxy_pixels(bbox, width, height):
    """Mirage boxes are in 0-1000 image coordinates."""
    x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
    return [x1 * width / 1000.0, y1 * height / 1000.0, x2 * width / 1000.0, y2 * height / 1000.0]


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def detections_from_record(record: dict, conf: float = CONF_THRESHOLD) -> list[dict]:
    """Target-class detections of one Mirage record (mapped names, score >= conf)."""
    out = []
    width, height = record["width"], record["height"]
    for i, o in enumerate(record["output"].get("objects", [])):
        cls = map_name(o.get("name", ""))
        score = float(o.get("score", 0.0))
        bbox = o.get("bbox")
        if cls is None or score < conf or not (isinstance(bbox, list) and len(bbox) >= 4):
            continue
        xyxy = box_xyxy_pixels(bbox, width, height)
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        out.append({"object_index": i, "class": cls, "name": o.get("name"), "score": score, "xyxy": xyxy})
    return out
