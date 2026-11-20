"""Greedy, class-matched detection metrics for VisDrone annotations.

Annotation rows are `left,top,width,height,score,category,truncation,occlusion`.
Categories outside the six supported classes, and boxes with score 0, are
ignored regions: predictions that sit on them are dropped, not counted as
false positives.
"""
from __future__ import annotations

from pathlib import Path

SUPPORTED_CLASSES = ("person", "bicycle", "car", "truck", "bus", "motorcycle")
VISDRONE_TO_TARGET = {1: "person", 2: "person", 3: "bicycle", 4: "car",
                      6: "truck", 9: "bus", 10: "motorcycle"}
EXCLUDED_VISDRONE = {0, 5, 7, 8, 11}
# COCO ids of the supported classes (the detector's native label space)
COCO_ID = {"person": 0, "bicycle": 1, "car": 2, "motorcycle": 3, "bus": 5, "truck": 7}
DENSITY_ALERT_COUNT = 8


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter)


def intersection_over_prediction(pred, region) -> float:
    ix1, iy1 = max(pred[0], region[0]), max(pred[1], region[1])
    ix2, iy2 = min(pred[2], region[2]), min(pred[3], region[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area = max(0.0, pred[2] - pred[0]) * max(0.0, pred[3] - pred[1])
    return inter / area if area > 0 else 0.0


def parse_visdrone_annotation(path: Path) -> list[dict]:
    objects = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        left, top, width, height, score, category = [int(float(x)) for x in line.split(",")[:6]]
        objects.append({
            "xyxy": [left, top, left + width, top + height],
            "category": category,
            "ignored": score == 0 or category in EXCLUDED_VISDRONE,
            "target": VISDRONE_TO_TARGET.get(category),
        })
    return objects


def detection_metrics(rows: list[dict], iou_threshold: float = 0.5,
                      conf_threshold: float = 0.25) -> dict:
    """Each row: `annotation_path` and `predictions` (xyxy, score, source_class_name)."""
    per_class = {c: {"tp": 0, "fp": 0, "fn": 0, "gt": 0} for c in SUPPORTED_CLASSES}
    count_errors = []
    density = {"alerts": 0, "false_alerts": 0, "missed_alerts": 0}
    for row in rows:
        gt_objects = parse_visdrone_annotation(row["annotation_path"])
        targets = [g for g in gt_objects if not g["ignored"] and g["target"]]
        ignored_regions = [g["xyxy"] for g in gt_objects if g["ignored"]]
        preds = []
        for p in row["predictions"]:
            if p.get("score", 0.0) < conf_threshold:
                continue
            if p.get("source_class_name") not in SUPPORTED_CLASSES and any(
                    intersection_over_prediction(p["xyxy"], r) >= 0.5 for r in ignored_regions):
                continue
            preds.append(p)
        matched_gt = set()
        for cls in SUPPORTED_CLASSES:
            cls_gts = [(i, g) for i, g in enumerate(targets) if g["target"] == cls]
            cls_preds = sorted([p for p in preds if p.get("source_class_name") == cls],
                               key=lambda p: -p.get("score", 0.0))
            per_class[cls]["gt"] += len(cls_gts)
            for pred in cls_preds:
                best_i, best_iou = None, 0.0
                for i, gt in cls_gts:
                    if i in matched_gt:
                        continue
                    value = iou(pred["xyxy"], gt["xyxy"])
                    if value > best_iou:
                        best_i, best_iou = i, value
                if best_i is not None and best_iou >= iou_threshold:
                    matched_gt.add(best_i)
                    per_class[cls]["tp"] += 1
                else:
                    per_class[cls]["fp"] += 1
            per_class[cls]["fn"] += sum(1 for i, _g in cls_gts if i not in matched_gt)
        gt_count = len(targets)
        pred_count = sum(1 for p in preds if p.get("source_class_name") in SUPPORTED_CLASSES)
        count_errors.append(abs(pred_count - gt_count))
        gt_alert, pred_alert = gt_count >= DENSITY_ALERT_COUNT, pred_count >= DENSITY_ALERT_COUNT
        density["alerts"] += int(pred_alert)
        density["false_alerts"] += int(pred_alert and not gt_alert)
        density["missed_alerts"] += int(gt_alert and not pred_alert)
    tp = sum(v["tp"] for v in per_class.values())
    fp = sum(v["fp"] for v in per_class.values())
    fn = sum(v["fn"] for v in per_class.values())
    return {
        "n": len(rows),
        "iou_threshold": iou_threshold,
        "conf_threshold": conf_threshold,
        "tp": tp, "fp": fp, "fn": fn,
        "precision": round(tp / (tp + fp), 6) if tp + fp else None,
        "recall": round(tp / (tp + fn), 6) if tp + fn else None,
        "count_mae": round(sum(count_errors) / len(count_errors), 6) if count_errors else None,
        "density_alert_threshold": DENSITY_ALERT_COUNT,
        "density": density,
        "per_class": {c: {**v,
                          "recall": round(v["tp"] / (v["tp"] + v["fn"]), 6) if v["tp"] + v["fn"] else None,
                          "precision": round(v["tp"] / (v["tp"] + v["fp"]), 6) if v["tp"] + v["fp"] else None}
                      for c, v in per_class.items()},
    }


def image_pairs(directory: Path, ids: list[str] | None = None) -> list[dict]:
    """Image/annotation pairs of a VisDrone split folder (`images/`, `annotations/`)."""
    directory = Path(directory)
    rows = []
    for image in sorted((directory / "images").glob("*.jpg")):
        annotation = directory / "annotations" / (image.stem + ".txt")
        if annotation.is_file():
            rows.append({"id": image.stem, "image_path": str(image), "annotation_path": str(annotation)})
    if ids is not None:
        by_id = {r["id"]: r for r in rows}
        rows = [by_id[i] for i in ids]
    return rows


def predict_rows(model, rows: list[dict], conf: float = 0.01) -> list[dict]:
    """Run an Ultralytics model on image paths and return metric-ready rows."""
    out = []
    for r in rows:
        res = model.predict(source=r["image_path"], imgsz=640, conf=conf, iou=0.7,
                            max_det=300, device="cpu", verbose=False, save=False)[0]
        out.append({"id": r["id"], "image_path": r["image_path"],
                    "annotation_path": r["annotation_path"],
                    "predictions": [{"xyxy": [float(x) for x in b], "score": float(s),
                                     "source_class_id": int(c), "source_class_name": model.names[int(c)]}
                                    for b, s, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                                       res.boxes.cls.tolist())]})
    return out


def alert_count(preds: list[dict], conf: float = 0.25) -> int:
    return sum(1 for r in preds for p in r["predictions"]
               if p["score"] >= conf and p["source_class_name"] in SUPPORTED_CLASSES)


def threshold_for_k(preds: list[dict], k: int) -> float:
    """Confidence at which exactly k supported-class alerts remain (-1 if fewer exist)."""
    scores = sorted((p["score"] for r in preds for p in r["predictions"]
                     if p["source_class_name"] in SUPPORTED_CLASSES), reverse=True)
    return float(scores[k]) if len(scores) > k else -1.0


def per_image_counts(preds: list[dict], conf: float = 0.25, iou_threshold: float = 0.5) -> dict:
    out = {}
    for row in preds:
        gts = [g for g in parse_visdrone_annotation(row["annotation_path"])
               if not g["ignored"] and g["target"] in SUPPORTED_CLASSES]
        ranked = sorted((p for p in row["predictions"]
                         if p["score"] >= conf and p["source_class_name"] in SUPPORTED_CLASSES),
                        key=lambda p: -p["score"])
        matched, tp, fp = set(), 0, 0
        for p in ranked:
            best_i, best_v = None, 0.0
            for i, g in enumerate(gts):
                if i in matched or g["target"] != p["source_class_name"]:
                    continue
                v = iou(p["xyxy"], g["xyxy"])
                if v > best_v:
                    best_i, best_v = i, v
            if best_i is not None and best_v >= iou_threshold:
                matched.add(best_i)
                tp += 1
            else:
                fp += 1
        out[row["id"]] = {"tp": tp, "fp": fp, "fn": len(gts) - len(matched)}
    return out


def regression_vs_reference(arm: dict, reference: dict) -> dict:
    """Images where an arm gained more false positives than true positives, and the reverse."""
    worse = sum(1 for i in arm if arm[i]["fp"] - reference[i]["fp"] > arm[i]["tp"] - reference[i]["tp"])
    better = sum(1 for i in arm if arm[i]["tp"] - reference[i]["tp"] > arm[i]["fp"] - reference[i]["fp"])
    return {"images_worse": worse, "images_better": better,
            "dtp": sum(arm[i]["tp"] - reference[i]["tp"] for i in arm),
            "dfp": sum(arm[i]["fp"] - reference[i]["fp"] for i in arm),
            "dfn": sum(arm[i]["fn"] - reference[i]["fn"] for i in arm)}
