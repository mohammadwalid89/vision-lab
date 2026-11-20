#!/usr/bin/env python3
"""Sparse class-row gain edit of YOLOv10s, with a random-direction placebo.

The edit multiplies the output rows of the classes the detector under-recalls
by exp(lambda_c) in the final class-logit convolutions (`model.23.cv3.*.2`).
No gradient, no optimizer. Lambda comes from the median confidence of the
near-miss boxes (single shot, no sweep). The original checkpoint is never
written; edited and placebo copies go to --work-dir.

Both copies are run on the same images and scored with the same matcher.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_var] = "1"
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from vision_lab.weights.accounting import require_unchanged, sha256_array, sha256_path  # noqa: E402
from vision_lab.weights.detection_metrics import (  # noqa: E402
    detection_metrics,
    image_pairs,
    iou,
    parse_visdrone_annotation,
)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

SEED = 20261104
CONF = 0.25          # metric confidence threshold
IOU_T = 0.5
NEAR_IOU = 0.30      # a near miss overlaps a ground-truth box but scores below CONF
TARGET_SCORE = 0.30  # confidence the edit aims for on near misses
MAX_LAMBDA = 1.5
CLASS_ROW_PATTERN = re.compile(r"model\.23\.cv3\.\d+\.2\.weight")


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return float(np.log(p / (1.0 - p)))


def infer_rows(model, rows: list[dict]) -> list[dict]:
    """Predictions at conf 0.05 (the metric then thresholds at CONF)."""
    out = []
    for row in rows:
        with Image.open(row["image_path"]) as image:
            rgb = image.convert("RGB")
        result = model.predict(source=rgb, imgsz=640, conf=0.05, iou=0.7, max_det=300,
                               device="cpu", verbose=False, save=False)[0]
        boxes = [{"xyxy": [float(x) for x in box], "score": float(score),
                  "source_class_id": int(cls), "source_class_name": model.names[int(cls)]}
                 for box, score, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(),
                                            result.boxes.cls.tolist())]
        out.append({"id": row["id"], "annotation_path": row["annotation_path"],
                    "predictions": boxes})
    return out


def near_misses(rows: list[dict]) -> dict:
    """Best same-class score among overlapping predictions, for each missed object."""
    near: dict[str, list] = {}
    for row in rows:
        gts = [g for g in parse_visdrone_annotation(row["annotation_path"])
               if not g["ignored"] and g["target"]]
        preds = [p for p in row["predictions"] if p["score"] >= CONF]
        used = set()
        for gt in gts:
            cls = gt["target"]
            best, best_i = 0.0, None
            for pi, p in enumerate(preds):
                if pi in used or p.get("source_class_name") != cls:
                    continue
                v = iou(gt["xyxy"], p["xyxy"])
                if v > best:
                    best, best_i = v, pi
            if best >= IOU_T:
                used.add(best_i)
            else:
                cand = [p["score"] for p in row["predictions"]
                        if p.get("source_class_name") == cls and iou(gt["xyxy"], p["xyxy"]) >= NEAR_IOU]
                near.setdefault(cls, []).append(max(cand) if cand else None)
    return near


def choose_edit(per_class: dict, near: dict, min_gt: int = 8, max_recall: float = 0.5, top_k: int = 3):
    """Under-recalled classes (gt >= 8, recall <= 0.5), at most top_k by false negatives."""
    eligible = {c: v for c, v in per_class.items()
                if v["gt"] >= min_gt and v["recall"] is not None and v["recall"] <= max_recall}
    eligible = dict(sorted(eligible.items(), key=lambda kv: -(kv[1]["gt"] - kv[1]["tp"]))[:top_k])
    targets, skipped = {}, {}
    for cls, v in per_class.items():
        if cls not in eligible:
            continue
        scores = [s for s in near.get(cls, []) if s is not None]
        if not scores:
            skipped[cls] = "no overlapping prediction at any confidence"
            continue
        s_med = float(np.median(scores))
        lam = float(min(max(logit(TARGET_SCORE) - logit(s_med), 0.05), MAX_LAMBDA))
        targets[cls] = {"near_miss_n": len(scores), "median_near_miss_score": round(s_med, 4),
                        "lambda": round(lam, 4), "gain": round(float(np.exp(lam)), 4),
                        "baseline_recall": v["recall"], "gt": v["gt"],
                        "fn_before": int(v["gt"] - v["tp"])}
    return targets, skipped


def save_copy(model, state: dict, path: Path) -> None:
    model.model.load_state_dict(state, strict=True)
    # saved as the in-memory FP32 module: YOLO.save() would cast to FP16
    torch.save({**model.ckpt, "model": model.model, "ema": None}, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("YOLOV10S_CKPT", "yolov10s.pt")))
    parser.add_argument("--eval-dir", type=Path, required=True,
                        help="VisDrone split folder with images/ and annotations/")
    parser.add_argument("--eval-ids", type=Path, default=None,
                        help="optional text file, one image id per line (the frozen subset)")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--out", type=Path, default=Path("results/class_gain_edit.json"))
    args = parser.parse_args()

    from ultralytics import YOLO

    started = time.process_time()
    ids = args.eval_ids.read_text().split() if args.eval_ids else None
    rows = sorted(image_pairs(args.eval_dir, ids), key=lambda r: r["id"])
    args.work_dir.mkdir(parents=True, exist_ok=True)

    sha_before = sha256_path(args.checkpoint)
    model = YOLO(str(args.checkpoint))
    baseline_rows = infer_rows(model, rows)
    baseline = detection_metrics(baseline_rows)

    per_class = {c: {"gt": v["gt"], "tp": v["tp"], "recall": v["recall"]}
                 for c, v in baseline["per_class"].items() if v["gt"]}
    targets, skipped = choose_edit(per_class, near_misses(baseline_rows))

    names = model.names
    name_to_id = {v: k for k, v in names.items()}
    target_ids = {name_to_id[c]: c for c in targets if c in name_to_id}
    state = {k: v.detach().clone() for k, v in model.model.state_dict().items()}
    changed, per_tensor = [], {}
    for key, tensor in list(state.items()):
        if not CLASS_ROW_PATTERN.fullmatch(key) or tensor.dim() != 4 or tensor.shape[0] != len(names):
            continue
        delta = torch.zeros_like(tensor)
        for cls_id, cls in target_ids.items():
            delta[cls_id] = tensor[cls_id] * (float(np.exp(targets[cls]["lambda"])) - 1.0)
        if not target_ids:
            continue
        state[key] = tensor + delta
        changed.append(key)
        per_tensor[key] = {"shape": list(tensor.shape), "classes_touched": len(target_ids),
                           "delta_l2": round(float(delta.norm()), 6),
                           "relative_delta_l2": round(float(delta.norm() / tensor.norm()), 6)}
    if not changed:
        raise SystemExit("no class-logit tensor matched; refusing a no-op edit")

    n_params = int(sum(v.numel() for v in state.values() if hasattr(v, "numel")))
    n_changed = int(sum(len(target_ids) * int(state[k][0].numel()) for k in changed))
    delta_l2 = float(sum(per_tensor[k]["delta_l2"] ** 2 for k in changed) ** 0.5)

    edit_ckpt = args.work_dir / "yolov10s_sparse_class_repair.pt"
    save_copy(model, state, edit_ckpt)
    require_unchanged(args.checkpoint, sha_before)
    reloaded = YOLO(str(edit_ckpt))
    reload_ok = all(torch.equal(reloaded.model.state_dict()[k].cpu(), state[k].cpu().float()) for k in changed)
    reload_hashes_match = (
        {k: sha256_array(reloaded.model.state_dict()[k].numpy()) for k in changed}
        == {k: sha256_array(state[k].numpy()) for k in changed})

    # Placebo: random noise on the same rows, per-row norm = the tensor's delta L2 divided by
    # the number of target rows. As run for the reported result, the noise is added to the
    # edited weights, not to the original ones (see README, Status).
    rng = np.random.default_rng(SEED + 1)
    placebo_state = dict(state)
    for key in changed:
        base = state[key].clone()
        delta = torch.zeros_like(base)
        for cls_id in target_ids:
            row_norm = float(per_tensor[key]["delta_l2"]) / max(1, len(target_ids))
            r = torch.from_numpy(rng.standard_normal(tuple(base[cls_id].shape)).astype(np.float32))
            delta[cls_id] = r / max(float(r.norm()), 1e-9) * row_norm
        placebo_state[key] = base + delta
    placebo_ckpt = args.work_dir / "yolov10s_sparse_placebo.pt"
    save_copy(model, placebo_state, placebo_ckpt)

    edited_rows = infer_rows(YOLO(str(edit_ckpt)), rows)
    placebo_rows = infer_rows(YOLO(str(placebo_ckpt)), rows)
    edited, placebo = detection_metrics(edited_rows), detection_metrics(placebo_rows)

    def deltas(arm: dict) -> dict:
        return {m: None if arm[m] is None or baseline[m] is None else round(arm[m] - baseline[m], 6)
                for m in ("precision", "recall", "tp", "fp", "fn", "count_mae")}

    repairs = {}
    for cls in targets:
        b, e = baseline["per_class"][cls], edited["per_class"][cls]
        repairs[cls] = {"fn_before": b["fn"], "fn_after": e["fn"], "repaired": max(0, b["fn"] - e["fn"]),
                        "tp_before": b["tp"], "tp_after": e["tp"], "fp_before": b["fp"], "fp_after": e["fp"]}

    keys = ("precision", "recall", "tp", "fp", "fn", "count_mae")
    result = {
        "schema": "vision-weight-edit-pilot2-1",
        "model": "yolov10s",
        "images": len(rows),
        "source_checkpoint_unchanged": sha256_path(args.checkpoint) == sha_before,
        "reload_ok": reload_ok, "reload_tensor_hashes_match": reload_hashes_match,
        "diagnosis": {"per_class_baseline": per_class, "target_classes": targets, "skipped_classes": skipped},
        "edit": {"changed_tensors": changed, "per_tensor": per_tensor,
                 "classes_touched": target_ids, "changed_output_rows": len(target_ids) * len(changed),
                 "model_parameters": n_params, "changed_parameter_count": n_changed,
                 "changed_parameter_fraction": round(n_changed / n_params, 8),
                 "delta_l2": round(delta_l2, 6)},
        "measurements": {
            "baseline": {k: baseline[k] for k in keys},
            "edited": {k: edited[k] for k in keys},
            "placebo": {k: placebo[k] for k in keys},
            "edited_minus_baseline": deltas(edited),
            "placebo_minus_baseline": deltas(placebo),
            "target_repairs": repairs,
            "images_with_changed_target_outputs": sum(
                int([(p["source_class_name"], round(p["score"], 3)) for p in b["predictions"]
                     if p["source_class_name"] in targets]
                    != [(p["source_class_name"], round(p["score"], 3)) for p in e["predictions"]
                        if p["source_class_name"] in targets])
                for b, e in zip(baseline_rows, edited_rows))},
        "cpu_s_total": round(time.process_time() - started, 1),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"targets": targets, "skipped": skipped,
                      "changed_parameter_fraction": result["edit"]["changed_parameter_fraction"],
                      "baseline": result["measurements"]["baseline"],
                      "edited": result["measurements"]["edited"],
                      "placebo": result["measurements"]["placebo"]}, indent=2))


if __name__ == "__main__":
    main()
