#!/usr/bin/env python3
"""Conventional head-only fine-tune of YOLOv10s as a positive control.

Question: does ordinary adaptation on the fit images move the evaluation images
at all, i.e. is there headroom that class-row edits could have reached?

Only layer 23 (the detection head) is trainable; backbone and neck are frozen.
SGD, batch 4, 640 px, CPU, validation disabled so the evaluation images are not
touched during training. The trained copy is scored with the same matcher at
confidence 0.25 and at an equal alert budget (as many alerts as ORIG at 0.25).

The two runs in the chapter differ only in --epochs/--lr0:
  1 epoch,  lr0 = 1e-3   (head-only run 1)
  3 epochs, lr0 = 3e-4   (head-only run 2)
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("YOLO_OFFLINE", "1")

import cv2  # noqa: E402
import torch  # noqa: E402

from vision_lab.weights.accounting import norm_stats, require_unchanged, sha256_path  # noqa: E402
from vision_lab.weights.detection_metrics import (  # noqa: E402
    alert_count,
    detection_metrics,
    image_pairs,
    per_image_counts,
    predict_rows,
    regression_vs_reference,
    threshold_for_k,
)

torch.set_num_threads(1)

# VisDrone category id -> COCO class id of the detector
VISDRONE_TO_COCO = {1: 0, 2: 0, 3: 1, 4: 2, 6: 7, 9: 5, 10: 3}
CONF, IOU = 0.25, 0.5
KEYS = ("precision", "recall", "tp", "fp", "fn", "count_mae")


def build_dataset(fit: list[dict], root: Path) -> Path:
    """Ultralytics dataset folder (symlinked images, YOLO labels) for the fit images."""
    img_dir, lab_dir = root / "images" / "train", root / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    for r in fit:
        link = img_dir / Path(r["image_path"]).name
        if not link.exists():
            link.symlink_to(Path(r["image_path"]).resolve())
        h, w = cv2.imread(r["image_path"]).shape[:2]
        lines = []
        for line in Path(r["annotation_path"]).read_text().splitlines():
            if not line.strip():
                continue
            left, top, bw, bh, score, cat = [int(float(x)) for x in line.split(",")[:6]]
            if score == 0 or cat not in VISDRONE_TO_COCO:
                continue
            lines.append(f"{VISDRONE_TO_COCO[cat]} {(left + bw / 2) / w:.6f} {(top + bh / 2) / h:.6f} "
                         f"{bw / w:.6f} {bh / h:.6f}")
        (lab_dir / (Path(r["image_path"]).stem + ".txt")).write_text("\n".join(lines) + "\n")
    yaml = root / "data.yaml"
    yaml.write_text(f"path: {root.resolve()}\ntrain: images/train\nval: images/train\nnames:\n"
                    + "".join(f"  {i}: c{i}\n" for i in range(80)))
    return yaml


def summary(metrics: dict) -> dict:
    return {k: metrics[k] for k in KEYS}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("YOLOV10S_CKPT", "yolov10s.pt")))
    parser.add_argument("--fit-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--eval-ids", type=Path, default=None)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--lr0", type=float, required=True)
    parser.add_argument("--k-alerts", type=int, default=None,
                        help="alert budget; default is ORIG's alert count at confidence 0.25")
    parser.add_argument("--directional-ckpt", type=Path, default=None,
                        help="optional edited checkpoint from directional_edit.py to score alongside")
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--out", type=Path, default=Path("results/head_finetune.json"))
    args = parser.parse_args()

    from ultralytics import YOLO

    sha_before = sha256_path(args.checkpoint)
    fit = image_pairs(args.fit_dir)[:args.max_fit]
    ids = args.eval_ids.read_text().split() if args.eval_ids else None
    eval_rows = image_pairs(args.eval_dir, ids)
    yaml = build_dataset(fit, args.work_dir / "dataset")
    print("fit", len(fit), "eval", len(eval_rows), flush=True)

    model = YOLO(str(args.checkpoint))
    n_params = sum(p.numel() for p in model.model.parameters())
    head_params = sum(p.numel() for p in model.model.model[23].parameters())
    base_state = {k: v.detach().clone() for k, v in model.model.state_dict().items()}

    run_dir = args.work_dir / "train"
    model.train(
        data=str(yaml), epochs=args.epochs, imgsz=640, batch=4, device="cpu", workers=0,
        freeze=list(range(23)), val=False, plots=False, verbose=True,
        project=str(run_dir.resolve()), name="run", exist_ok=True, seed=0, deterministic=True,
        optimizer="SGD", lr0=args.lr0, lrf=1.0, momentum=0.9, weight_decay=0.0005,
        warmup_epochs=0.0, mosaic=0.0, mixup=0.0, close_mosaic=0, amp=False, cache=False,
    )
    weights = run_dir / "run" / "weights"
    trained_path = weights / "best.pt" if (weights / "best.pt").is_file() else weights / "last.pt"

    trained = YOLO(str(trained_path))
    # the training data.yaml names classes c0..c79; ids are COCO ids, so restore the names
    trained.model.names = dict(YOLO(str(args.checkpoint)).names)
    state = trained.model.state_dict()
    changed = {k: int(((state[k].float() - base_state[k].float()) != 0).sum())
               for k in state if k in base_state and ((state[k].float() - base_state[k].float()) != 0).any()}
    backbone_changed = {k: v for k, v in changed.items() if not k.startswith("model.23.")}
    head_changed = {k: v for k, v in changed.items() if k.startswith("model.23.")}
    stats = None
    if head_changed:
        stats = norm_stats({k: base_state[k] for k in head_changed}, {k: state[k] for k in head_changed})
        stats["changed_frac_of_model"] = round(stats["changed_nonzero"] / n_params, 8)
        stats["changed_frac_of_head"] = round(stats["changed_nonzero"] / max(head_params, 1), 8)

    preds_head = predict_rows(trained, eval_rows)
    preds_orig = predict_rows(YOLO(str(args.checkpoint)), eval_rows)
    k = args.k_alerts if args.k_alerts is not None else alert_count(preds_orig)

    arms = {"ORIG": preds_orig, "HEAD_FINETUNE": preds_head}
    if args.directional_ckpt is not None:
        arms["DIRECTIONAL_a0.1"] = predict_rows(YOLO(str(args.directional_ckpt)), eval_rows)
    fixed = {name: detection_metrics(p, iou_threshold=IOU, conf_threshold=CONF) for name, p in arms.items()}
    thresholds = {name: threshold_for_k(p, k) for name, p in arms.items()}
    equal = {name: detection_metrics(p, iou_threshold=IOU, conf_threshold=thresholds[name])
             for name, p in arms.items()}

    counts_orig, counts_head = per_image_counts(preds_orig), per_image_counts(preds_head)
    require_unchanged(args.checkpoint, sha_before)
    result = {
        "protocol": {"fit_images": len(fit), "eval_images": len(eval_rows), "epochs": args.epochs,
                     "lr0": args.lr0, "frozen_layers": "0-22", "optimizer": "SGD, batch 4, 640 px, CPU",
                     "matcher": f"IoU {IOU}; confidence {CONF} and equal alert budget K"},
        "training": {"n_model_params": int(n_params), "n_head_params": int(head_params),
                     "changed_tensors_head": len(head_changed),
                     "changed_tensors_backbone": len(backbone_changed),
                     "backbone_unchanged": not backbone_changed,
                     "head_changed_entries": int(sum(head_changed.values())),
                     "head_edit_accounting": stats},
        "source_checkpoint_unchanged": True,
        "results_fixed_conf": {name: summary(m) for name, m in fixed.items()},
        "results_equal_alert": {"K": k, **{name: summary(m) for name, m in equal.items()},
                                "thresholds": thresholds},
        "per_class_head": fixed["HEAD_FINETUNE"]["per_class"],
        "per_image_regression_vs_orig": regression_vs_reference(counts_head, counts_orig),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result["results_fixed_conf"], indent=1))
    print("equal alerts:", json.dumps(result["results_equal_alert"], indent=1))


if __name__ == "__main__":
    main()
