#!/usr/bin/env python3
"""Directional class-row edit of YOLOv10s, tested against controls at equal alerts.

Mechanism under test: adding a bounded component along the fit-split anchor
discriminant (mean class anchor feature minus mean background feature) to each
class row of the final class convolutions raises class logits at object anchors
and lowers them at background anchors.

Gate: before any detection metric is computed, the median object-vs-background
specificity AUC of the edit's logit change on held-out fit images must be at
least 0.55. Otherwise the run stops with the preflight numbers.

Arms on the same evaluation images: ORIG, NEW at three doses (alpha), PLACEBO
(same rows and per-row norm, random direction orthogonal to the edit), FITTED
(per-class logistic regression on the same anchor features, written into the
same rows), CALIB_ONLY (original scores, threshold matched to NEW's alert count)
and, optionally, a previously saved edited checkpoint. All arms are also compared
at an equal alert budget: the threshold at which each arm raises as many alerts as
ORIG does at confidence 0.25.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from vision_lab.weights.accounting import norm_stats, require_unchanged, sha256_path  # noqa: E402
from vision_lab.weights.detection_metrics import (  # noqa: E402
    COCO_ID,
    SUPPORTED_CLASSES,
    alert_count,
    detection_metrics,
    image_pairs,
    parse_visdrone_annotation,
    per_image_counts,
    predict_rows,
    regression_vs_reference,
)

torch.set_num_threads(1)

CLASSES = list(SUPPORTED_CLASSES)
CONF = 0.25
IOU = 0.5
ALPHAS = [0.05, 0.10, 0.20]
CLASS_CONVS = [f"model.23.cv3.{i}.2" for i in range(3)]


def mets(preds, conf=CONF):
    return detection_metrics(preds, iou_threshold=IOU, conf_threshold=conf)


def load_gt(row):
    return [g for g in parse_visdrone_annotation(row["annotation_path"])
            if not g["ignored"] and g["target"] in COCO_ID]


def collect_anchor_features(model, rows, handler):
    """Run the model with forward-pre-hooks on the three class convs.

    `handler(level, feature_map, strides, orig_hw, gts)` is called per image/level.
    """
    model.predictor = None  # force a fresh predictor so the hooks are inside it
    det = model.model.model[23]
    strides = [float(s) for s in det.stride]
    captured = [None] * 3
    hooks = []
    for i in range(3):
        def make(idx):
            def hook(_module, inputs):
                captured[idx] = inputs[0].detach()
            return hook
        hooks.append(det.cv3[i][2].register_forward_pre_hook(make(i)))
    try:
        for row in rows:
            captured[:] = [None] * 3
            res = model.predict(source=row["image_path"], imgsz=640, conf=0.25, iou=0.7,
                                max_det=300, device="cpu", verbose=False, save=False)[0]
            h0, w0 = res.orig_shape
            gts = load_gt(row)
            for i in range(3):
                if captured[i] is not None:
                    handler(i, captured[i], strides[i], (h0, w0), gts)
    finally:
        for h in hooks:
            h.remove()
    return strides


def anchor_cell(feature_map, stride, orig_hw, gt):
    """Grid cell of a ground-truth box centre in a letterboxed feature map."""
    _, _, H, W = feature_map.shape
    h0, w0 = orig_hw
    ih, iw = H * stride, W * stride
    r = min(ih / h0, iw / w0)
    dw, dh = (iw - w0 * r) / 2, (ih - h0 * r) / 2
    x1, y1, x2, y2 = gt["xyxy"]
    ax = int(((x1 + x2) / 2 * r + dw) // stride)
    ay = int(((y1 + y2) / 2 * r + dh) // stride)
    return (ax, ay) if 0 <= ax < W and 0 <= ay < H else None


def anchor_dataset(model, rows, per_class=25, bg_per_img=60, seed=7):
    """Class-anchor features per level and class, and background features per level."""
    rng = np.random.default_rng(seed)
    pos = {c: [[] for _ in range(3)] for c in CLASSES}
    bg = [[] for _ in range(3)]

    def handler(level, fmap, stride, orig_hw, gts):
        _, channels, H, W = fmap.shape
        feat = fmap[0].permute(1, 2, 0).numpy()
        occupied = np.zeros((H, W), bool)
        for g in gts:
            cell = anchor_cell(fmap, stride, orig_hw, g)
            if cell is None:
                continue
            ax, ay = cell
            occupied[ay, ax] = True
            if len(pos[g["target"]][level]) < per_class * 4:
                pos[g["target"]][level].append(feat[ay, ax])
        idx = np.flatnonzero(~occupied.reshape(-1))
        if len(idx) > bg_per_img:
            idx = rng.choice(idx, bg_per_img, replace=False)
        bg[level].append(feat.reshape(-1, channels)[idx])

    strides = collect_anchor_features(model, rows, handler)
    return pos, bg, strides


def build_deltas(pos, bg, model, alpha):
    """d_c = alpha * ||w_c|| * unit(mu_c - mu_bg) for each level and class."""
    deltas, mechanism = {}, {}
    state = model.model.state_dict()
    for i in range(3):
        background = np.concatenate(bg[i], 0).astype(np.float32)
        mu_b = background.mean(0)
        for cname in CLASSES:
            anchors = np.asarray(pos[cname][i], dtype=np.float32)
            if len(anchors) == 0:
                continue
            d = anchors.mean(0) - mu_b
            nd = float(np.linalg.norm(d))
            if nd < 1e-9:
                continue
            name = f"{CLASS_CONVS[i]}.weight"
            row = COCO_ID[cname]
            wn = float(torch.linalg.norm(state[name][row].detach().float()))
            delta = (alpha * wn / nd) * d
            key = f"L{i}:{cname}"
            deltas[key] = (name, row, delta.astype(np.float32))
            mechanism[key] = {"level": i, "class": cname, "row": row, "row_norm": wn,
                              "delta_norm": float(np.linalg.norm(delta)),
                              "rel_row_change": float(np.linalg.norm(delta) / max(wn, 1e-9)),
                              "n_positive_anchors": int(len(anchors)),
                              "n_background_anchors": int(len(background))}
    return deltas, mechanism


def apply_deltas(model, deltas):
    state = model.model.state_dict()
    base = {k: v.detach().clone() for k, v in state.items()}
    with torch.no_grad():
        for name, row, d in deltas.values():
            state[name][row].add_(torch.as_tensor(d, dtype=state[name].dtype).view_as(state[name][row]))
    edited = {name: (base[name], state[name].detach().clone()) for name, _, _ in deltas.values()}
    model.predictor = None  # Ultralytics deep-copies the model into its predictor
    return base, edited


def restore(model, base):
    state = model.model.state_dict()
    with torch.no_grad():
        for k, v in base.items():
            state[k].copy_(v)
    model.predictor = None


def preflight_logit_change(model, rows, deltas, seed=13):
    """Change in class logit (feature . delta) at ground-truth anchors vs background."""
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}

    def handler(level, fmap, stride, orig_hw, gts):
        _, channels, H, W = fmap.shape
        feat = fmap[0].permute(1, 2, 0).numpy()
        occupied = np.zeros((H, W), bool)
        for g in gts:
            cell = anchor_cell(fmap, stride, orig_hw, g)
            if cell is None:
                continue
            ax, ay = cell
            occupied[ay, ax] = True
            key = f"L{level}:{g['target']}"
            if key in deltas:
                out.setdefault(key, {"object": [], "background": []})["object"].append(
                    float(feat[ay, ax] @ deltas[key][2]))
        idx = np.flatnonzero(~occupied.reshape(-1))
        if len(idx) > 200:
            idx = rng.choice(idx, 200, replace=False)
        background = feat.reshape(-1, channels)[idx]
        for cname in CLASSES:
            key = f"L{level}:{cname}"
            if key in deltas:
                out.setdefault(key, {"object": [], "background": []})["background"].extend(
                    (background @ deltas[key][2]).tolist())

    collect_anchor_features(model, rows, handler)
    return out


def specificity_report(logit_change):
    report = {}
    for key, vals in logit_change.items():
        obj, bgn = np.array(vals["object"]), np.array(vals["background"])
        if len(obj) < 5 or len(bgn) < 50:
            continue
        rng = np.random.default_rng(5)
        sample = rng.choice(bgn, min(len(bgn), 4000), replace=False)
        wins = (obj[:, None] > sample[None, :]).mean()
        report[key] = {"n_object": int(len(obj)), "n_background": int(len(bgn)),
                       "delta_object_mean": float(obj.mean()), "delta_object_std": float(obj.std()),
                       "delta_background_mean": float(bgn.mean()), "delta_background_std": float(bgn.std()),
                       "specificity_auc": float(wins)}
    return report


def fit_logistic_rows(model, pos, bg):
    """Per-class logistic regression on the anchor features, scaled to the original row norm."""
    from sklearn.linear_model import LogisticRegression

    state = model.model.state_dict()
    fitted = {}
    for i in range(3):
        background = np.concatenate(bg[i], 0).astype(np.float32)
        mu, sg = background.mean(0), background.std(0) + 1e-6
        for cname in CLASSES:
            anchors = np.asarray(pos[cname][i], dtype=np.float32)
            if len(anchors) < 10:
                continue
            X = np.r_[(anchors - mu) / sg, (background - mu) / sg]
            y = np.r_[np.ones(len(anchors)), np.zeros(len(background))]
            lr = LogisticRegression(C=0.1, max_iter=2000).fit(X, y)
            name = f"{CLASS_CONVS[i]}.weight"
            row = COCO_ID[cname]
            w = lr.coef_[0].astype(np.float32) / sg
            b = float(lr.intercept_[0] - (mu / sg) @ w)
            scale = float(np.linalg.norm(state[name][row].detach().float().numpy())) / max(float(np.linalg.norm(w)), 1e-9)
            fitted[f"L{i}:{cname}"] = {"name": name, "row": row, "w": (w * scale).astype(np.float32),
                                       "b": float(b * scale)}
    return fitted


def apply_fitted(model, fitted):
    state = model.model.state_dict()
    base = {k: v.detach().clone() for k, v in state.items()}
    with torch.no_grad():
        for f in fitted.values():
            row = state[f["name"]][f["row"]]
            row.copy_(torch.as_tensor(f["w"], dtype=state[f["name"]].dtype).view_as(row))
        for i in range(3):
            bias = f"{CLASS_CONVS[i]}.bias"
            if bias in state:
                for cname in CLASSES:
                    f = fitted.get(f"L{i}:{cname}")
                    if f:
                        state[bias][f["row"]] = torch.as_tensor(f["b"], dtype=state[bias].dtype)
    names = {f["name"] for f in fitted.values()} | {f"{c}.bias" for c in CLASS_CONVS if f"{c}.bias" in state}
    edited = {k: (base[k], state[k].detach().clone()) for k in names}
    model.predictor = None
    return base, edited


def edit_accounting(edited, n_params, **extra):
    stats = norm_stats({k: v[0] for k, v in edited.items()}, {k: v[1] for k, v in edited.items()})
    stats["changed_frac_of_model"] = round(stats["changed_nonzero"] / n_params, 8)
    return {**stats, **extra}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, default=Path(os.environ.get("YOLOV10S_CKPT", "yolov10s.pt")))
    parser.add_argument("--fit-dir", type=Path, required=True, help="VisDrone train folder (images/, annotations/)")
    parser.add_argument("--eval-dir", type=Path, required=True, help="VisDrone val folder (images/, annotations/)")
    parser.add_argument("--eval-ids", type=Path, default=None, help="text file with the frozen evaluation ids")
    parser.add_argument("--old-edit-ckpt", type=Path, default=None,
                        help="checkpoint written by class_gain_edit.py, scored as the OLD_FAILED arm")
    parser.add_argument("--max-fit", type=int, default=None)
    parser.add_argument("--holdout", type=int, default=6, help="fit images (last N) used for the preflight")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--out", type=Path, default=Path("results/directional_edit.json"))
    args = parser.parse_args()

    from ultralytics import YOLO

    sha_before = sha256_path(args.checkpoint)
    fit = image_pairs(args.fit_dir)[:args.max_fit]
    ids = args.eval_ids.read_text().split() if args.eval_ids else None
    eval_rows = image_pairs(args.eval_dir, ids)
    print("fit", len(fit), "eval", len(eval_rows), flush=True)

    model = YOLO(str(args.checkpoint))
    n_params = sum(p.numel() for p in model.model.parameters())

    pos, bg, _ = anchor_dataset(model, fit)
    deltas_by_alpha, mechanism = {}, {}
    for a in ALPHAS:
        deltas_by_alpha[a], mechanism[str(a)] = build_deltas(pos, bg, model, a)

    holdout = fit[-args.holdout:]
    pre = {str(a): specificity_report(preflight_logit_change(model, holdout, deltas_by_alpha[a]))
           for a in ALPHAS}
    spec_auc = [v["specificity_auc"] for v in pre["0.1"].values()]
    gate = {"median_specificity_auc_at_0.1": float(np.median(spec_auc)) if spec_auc else None,
            "min_specificity_auc_at_0.1": float(np.min(spec_auc)) if spec_auc else None,
            "passed": bool(spec_auc and np.median(spec_auc) >= 0.55),
            "rule": "median specificity AUC across level/class pairs >= 0.55"}
    print("preflight gate:", gate, flush=True)

    results, arm_preds, edit_records = {}, {}, {}
    base_preds = predict_rows(model, eval_rows)
    results["ORIG"] = mets(base_preds)
    arm_preds["ORIG"] = base_preds
    reload_ok = reload_err = None
    deterministic = None

    if gate["passed"]:
        for a in ALPHAS:
            base, edited = apply_deltas(model, deltas_by_alpha[a])
            edit_records[str(a)] = edit_accounting(
                edited, n_params, changed_rows=len(deltas_by_alpha[a]),
                weight_support=sorted({(v[0], int(v[1])) for v in deltas_by_alpha[a].values()}))
            preds = predict_rows(model, eval_rows)
            results[f"NEW_a{a}"] = mets(preds)
            arm_preds[f"NEW_a{a}"] = preds
            restore(model, base)

        rng = np.random.default_rng(777)
        placebo = {}
        for key, (name, row, d) in deltas_by_alpha[0.1].items():
            r = rng.standard_normal(d.shape).astype(np.float32)
            r -= (r @ d) / max(float(d @ d), 1e-12) * d
            r *= np.linalg.norm(d) / max(np.linalg.norm(r), 1e-12)
            placebo[key] = (name, row, r)
        base, edited = apply_deltas(model, placebo)
        edit_records["placebo_a0.1"] = edit_accounting(edited, n_params, changed_rows=len(placebo))
        preds = predict_rows(model, eval_rows)
        results["PLACEBO_a0.1"], arm_preds["PLACEBO_a0.1"] = mets(preds), preds
        restore(model, base)

        base, edited = apply_fitted(model, fit_logistic_rows(model, pos, bg))
        edit_records["FITTED"] = edit_accounting(
            edited, n_params, fit="per-class L2 logistic regression on the same anchor features (C=0.1)")
        preds = predict_rows(model, eval_rows)
        results["FITTED"], arm_preds["FITTED"] = mets(preds), preds
        restore(model, base)

        # one saved copy of the primary dose, reloaded and compared with the intended weights
        base, _ = apply_deltas(model, deltas_by_alpha[0.1])
        args.work_dir.mkdir(parents=True, exist_ok=True)
        saved = args.work_dir / "yolov10s_directional_a0.1.pt"
        torch.save({**model.ckpt, "model": model.model, "ema": None}, saved)
        restore(model, base)
        reloaded = YOLO(str(saved))
        reload_err = max(float((reloaded.model.state_dict()[n][row]
                                - (base[n][row] + torch.as_tensor(d).view_as(base[n][row]))).abs().max())
                         for n, row, d in deltas_by_alpha[0.1].values())
        reload_ok = reload_err < 1e-6

    if args.old_edit_ckpt is not None:
        old_preds = predict_rows(YOLO(str(args.old_edit_ckpt)), eval_rows)
        results["OLD_FAILED_pilot2"], arm_preds["OLD_FAILED_pilot2"] = mets(old_preds), old_preds

    if "NEW_a0.1" in arm_preds:
        target_alerts = alert_count(arm_preds["NEW_a0.1"])
        all_scores = sorted((p["score"] for r in base_preds for p in r["predictions"]
                             if p["source_class_name"] in CLASSES), reverse=True)
        thr = all_scores[min(target_alerts, len(all_scores) - 1)] if all_scores else CONF
        results["CALIB_ONLY"] = mets(base_preds, conf=thr)
        results["CALIB_ONLY"]["matched_threshold"] = float(thr)

    # equal alert budget: each arm's threshold gives it as many alerts as ORIG at conf 0.25
    k = alert_count(base_preds)
    matched_alert = {}
    for arm, preds in arm_preds.items():
        scores = sorted((p["score"] for r in preds for p in r["predictions"]
                         if p["source_class_name"] in CLASSES), reverse=True)
        thr = float(scores[k]) if len(scores) > k else -1.0
        matched_alert[arm] = {**mets(preds, conf=thr), "matched_threshold": thr}
    results["_matched_alert_budget_K"] = k
    results["_matched_alert_metrics"] = matched_alert
    placebo_equals_new = None
    if "NEW_a0.1" in matched_alert and "PLACEBO_a0.1" in matched_alert \
            and matched_alert["NEW_a0.1"]["precision"] is not None \
            and matched_alert["PLACEBO_a0.1"]["precision"] is not None:
        new, plc = matched_alert["NEW_a0.1"], matched_alert["PLACEBO_a0.1"]
        placebo_equals_new = bool(abs(new["precision"] - plc["precision"]) < 0.02
                                  and abs(new["tp"] - plc["tp"]) <= 5)

    counts = {arm: per_image_counts(p) for arm, p in arm_preds.items()}
    vs_orig = {arm: regression_vs_reference(counts[arm], counts["ORIG"]) for arm in counts if arm != "ORIG"}

    if "NEW_a0.1" in arm_preds:
        base, _ = apply_deltas(model, deltas_by_alpha[0.1])
        again = predict_rows(model, eval_rows)
        restore(model, base)
        first = arm_preds["NEW_a0.1"]
        deterministic = len(first) == len(again) and all(
            len(r1["predictions"]) == len(r2["predictions"]) and all(
                abs(a["score"] - b["score"]) <= 1e-9 and a["xyxy"] == b["xyxy"]
                for a, b in zip(r1["predictions"], r2["predictions"]))
            for r1, r2 in zip(first, again))

    record = {
        "model": "yolov10s",
        "source_checkpoint_unchanged": sha256_path(args.checkpoint) == sha_before,
        "n_model_params": int(n_params),
        "dataset": {"gt_objects_in_eval": {c: sum(1 for r in eval_rows for g in load_gt(r) if g["target"] == c)
                                           for c in CLASSES},
                    "fit_images": len(fit), "eval_images": len(eval_rows),
                    "fit_image_ids_sha256": hashlib.sha256("|".join(r["id"] for r in fit).encode()).hexdigest()},
        "actuator_preflight": {"per_dose": pre, "gate": gate, "heldout_fit_images": len(holdout)},
        "dose_alpha": ALPHAS,
        "mechanism_accounting": mechanism,
        "edit_accounting": edit_records,
        "results": results,
        "per_image_vs_orig": vs_orig,
        "primary_arm_rerun_deterministic": deterministic,
        "actuator_specific": bool(gate["passed"]),
        "placebo_equals_new_at_equal_alerts": placebo_equals_new,
        "saved_copy_reload": {"ok": reload_ok, "max_abs_error": reload_err},
    }
    require_unchanged(args.checkpoint, sha_before)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    keys = ("precision", "recall", "tp", "fp", "fn", "count_mae", "matched_threshold")
    print(json.dumps({arm: {kk: vv for kk, vv in v.items() if kk in keys}
                      for arm, v in results.items() if isinstance(v, dict) and arm != "_matched_alert_metrics"},
                     indent=1))
    print("equal alerts K =", k)
    for arm, v in matched_alert.items():
        print(f"  {arm:20s} tp={v['tp']} fp={v['fp']} thr={v['matched_threshold']:.3f}")


if __name__ == "__main__":
    main()
