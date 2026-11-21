#!/usr/bin/env python3
"""Small, CPU-only EmbeddingGemma 2 kit diagnostic on previously exposed Phase 3 dev data."""

import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
# Phase 3 manifests and development scores (chapter 04's run folder)
P3 = Path(os.environ.get("P3_OUT", HERE.parents[1] / "04-kits-logical-faults" / "code" / "out"))
SEED = "VISION-P4-EG2-PILOT-20261028"
CATS = ("juice_bottle", "splicing_connectors")
STRATA = ("good", "logical_anomalies", "structural_anomalies")
PLAN = HERE / "pilot_plan.json"
OUT = HERE / "pilot_results.json"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose(rows, n, tag):
    return sorted(rows, key=lambda r: hashlib.sha256((SEED + tag + r["sha256"]).encode()).hexdigest())[:n]


def make_plan():
    train = json.loads((P3 / "p3_train.json").read_text())
    val = json.loads((P3 / "p3_validation.json").read_text())
    dev = json.loads((P3 / "p3_dev.json").read_text())
    p = {
        "seed": SEED,
        "source": "Phase 3 exposed development only; prior Phase 3 final remains unopened by this program",
        "source_sha256": {n: digest(P3 / ("p3_" + n + ".json")) for n in ("train", "validation", "dev")},
        "cohort": [],
        "settings": {
            "train_good_per_category": 16,
            "validation_good_per_category": 8,
            "dev_per_category_stratum": {"good": 12, "logical_anomalies": 12, "structural_anomalies": 8},
            "vision_soft_tokens": 70,
            "dimensions": [768, 256],
            "dtype": "float32",
            "threads": 2,
        },
    }
    for cat in CATS:
        for split, source, n in [("train", train, 16), ("validation", val, 8)]:
            for r in choose([x for x in source if x["category"] == cat], n, cat + split):
                p["cohort"].append({"split": split, **r})
        for stratum, n in p["settings"]["dev_per_category_stratum"].items():
            sample = choose([x for x in dev if x["category"] == cat and x["stratum"] == stratum], n, cat + stratum)
            for i, r in enumerate(sample):
                # Hash order is fixed before seeing any new embedding scores.
                p["cohort"].append({"split": "calibration" if i % 2 == 0 else "evaluation", **r})
    PLAN.write_text(json.dumps(p, indent=2) + "\n")
    return p


def baseline_rows():
    data = {}
    for cat in CATS:
        for model in ("resnet", "wrn", "component_relational"):
            j = json.loads((P3 / f"p3_{model}_{cat}_dev.json").read_text())
            for r in j["development"]:
                data.setdefault((cat, r["sha256"]), {}).update(
                    {model: r["score"] if model != "component_relational" else r["component"]}
                )
    return data


def score_report(rows, key):
    ans = {"by_category": {}, "all": {}}
    for cat in (*CATS, "all"):
        z = [r for r in rows if r["split"] == "evaluation" and (cat == "all" or r["category"] == cat)]
        y = [r["stratum"] != "good" for r in z]
        scores = [r[key] for r in z]
        result = {
            "n_good": y.count(False),
            "n_defect": y.count(True),
            "n_logical": sum(r["stratum"] == "logical_anomalies" for r in z),
            "n_structural": sum(r["stratum"] == "structural_anomalies" for r in z),
            "auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
        }
        if cat == "all":
            ans["all"] = result
        else:
            ans["by_category"][cat] = result
    return ans


def main():
    if OUT.exists():
        raise SystemExit("pilot_results.json exists; refusing to overwrite evidence")
    p = json.loads(PLAN.read_text()) if PLAN.exists() else make_plan()
    assert p["source_sha256"] == {n: digest(P3 / ("p3_" + n + ".json")) for n in ("train", "validation", "dev")}
    import torch
    from transformers import AutoProcessor, EmbeddingGemma2Model

    torch.set_num_threads(2)
    model = EmbeddingGemma2Model.from_pretrained(
        HERE / "model", torch_dtype=torch.float32, local_files_only=True
    ).eval()
    proc = AutoProcessor.from_pretrained(HERE / "model", local_files_only=True)
    proc.image_seq_length = 70
    proc.image_processor.max_soft_tokens = 70
    base = baseline_rows()
    rows = []
    for i, r in enumerate(p["cohort"]):
        if os.statvfs(HERE).f_bavail * os.statvfs(HERE).f_frsize < 8 * 1024**3:
            raise SystemExit("Disk floor violated")
        t = time.perf_counter()
        with Image.open(r["path"]) as im:
            inputs = proc(images=im.convert("RGB"), return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs)
            hidden = output.last_hidden_state.float()
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = ((hidden * mask).sum(dim=1) / mask.sum(dim=1)).squeeze(0).numpy()
        emb = emb / np.linalg.norm(emb)
        elapsed = (time.perf_counter() - t) * 1000
        z = {
            "sha256": r["sha256"],
            "category": r["category"],
            "stratum": r["stratum"],
            "split": r["split"],
            "latency_ms": elapsed,
            "embedding": emb.tolist(),
        }
        z.update(base.get((r["category"], r["sha256"]), {}))
        rows.append(z)
        if (i + 1) % 8 == 0:
            print("embedded", i + 1, "/", len(p["cohort"]), "ms", round(elapsed), flush=True)
    # Normal-only memory bank, same per-category cohort for both dimensions.
    for dim in (768, 256):
        key = f"eg2_{dim}"
        for cat in CATS:
            bank = []
            for r in rows:
                if r["category"] == cat and r["split"] == "train":
                    x = np.asarray(r["embedding"][:dim])
                    bank.append(x / np.linalg.norm(x))
            bank = np.stack(bank)
            for r in rows:
                if r["category"] == cat:
                    x = np.asarray(r["embedding"][:dim])
                    x = x / np.linalg.norm(x)
                    r[key] = float(1 - np.max(bank @ x))
    evaluated = []
    for r in rows:
        if r["split"] in ("calibration", "evaluation"):
            evaluated.append({k: v for k, v in r.items() if k != "embedding"})
    metrics = {
        key: score_report(evaluated, key) for key in ("eg2_768", "eg2_256", "resnet", "wrn", "component_relational")
    }
    # Calibrate zero-release-on-known-defects threshold only on calibration half.
    policies = {}
    for key in ("eg2_768", "eg2_256"):
        by_cat = {}
        for cat in CATS:
            cal_bad = [
                r[key]
                for r in evaluated
                if r["split"] == "calibration" and r["category"] == cat and r["stratum"] != "good"
            ]
            threshold = float(np.nextafter(min(cal_bad), -np.inf))
            ev = [r for r in evaluated if r["split"] == "evaluation" and r["category"] == cat]
            by_cat[cat] = {
                "threshold": threshold,
                "good_releases": sum(r[key] <= threshold for r in ev if r["stratum"] == "good"),
                "good_n": sum(r["stratum"] == "good" for r in ev),
                "defect_escapes": sum(r[key] <= threshold for r in ev if r["stratum"] != "good"),
                "defect_n": sum(r["stratum"] != "good" for r in ev),
            }
        policies[key] = by_cat
    durations = [r["latency_ms"] for r in rows]
    payload = {
        "plan_sha256": digest(PLAN),
        "model_sha256": digest(HERE / "model" / "model.safetensors"),
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "cpu_threads": 2,
            "vision_soft_tokens": 70,
            "model_load_excluded_from_latency": True,
        },
        "rows": evaluated,
        "metrics": metrics,
        "policies": policies,
        "latency_ms": {
            "n": len(durations),
            "p50": float(np.median(durations)),
            "p95": float(np.percentile(durations, 95)),
        },
        "disk_free_bytes_after": os.statvfs(HERE).f_bavail * os.statvfs(HERE).f_frsize,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        "RESULT",
        json.dumps({"metrics": metrics, "policies": policies, "latency_ms": payload["latency_ms"]}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
