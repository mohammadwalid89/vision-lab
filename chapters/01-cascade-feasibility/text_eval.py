"""Scene-text arm: PaddleOCR against Tesseract on 40 Total-Text test scenes.

Word recall is the share of reference words (lower-cased alphanumeric tokens of
length >= 2) found in the OCR output. Text reads never auto-release: every scene is
queued. The recall gap gets a paired bootstrap that resamples scenes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests

from vision_lab.cascade.hashing import phash
from vision_lab.cascade.metrics import summarize, wilson

REPO = "Voxel51/Total-Text-Dataset"
# the dataset commit the published run used (results/total_text.json); not the current HEAD
REVISION = "0ce21350d2ed3f47f7269c7a19d0eeb702952ace"
SOURCE = "https://github.com/cs-chan/Total-Text-Dataset"
LICENSE = "BSD-3-Clause"
BOOTSTRAP_SEED = 20261022
BOOTSTRAP_ITERATIONS = 2000


def words(text: str) -> Counter:
    return Counter(t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 2)


def build_manifest(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    revision = REVISION
    base = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/"
    response = requests.get(base + "samples.json", timeout=30)
    response.raise_for_status()
    samples = response.json()["samples"]
    selected = sorted((s for s in samples if "Test" in s.get("tags", [])),
                      key=lambda item: item["filepath"])[:40]
    if len(selected) != 40:
        raise ValueError("source did not provide 40 test scenes")
    rows = []
    for sample in selected:
        name = sample["filepath"]
        image = root / name
        if not image.exists():
            image.parent.mkdir(parents=True, exist_ok=True)
            result = requests.get(base + quote(name), timeout=90)
            result.raise_for_status()
            image.write_bytes(result.content)
        labels = [p.get("label", "") for p in sample.get("ground_truth_polylines", {}).get("polylines", [])]
        expected = [label for label in labels if label not in ("#", "###") and words(label)]
        rows.append({"id": "total-text-" + Path(name).stem, "unit": name, "split": "test", "path": name,
                     "expected_words": expected,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(), "phash": phash(image)})
    if len({r["sha256"] for r in rows}) != len(rows) or len({r["phash"] for r in rows}) != len(rows):
        raise ValueError("duplicate Total-Text test scene")
    manifest = {
        "dataset": "Total-Text", "source": SOURCE, "mirror": "https://huggingface.co/datasets/" + REPO,
        "revision": revision, "license": LICENSE,
        "attribution": "Ch'ng, Chan, and Liu; Total-Text Dataset",
        "split_rule": ("40 lexicographically first source-tagged Test scenes; no training or calibration; "
                       "unique source paths and perceptual hashes"),
        "rows": rows,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def read_text(engine: str, image: Path, ocr) -> str:
    if engine == "paddleocr":
        result = ocr.predict(str(image))
        return " ".join(result[0].get("rec_texts", [])) if result else ""
    process = subprocess.run(["tesseract", str(image), "stdout", "--psm", "11"],
                             capture_output=True, text=True, timeout=30)
    return process.stdout


def evaluate(manifest: dict, root: Path) -> dict:
    from paddleocr import PaddleOCR

    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    ocr = PaddleOCR(use_doc_orientation_classify=False, use_doc_unwarping=False,
                    use_textline_orientation=False, text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec", enable_mkldnn=False)
    scene_recall: dict = {}
    runs = []
    for engine in ("paddleocr", "tesseract"):
        cases = []
        for source in manifest["rows"]:
            begin = time.perf_counter()
            text = read_text(engine, root / source["path"], ocr)
            latency = round((time.perf_counter() - begin) * 1000, 2)
            expected = Counter()
            for label in source["expected_words"]:
                expected.update(words(label))
            matched = sum((expected & words(text)).values())
            scene_recall.setdefault(engine, {})[source["id"]] = (matched, sum(expected.values()))
            cases.append({"id": source["id"], "truth": "reference_text", "state": "queued", "verdict": None,
                          "resolved_by": "HUMAN", "latency_ms": latency, "neurons": 0, "correct": None})
        matches = sum(p[0] for p in scene_recall[engine].values())
        reference = sum(p[1] for p in scene_recall[engine].values())
        runs.append({"engine": engine, "metrics": summarize(cases),
                     "word_recall": matches / reference if reference else 0,
                     "matched_words": matches, "reference_words": reference,
                     "word_recall_wilson95_descriptive_only": wilson(matches, reference)})
    ids = sorted(scene_recall["paddleocr"])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    differences = []
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = rng.choice(ids, size=len(ids), replace=True)
        totals = []
        for engine in ("paddleocr", "tesseract"):
            matches = sum(scene_recall[engine][key][0] for key in sample)
            reference = sum(scene_recall[engine][key][1] for key in sample)
            totals.append(matches / reference if reference else 0)
        differences.append(totals[0] - totals[1])
    return {"dataset": "Total-Text", "revision": manifest["revision"], "runs": runs,
            "paired_paddle_minus_tesseract_word_recall_ci95": np.quantile(differences, [0.025, 0.975]).tolist(),
            "paired_case_ids": ids,
            "limitation": "40 scenes; words within a scene are correlated; bootstrap resamples scenes"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True,
                        help="directory for the Total-Text images and manifest.json")
    parser.add_argument("--out", type=Path, required=True, help="summary JSON to write")
    parser.add_argument("--download", action="store_true", help="build the manifest and fetch images first")
    args = parser.parse_args()
    if args.download or not (args.data_root / "manifest.json").exists():
        manifest = build_manifest(args.data_root)
    else:
        manifest = json.loads((args.data_root / "manifest.json").read_text())
    summary = evaluate(manifest, args.data_root)
    args.out.write_text(json.dumps(summary, indent=2))
    for run in summary["runs"]:
        print(f"{run['engine']}: {run['matched_words']}/{run['reference_words']} words")


if __name__ == "__main__":
    main()
