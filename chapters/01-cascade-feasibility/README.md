# 01. Cascade feasibility

**Question.** Can a low-cost cascade cover surface defects, kits, damage, identity and text, and abstain when evidence is weak?

**Setup.** Public cohorts only: MVTec AD (15 categories), MVTec LOCO AD (five categories, used as the "kit / pack completeness" case), and Total-Text (40 scenes). Each surface and kit category uses a fixed cohort of up to 40 normal and 10 anomaly training images, 20 separate normal calibration images, and 10 normal plus 15 anomaly test images (375 test images per seed for MVTec AD, 125 for LOCO). The anomaly training images come from the datasets' official test folders, so these are custom splits and not the official benchmarks. The scorer is a frozen ImageNet ResNet-18 embedding, a sigmoid-calibrated logistic head trained on 80% of each category's training images (seeds 11, 23, 47), and a nearest-good-image distance with a threshold at the 98th percentile of the calibration normals. A case is called EXCEPTION when the head probability is at least 0.8, released as OK only when the probability is below 0.2 and the nearest-good distance is under the threshold, and otherwise goes to a person. A wrong automatic accept is a labelled exception released as OK. Repeated seeds reuse the same test images, so counts across seeds are not additive. Scene text compares PaddleOCR (PP-OCRv5 mobile) with Tesseract on word recall; OCR reads are never released automatically.

**Result.** From `results/mvtec_ad_surface.json`, `results/mvtec_loco_surface.json` and `results/total_text.json` (outputs of the scripts here, run once on 2026-10-22):

| Cohort | Test images per seed | Automatic coverage | Wrong automatic accepts per seed |
| --- | ---: | ---: | ---: |
| MVTec AD surface, 15 categories | 375 | 117 to 128 (31.2% to 34.1%) | 30, 35, 36 |
| MVTec LOCO kits, 5 categories | 125 | 41 to 48 (32.8% to 38.4%) | 19, 23, 23 |
| Total-Text, 40 scenes | 40 | 0 of 40 (all queued) | 0 |

On Total-Text, PaddleOCR matched 82 of 246 reference words and Tesseract 35 of 246. The paired scene bootstrap 95% interval for the recall gap is 0.139 to 0.248.

Coverage, wrong accepts and OCR counts above are in the result files. Figures for the other cohorts in the original run (carton damage, grocery identity, barcodes, VisA candle: 1 to 2 wrong accepts of 36, 1 of 63 resolved, 8 of 10 resolved) and the 47 cross-split duplicate groups in the first synthetic pack data are from the run report (not published), because their code is not part of this chapter.

**What it means.** A better perception score and a useful review queue did not make release safe: both image cohorts failed the wrong-accept gate in every seed, so the cascade was rejected for automatic release on these cohorts. Release needs a gate calibrated per category against an explicit risk, not a better score alone. Limits: custom splits, a single image pool, no physical-unit identifiers in the sources, and different denominators per use case.

**How to run.** Data and model weights are not included. With the ImageNet ResNet-18 weights in the torch cache (`TORCH_HOME`):

```
export PYTHONPATH=src
python chapters/01-cascade-feasibility/build_manifest.py mvtec_ad --data-root "$VISION_DATA_ROOT/mvtec_ad"
python chapters/01-cascade-feasibility/surface_eval.py --data-root "$VISION_DATA_ROOT/mvtec_ad" --out mvtec_ad_surface.json
python chapters/01-cascade-feasibility/build_manifest.py mvtec_loco --data-root "$VISION_DATA_ROOT/mvtec_loco"
python chapters/01-cascade-feasibility/surface_eval.py --data-root "$VISION_DATA_ROOT/mvtec_loco" --out mvtec_loco_surface.json
python chapters/01-cascade-feasibility/text_eval.py --data-root "$VISION_DATA_ROOT/total_text" --out total_text.json --download
```

`text_eval.py` needs PaddleOCR and the `tesseract` binary. Both scripts fetch the dataset commits recorded in `results/` (Total-Text, MVTec AD and LOCO mirrors); the VisA mirror has no recorded commit and is read at its current head. MVTec AD and MVTec LOCO are CC BY-NC-SA 4.0 and must be fetched by the user. A toy run that needs no data: `python chapters/01-cascade-feasibility/synthetic_smoke.py`.

**Status.** Done (negative for release). Code cleaned from the run version: the per-case trace store, run ledger and dashboard bookkeeping were dropped, so the summary JSON here has the same metrics but fewer per-case fields; the three result files are the original run outputs. Verified here: `py_compile` and `ruff` pass; `synthetic_smoke.py` ran end to end on a generated 2-category toy cohort (50 test images, random textures) using the cached ResNet-18 weights. Not verified here: the data download, `text_eval.py` (needs PaddleOCR and the OCR models), and a numeric re-run of the cleaned scripts against the published numbers (the real images are not on hand in this repo).
