# 14. Kits rework: written part checks

**Question.** Can a kit's specification, written down as part checks, decide what image anomaly scores could not?

**Setup.** MVTec LOCO AD, logical anomalies and good images only (structural anomalies are out of scope). SigLIP2-base patch tokens (14 x 14 grid, 768-d) are cached once. A 32-anchor k-means part dictionary is fitted on normal training images only. Five written checks (presence, count, position, extent, co-occurrence) each have a tolerance equal to the 99.5th percentile of that check's own deviation on normal images, and fixed a-priori weights 1.00 / 1.00 / 0.75 / 1.00 / 1.00. A small logistic part (214 parameters on a 213-d compact vector) is blended with the written score at alpha = 0.5 (primary), both standardised on calibration normals. The exact algebra is in `kits_head.py`; all constants were fixed before any development image was scored.

Baselines and controls on the same split: a linear head on pooled features, a matched-compute MLP on the compact vector, a PatchCore-style patch memory bank, the written score with shuffled anchor references, and the written score with uniform weights. Evaluation images are split per (category, kind) stratum by a hash of the file path: 40% train, 30% dev, the rest sealed holdout. The holdout was not read.

**Result.** Development run on juice bottles only: 71 images (43 logical anomalies, 28 good). AUC from `results/summary.json` (computed from the per-image scores in `results/dev_scores_juice_bottle.json`):

| Head | AUC |
|---|---:|
| Written checks + learned part (alpha 0.5, primary) | 0.8223 |
| Trained linear head | 0.8879 |
| Matched-compute MLP | 0.8804 |
| Written score only (alpha 1.0) | 0.7471 |
| Learned part only (alpha 0.0) | 0.8231 |
| Written score, uniform weights | 0.7479 |
| Patch memory bank | 0.7243 |
| Written score, shuffled references (placebo) | 0.4527 |

Individual checks on their own: presence 0.6395, count 0.5814, position 0.5897, extent 0.5581, co-occurrence 0.7168.

The splicing-connectors category did not run (see Status).

**What it means.** Writing the specification as checks works better than its shuffled placebo (0.82 against 0.45) and the trained linear head is still ahead (0.89). The recorded standardiser for the written score on calibration normals has median 0 and scale 0.001 for every head (`written_std` in the result file), so the written score is zero on most normal images by construction. With 43 defects in one category this is a coarse instrument: the run is exploratory, not a certificate of any escape rate. The pass/fail bars registered for this track (a missed-defect rate, a release rate) were not computed for this README; only AUC is published here.

**How to run.** Data and weights are not included; MVTec LOCO AD is licensed CC BY-NC-SA 4.0 and must be downloaded by you (see the repository README).

```
export LOCO_ROOT=/path/to/mvtec_loco_anomaly_detection
export SIGLIP2_DIR=/path/to/siglip2_base          # local directory with config.json + model.safetensors
python kits_split.py --out work/
python kits_features.py --manifest work/SPLIT_MANIFEST.json --out work/features
python kits_eval.py --features work/features --category juice_bottle --out work/
python summarize.py work/DEV_RESULTS_juice_bottle.json --out work/summary.json
python smoke_synthetic.py        # no data or weights needed
```

Requires `numpy`, `scipy`, `scikit-learn`; `torch` for the MLP baseline and `transformers`, `safetensors`, `Pillow` for feature extraction. The scripts import each other, so run them from this folder.

**Status.** Partial, exploratory: one category (juice bottles) scored on development data; the connector category did not run, and the holdout stays sealed. Code cleaned from the run version: absolute paths replaced by arguments/environment variables; the single-shot gate files (`DEV_UNLOCK`, `DEV_SCORING_STARTED`) replaced by a refusal to overwrite an existing result; the frozen-parameter dump, the environment check and the earlier-approach audit scripts left out; unused methods removed; scoring logic unchanged. Verified here: `py_compile` and `ruff check` clean; `smoke_synthetic.py` run on a synthetic 196 x 32 cache (finite scores, deterministic, defects score higher); the cleaned `kits_eval` functions give identical scores (max absolute difference 0.0) to the run version on that synthetic cache. Not verified here: `kits_split.py` and `kits_features.py` were not run (they need the dataset and encoder weights), and the juice-bottle result was not re-run.
