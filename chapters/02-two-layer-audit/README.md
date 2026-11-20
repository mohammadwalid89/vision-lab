# 02. Independent audit: two layers

**Question.** Can per-category calibration and a selective second layer buy safe autonomy more cheaply than always running a heavy model?

**Setup.** The audit re-uses cached scores from the earlier v3.1 pipeline: 3,381 rows (6 MVTec AD and 3 MVTec LOCO categories) plus 855 images from nine more MVTec AD categories scored here with the same PatchCore + CLIP recipe, so 4,236 images over 18 categories. Labels are the datasets' folder labels, used as a simulated oracle, not operator corrections. The first layer is a PatchCore score; the second layer is a cached semantic kit vector (cheap) or the full AnomalyCLIP-based stack (expensive). Per category, scores are robust z-scores (median/MAD) from good calibration images only, turned into upper-tail conformal p-values. A three-way rule releases above `alpha_release`, holds below `alpha_hold`, and sends the rest to a person; a person's case is never counted as correct. In the release-only gate the second layer runs only when the first layer would release. The MVTec LOCO evaluation set is 592 images (259 defects, 333 good). All of this is exposed development data, not a fresh test.

**Result.** From `results/` (outputs of the scripts here):

| Score on LOCO kit defects | AUC | Bootstrap 95% interval | Incremental CPU, median |
| --- | ---: | --- | ---: |
| First layer (PatchCore) | .673 | .580 to .758 | 0.461 s (first layer itself) |
| Semantic second layer | .732 | .706 to .795 | 0.751 s |
| Full second layer | .733 | .682 to .818 | 7.215 s |
| First OR semantic | .746 | .688 to .819 | |

Who caught the 259 defects (flag at conformal p < 0.05; `l2_complementarity.json`):

| Defect type | Defects | First layer caught | Semantic second layer caught | First missed, second caught | Both missed |
| --- | ---: | ---: | ---: | ---: | ---: |
| Structural | 113 | 36 | 49 | 34 | 43 |
| Logical | 146 | 31 | 53 | 29 | 86 |
| All | 259 | 67 | 102 | 63 | 129 |

With the semantic second layer in the release-only gate and zero wrong releases in sample (0 of 259), 57.1% of images went to a person, 66 of 333 good images were falsely held, and the one-sided 95% Clopper-Pearson upper bound on the wrong-release rate is 1.41%. A more aggressive rule (hold if either layer flags, release if both agree) cut the person share to 10.8% and raised false holds to 224 of 333 (`l2_operating_points.csv`).

On nine MVTec AD categories the earlier scorer had not seen (225 test images, 135 defects, 90 good): a threshold taken from the six original categories released 88 of 135 defects wrongly; per-category calibration on each new category's normals cut that to 23 of 135, with good images falsely held rising from 1 to 17 of 90 (`heldout_adaptation_raw.csv`). On 180 images absent from the earlier surface run, wrong releases went from 61 of 90 to 22 of 90 and false holds from 0 to 7 of 90 (`prior_unseen_eval.csv`).

**What it means.** A calibrated per-category gate did most of the work, and a cheap second layer adds defects the first misses at about a tenth of the cost of the full stack for almost the same AUC. The limits are large: this is an exposed research set, the nine "new" categories overlap an earlier surface experiment (they were unseen by the v3.1 scorer, not blind), zero wrong releases in sample is bounded at 1.41%, not zero, and the cost and economics scenarios use assumed penalties. A summary table in the original run report printed 71 and 106 catches in its "All anomalies" row; the scored JSON gives 67 and 102 (the sums of the two type rows), and the 63 and 129 agree. The table above uses the JSON.

**How to run.** The cached v3.1 score tables are not in this repo, so only the analysis scripts can run, against your own copy of those tables:

```
export VISION_V31_ROOT=/path/to/v31_tree      # results/v31/... score tables, data/public/...
export VISION_OUT=out                         # metrics land in $VISION_OUT/metrics
export VISION_MODEL_CACHE=/path/to/model_cache # clip/RN50.pt, torch/hub/checkpoints/resnet18-f37072fd.pth
python chapters/02-two-layer-audit/score_heldout_ad.py       # nine extra AD categories; needs anomalib, clip, MVTec AD images
python chapters/02-two-layer-audit/calibration_selective.py
python chapters/02-two-layer-audit/l2_complementarity.py
python chapters/02-two-layer-audit/heldout_categories.py
python chapters/02-two-layer-audit/prior_unseen_eval.py
```

To check the scripts without any data: `python chapters/02-two-layer-audit/make_toy_inputs.py --root toy --out toy_out`, then run the four analysis scripts with `VISION_V31_ROOT=toy VISION_OUT=toy_out`. MVTec AD and LOCO are CC BY-NC-SA 4.0 and are not redistributed.

**Status.** Done (exploratory). Code is the independent-audit scripts, cleaned from the run version: paths now come from environment variables, unused helpers and the plotting, forgetting, label-noise and learning-curve scripts are left out because no card number uses them, and the per-script intermediate outputs were not kept. The numbers above are in the original run's metric files, copied into `results/`. Verified here: `py_compile` and `ruff` pass; the four analysis scripts ran to completion on a toy input made by `make_toy_inputs.py` (random numbers, so their output was not checked against anything). Not verified here: a re-run on the real score tables (not available in this repo), and `score_heldout_ad.py` (needs anomalib, CLIP weights and the images).
