# 06. Programming weights: an anomaly head

**Question.** Can closed-form formulas write useful network weights without gradient descent?

**Setup.** 855 cached MVTec AD feature rows from nine categories (CLIP RN50 1,024-d and ResNet18 2,560-d embeddings, plus a category one-hot: 3,593 inputs). Roles: 450 training rows (360 good, 90 defective), 180 good rows that set every arm's threshold, and 225 test rows (90 good, 135 defective). The test rows were already used by an earlier experiment, so they are exposed, not blind. The network has 18 ReLU units (two signed units per category, gated by the category) and 64,711 parameters.
Arms on identical rows: untrained (all zero), a random-direction placebo of the same shape and norms, the programmed head, a head of the same shape trained with Adam on the same 450 labelled rows, and two non-neural comparators that only calibrate the existing PatchCore and CLIP scores per category. Every threshold is the 95th percentile of the same good calibration rows.

**Result.** Test AUC on the 225 exposed rows, from `results/summary.json`:

| Arm | AUC | 95% bootstrap interval | Wrong releases (of 135 defects) | False holds (of 90 good) |
| --- | ---: | --- | ---: | ---: |
| Untrained | 0.5000 | 0.500 to 0.500 | 0 | 90 |
| Random placebo | 0.5164 | 0.447 to 0.594 | 107 | 11 |
| Programmed (closed form) | 0.8158 | 0.761 to 0.866 | 68 | 9 |
| Adam, same labels | 0.8580 | 0.808 to 0.906 | 61 | 2 |
| PatchCore, calibrated | 0.8886 | 0.842 to 0.929 | 53 | 5 |
| CLIP score, calibrated | 0.7715 | 0.710 to 0.834 | 74 | 12 |

The programmed head's worst category (toothbrush) scores 0.4867 AUC against 0.68 for the trained head (`arms.*.worst_category_auc`).
A later revisit with five-fold cross-validation inside the 450 training rows (from the run report, not published; the code for that revisit is not included here) gave pooled AUC 0.8261 for the original rule, 0.8542 for Adam on the same labels and 0.8956 for ordinary logistic regression. Adding two, three or five Fisher directions per category lowered the pooled score (0.7605, 0.7079, 0.6872), so the explanation that sparse defects needed several directions did not hold, and the toothbrush collapse did not reappear in cross-validation (0.797).

**What it means.** Weights can be written by formula and carry real signal: the programmed head is far above the untrained and placebo heads. It is not zero-shot, because the formulas use per-category class means from labelled rows, and it does not beat a head fitted by gradient descent on the same labels or the calibrated PatchCore score. The data are exposed and about ten defects per category, so the ranking between the programmed and trained heads is not a general claim. This chapter does not test the loop of the source paper, where a model reads the weights, edits them and sees feedback.

**How to run.**
The input table is produced by chapter 02 (`score_heldout_ad.py`, which needs MVTec AD; see the dataset notice in the repository README). Then:
```
python programmed_head.py --cohort-dir <folder with heldout_ad_scores.jsonl, clip_embeddings.npz, resnet_embeddings.npz> \
    [--protocol <role-labelled protocol json>] --out results/summary.json
```
One CPU thread, about six seconds. `--protocol` runs an overlap check against reserved rows and is optional. The script imports the `vision_lab` package, so install it or put `src` on the path first (root README, "How to run").

**Status.** Done. Code is cleaned from the run script: server paths became arguments, the saved weight arrays, code hash and artifact folder were dropped, the reserved-row overlap check became optional, and the shared gated-unit construction was factored out of the programmed and placebo builders; the formulas are unchanged. Verified here: `python -m py_compile` and `ruff check` pass; the cleaned script was run on the original cached feature table and reproduces the numbers above (`results/summary.json` is that output); on a synthetic cohort (three categories) the cleaned script and the original run script give identical AUCs, intervals, operating points, per-category tables and weight hashes for all six arms.
