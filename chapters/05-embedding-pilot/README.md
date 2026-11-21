# 05. Multimodal embedding pilot

**Question.** Can a lightweight multimodal embedding replace or add to anomaly scores on hard kits?

**Setup.** 32 evaluation images (12 good, 20 defective) from the two MVTec LOCO categories used in chapter 04. They were already exposed in that chapter's development phase, so the sample is not independent. EmbeddingGemma at 768 and 256 (projected) dimensions, with 16 normal images per category in the reference bank and a score threshold chosen from calibration defects. The comparison scores are the Phase 3 ResNet-18 and WRN-50 PatchCore and component scores on the same 32 images, fitted with different (larger) budgets.

**Result.** From `results/pilot_results.json` (per-image scores and AUCs, written by `code/pilot.py`) and `results/matched_comparison.json` (escapes under one threshold rule, written by `code/compare.py`):

| Score | AUC over 32 images (0.5 = chance) |
| --- | ---: |
| EmbeddingGemma 768-d | .483 |
| EmbeddingGemma 256-d | .488 |
| Component score | .588 |
| ResNet-18 PatchCore | .838 |
| WRN-50 PatchCore | .846 |

1 of 20 defects escaped with EmbeddingGemma at either size. Preprocessing and inference over 112 encoded images took a median of 2.25 s (2,252 ms; 95th percentile 3,089 ms) on the CPU, excluding first load and nearest-neighbour scoring.

**What it means.** On this tiny, already-exposed sample the embedding scored near chance and below both PatchCore scores. It does not support a verdict on the model family: the sample is 32 images, the bank is 16 normals per category, and the fitting budgets are unequal. The kits rework plans to re-run the idea as an equal-budget baseline.

**How to run.** Run from `code/` in this order:

```
python download_vision_subset.py   # language and vision tensors of google/embeddinggemma-2 at a pinned revision, into code/model/
python pilot.py                    # writes pilot_plan.json once, then pilot_results.json (refuses to overwrite)
python compare.py                  # matched_comparison.json from pilot_results.json (PILOT_DIR to point elsewhere)
```

`pilot.py` samples only Phase 3 development images and reads their saved Phase 3 scores from chapter 04's run folder (`P3_OUT`, default `../04-kits-logical-faults/code/out`), so chapter 04's run must exist first. It needs transformers 5.20.2 (the `EmbeddingGemma2Model` class) and 877,599,816 bytes of disk for the model subset.

**Status.** Done (an inconclusive pilot). Code restored from the run archive on 2026-11-21. Cleaned from the run version: the Phase 3 folder and the pilot folder come from environment variables with defaults relative to this repo, one unused import removed, code reformatted with ruff; no scoring or threshold logic was changed. Left out: a later exploratory continuation (`phase4b`) whose results and report are not in the restored archive and which no Lab card cites.

Verified here: py_compile and ruff pass. `compare.py` run on `results/pilot_results.json` rebuilt `results/matched_comparison.json` byte for byte. Every number on the Lab card was checked against the two JSON files and matches; the 16-image bank size comes from the code. `pilot.py` was not re-run (model download and images needed).
