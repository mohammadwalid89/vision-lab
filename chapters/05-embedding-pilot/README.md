# 05. Multimodal embedding pilot

**Question.** Can a lightweight multimodal embedding replace or add to anomaly scores on hard kits?

**Setup.** 32 evaluation images (12 good, 20 defective) from the two MVTec LOCO categories used in chapter 04. They were already exposed in that chapter's development phase, so the sample is not independent. EmbeddingGemma at 768 and 256 (projected) dimensions, with 16 normal images per category in the reference bank and a score threshold chosen from calibration defects. The comparison scores are the Phase 3 ResNet-18 and WRN-50 PatchCore and component scores on the same 32 images, fitted with different (larger) budgets.

**Result.** From the Phase 4 pilot report as quoted in the Lab research atlas of 2026-11-11 (report and result JSON not published, not in this repo):

| Score | AUC over 32 images (0.5 = chance) |
| --- | ---: |
| EmbeddingGemma 768-d | .483 |
| EmbeddingGemma 256-d | .488 |
| Component score | .588 |
| ResNet-18 PatchCore | .838 |
| WRN-50 PatchCore | .846 |

1 of 20 defects escaped with EmbeddingGemma at either size. Preprocessing and inference over 112 encoded images took a median of 2.25 s (2,252 ms; 95th percentile 3,089 ms) on the CPU, excluding first load and nearest-neighbour scoring.

**What it means.** On this tiny, already-exposed sample the embedding scored near chance and below both PatchCore scores. It does not support a verdict on the model family: the sample is 32 images, the bank is 16 normals per category, and the fitting budgets are unequal. The kits rework plans to re-run the idea as an equal-budget baseline.

**How to run.** Not available. See Status.

**Status.** Code not in this repository. The pilot scripts and `pilot_results.json` were in the same review folder, which was moved to the archive after the run, and no embedding-model code or results remain in the sources used for this repo (a later CPU experiment that mentions EmbeddingGemma in its README only ran SigLIP and says so in its output). The numbers above come from the atlas and were not recomputed; nothing was run for this chapter. If the archive is restored the chapter should hold the encoder wrapper, the bank and threshold script, and `pilot_results.json` and `matched_comparison.json`.
