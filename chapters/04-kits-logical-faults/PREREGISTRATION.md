# Vision System One · Phase 3 — logical kit generalization
Written 2026-10-28 UTC, BEFORE loading any new category evaluation images/scores.

## Objective
Improve good-kit automatic release at an explicitly bounded defect escape risk on genuinely NEW, previously unused product categories, without overfitting the finished Phase 2 test images. Do not modify the deployed v001 or Phase 2 preview while selecting models.

## New categories and provenance
Two MVTec LOCO AD categories NOT previously available on the VPS or used in Phase 2: `juice_bottle` and `splicing_connectors`. Download from a public Hugging Face mirror `zhaolutuan/0423` of the original MVTec LOCO AD, then verify file counts, image paths, retained license text, and SHA256 manifest. Canonical dataset citation: Bergmann et al., IJCV 2022. **License: CC BY-NC-SA 4.0; research/noncommercial use only.** Do not mirror or publish raw images outside the research environment, and do not treat this as real production distribution.

## Data separation (precommitted before download)
- Train normal only: fit feature extractors, patch memory banks, object models and normal baselines.
- Validation normal only: choose/check calibration source and acceptable normal variation, not as an independent final test.
- Test images: construct a hash-seeded, category×stratum stratified selection/final assignment based on relative path **before any candidate model scores are inspected**. Use seed string `VISION-P3-KITS-20261028`; within each stratum sort by SHA256(seed||relative_path), allocate first 40% to development/selection, remainder 60% to locked final. Save the SHA256 of this manifest. Test ground-truth labels only for exposed development until candidate/policy frozen. Hold final evaluation to a single read after freezing.
- No Phase 2 evaluation images, Phase 2 policies, or human feedback claims may be re-labeled as fresh.

## Candidate models and comparison
Compare at minimum: existing component verifier (color/registration/regions), WRN-50 PatchCore, a simpler ResNet-18 PatchCore control, and a feasible structured object/relational alternative if the component-based approach remains weak. They must share identical development/final populations and use same category gate/refusal policy. Optimize for good-part releases and person workload under a zero-dev-escape constraint, not AUC alone. Consider per-category policy, normality representations, and false alarms from valid rearrangements. Don't select models based on final data.

## Reporting
Primary: good-kit automatic release (%) / 1000 good kits; escapes separately for logical and structural defects; one-sided 95% CP escape upper bound per family and category; false holds; manual reviews / 1000 inspections at 2% assumed defects; CPU p50/p95; modeled total decision cost. Record distribution shift, relevant negative findings, and limitations. Acceptance safety claim requires zero escapes on the dev-policy selection set and ≤2.5% upper bound on final family escape rate; no factory readiness implication. If infeasible, preserve failure as a research result.

## Execution safeguards
Disk budget 3 GiB max for new downloaded dataset; stop if root free <8 GiB or before touching unrelated files. Preserve original MVTec and all active experiments. Download one mirror only; avoid duplicating Hugging Face caches. Maintain a separate Phase 3 workspace and separate preview until end-to-end validation.

## Data-source revision correction, 2026-10-28 UTC before any scores
The initial mirror revision `cd63ce7` was discovered to contain an incomplete juice_bottle directory (test only). Import was stopped without analyzing any labels or scores. The new fully populated pinned revision `464875d33af84257cac350ebcf2c4210343f85a3` includes train, validation, test and ground_truth for both previously unused categories. The precommitted path-hash split and evaluation protocol are unchanged. Previously downloaded bytes are retained for the download tool to verify/resume.
