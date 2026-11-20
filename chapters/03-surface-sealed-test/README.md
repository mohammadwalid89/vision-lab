# 03. Surface parts, sealed test

**Question.** Does a frozen WRN-50 PatchCore score with a per-category rule release more good surface parts than the original pipeline, without releasing defects?

**Setup.** Memory banks fit on MVTec AD training normals; five-fold out-of-fold calibration. 300 seeded development half-splits chose one policy. The policy and the scores were hash-frozen before a fresh set of 100 good and 200 defective images was opened once. Controls: the original pipeline and a threshold-only version, scored on the identical images.

**Result.** From the Phase 2 evaluation report as quoted in the Lab research atlas of 2026-11-11 (the report and its result JSON are not published and not in this repo):

| Arm (same 100 good, 200 defective images) | Good released | Defects released | Modeled person reviews per 1,000 parts |
| --- | ---: | ---: | ---: |
| Original pipeline | 4 of 100 | 0 of 200 | 956.8 |
| Threshold-only | 4 of 100 | 0 of 200 | 939.9 |
| WRN-50 stack | 84 of 100 | 0 of 200 | 162.9 |

80 good parts gained release and none lost it (exact paired test p about 1.7e-24). By category, an in-sample subgroup view of the same 100 good images (good parts released): bottle 7 of 7, cable 22 of 25, hazelnut 13 of 15, tile 15 of 15, transistor 27 of 27, pill 0 of 11. Zero defects released out of 200 bounds the escape rate at 1.49% (one-sided 95% Clopper-Pearson), not at zero. Scoring time, median and 95th percentile, rose from 145 and 278 ms to 407 and 2,531 ms.

**What it means.** On this public benchmark split a calibrated per-category gate released far more good parts at no observed escapes. It does not cover every category (pill released none), it is not a factory lot test, and the review costs are modeled from assumed prices and a 2% defect rate, not measured.

**How to run.** Not available. See Status.

**Status.** Code not in this repository. The scripts that fit the WRN-50 banks, froze the policy and scored the sealed images lived in a review folder that was moved to the archive after the run and has not been restored. The numbers above are therefore taken from the atlas, not recomputed, and nothing here was run. Chapter 02 holds the audit code for the earlier two-layer evaluation; it is a different experiment (PatchCore ResNet-18 plus a second layer on MVTec LOCO) and is not a substitute. If the archive is restored, the chapter should hold the frozen-policy script, the scoring script and `p2_final_results.json`.
