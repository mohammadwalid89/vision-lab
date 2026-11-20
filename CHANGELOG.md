# Changelog

The history behind the chapters: failed attempts, re-runs, infrastructure problems, process stops and corrections. The chapter READMEs report results; this file records how they were reached.

## 2026-10-22 — Cascade feasibility (chapter 01)
- The real-image surface, kit and carton arms all failed the wrong-automatic-accept gate and were rejected; scene text never auto-released.
- An audit of the earlier synthetic pack data found 47 groups of byte-identical images crossing its train, validation and test splits. The pack figures were kept as historical only and excluded from the evidence.

## 2026-10-28 — Two-layer audit, sealed surface test, kits (chapters 02–04)
- Chapter 02: the run report's summary table printed 71 and 106 catches in its "All anomalies" row, contradicting its own subtype rows and the scored JSON (67 and 102). The scored JSON is used.
- Chapter 02: the nine "new" MVTec AD categories were unseen by the scorer but not blind: all 225 of their test images overlap an earlier surface experiment, so a smaller 180-image prior-unseen check was added.
- Chapter 03: the sealed surface set was opened once.
- Chapter 04: none of the ten kit candidates met the 1% development escape rule, so the pre-registered all-to-a-person fallback applied. An unselected candidate (component + WRN) showed 17 good releases and 0 escapes on the final set; it was not adopted because it failed the development rule.

## 2026-11-04 — Writing weights (chapters 06–08)
- Chapter 06: a later attempt to reproduce the programmed head returned 0.578 instead of 0.8158 because the original uses a soft category gate; the reimplementation was corrected before anything was interpreted.
- Chapter 07: the first class-gain edit raised recall from 0.177 to 0.453 at 5,757 extra false positives. Its placebo added random noise to the already edited weights, so it is not an independent control (found during the 2026-11-20 cleanup). The 341 vs 338 comparison uses the directional edit and its own same-size placebo.
- Chapter 07: a background-subspace run reported identical metrics for every arm because Ultralytics copies the model into its predictor on first use, so in-memory edits never reached inference; it also had a tensor-aliasing bug in its accounting. Excluded as invalid. The directional edit resets the predictor after every edit and verifies a saved copy by reloading it.
- Chapter 07: the first head-only fine-tune reported zero detections because the trained checkpoint carried class names c0–c79 while ids were COCO ids. Names were restored for reporting and the run re-scored (338 true positives at equal alerts).
- Text-encoder edits (not in this repository): CLIP RN50 edits could not move the text embedding by more than 0.0109 at the largest dose and lowered AUC from 0.9467 to 0.9000; the SigLIP2 token edit showed zero change because the runtime's text encoder is position-diagonal, so that result was declared invalid, not negative.
- Chapter 08: the first R3D-18 clip sampler used a 0.64-second centred window (probe 0.4167); replaced by uniform sampling over the whole clip (0.7037). The kinematic motion rule was checked before compiling weights and abandoned: probe 0.4067, all clips labelled walking.
- Chapter 08: the low-pass dose ladder was revised from α = 0.02, 0.10, 0.25 to 0.001, 0.02, 0.10 before the accuracy run, after the preflight showed 0.10 and 0.25 were destructive.

## 2026-11-06 — Reader loop (chapter 09)
- Preflight on three VisDrone images found 1 true positive and 475 misses against 476 objects; the task changed to image-level presence on the same frozen images before any arm ran. The study was paused by the owner before any comparison.

## 2026-11-08 — Vision-language protocol (chapter 10)
- 1,000 images and prompts frozen before inference. SmolVLM2 at full resolution timed out at 420 s; a 512-pixel crop adapter finished one engineering prompt in 29.93 s. Qwen3-VL timed out at 420 s per image on one core. Florence-2 first failed on a generation-cache error, then produced four tokens in 46.4 s after a workaround. No primary image was scored: 2,000 predictions exceed the 10-hour one-core budget declared in advance.

## 2026-11-12 to 11-14 — Shortcut Rounds 1 and 2, probes (chapters 11–12)
- The study was first prepared on MVTec LOCO structural images only; the design and six gates were reviewed and signed before any arm ran. The first development order (one attempt per family) ran and every family failed; Round 1 was authorised as a single revision with at most three registered iterations per family.
- Round 1 data: MVTec AD was the intended extra source but its download form rejected the request, so the official VisA release (CC BY 4.0) was used, plus the LOCO structural and good images. 135 logical or unknown VisA cases were excluded.
- Round 1 mask and variant construction was interrupted twice and resumed; per-source timings are missing for the interrupted prefix. An interrupted fit is set aside and repeated under a new run id with the same configuration and seeds.
- Round 1: 9 attempts, none passed. Each iteration changed the recipe after reading the previous failure.
- Round 2: the first registration attempt was rejected by the registry schema (an unregistered threshold-reference alias); no attempt had been registered and no data accessed. The first campaign launch stopped because earlier state needed an identical-work audit, and continued after it. The nine attempts then ran in 479 s; none passed.
- Kits audit (chapter 04 record): the ten Phase 3 candidates' development tables were rebuilt from stored per-image scores and matched the frozen record for 10 of 10. The audit found the development "zero escapes" row is circular (the release threshold is the largest defect p-value) and that candidates were selected on a population one third structural.
- Deletion probe: the first full execution crashed in CSV aggregation after all arms had run; no metrics were written or inspected. Only the reporting block was patched and the whole probe re-run; reported numbers come from the re-run.
- Re-anchor probe: three amendments to the pre-registration were recorded, each before the run it affects.

## 2026-11-16 — Infrastructure
- The folder that held the Phase 2 (WRN-50 sealed test), Phase 3 (kit candidates) and Phase 4 (embedding pilot) code was moved to the archive. The vision-language protocol folder was also absent when this repository was built. Chapters 03, 05 and 10 therefore have no code here, and chapter 04 holds only the later audit script.

## 2026-11-17 to 11-19 — Round 3 (chapter 13)
- 11-17: the weight-space library was written with 59 tests passing, plus a mechanics-only smoke on 36 images (not evidence).
- 11-17: design review found the slot-1 random-area placebo would likely remove the planted marker in about half the draws (52% in the smoke), making the placebo gate unpassable by construction. A padding-sparing placebo was implemented.
- 11-17: a second encoder arm (DINOv2 ViT-B/14 with registers) was approved with the same slots, recipes, gates, sample sizes, seeds and calibration partition; the two arms are reported separately and never pooled.
- 11-18: the mask-manifest parser was fixed: the old code would have compared no mask hashes yet reported PASS. Tests grew from 59 to 74.
- 11-18: before the preflight, SigLIP `mean_patch` was changed to the mean of block-output tokens before the final layer norm, matching the Round 1 cache recipe.
- 11-18: the plan with the random-area placebo was replaced by the padding-sparing plan on the owner's sign-off, recorded as given before any result was read (the random-area placebo removed the marker in about 59% of draws in a label-free precompute). The plan file and preflight logs carry file times earlier than the recorded sign-off time; all attempts ran on 11-19, after both.
- 11-19: all 18 attempts (9 per encoder) ended null or negative with no-op integrity holding. No attempt was re-run.

## 2026-11-13 to 11-19 — Kits written checks, agent edits, learning suite (chapters 14–16)
- 11-13: the kits written-check track was prepared and stopped before any development scoring (split sealed, features for train and development rows only, 33 synthetic tests).
- 11-18: pass/fail bars for the kits track ratified and development scoring unlocked; holdout bars deferred.
- 11-19: development scoring for juice bottles ran once (71 images). The runner's single-shot rule (one marker per track, not per category) then refused the splicing-connectors run, so connectors were never scored. A process stop, not a failure of the method.
- 11-19: a first model-in-the-loop attempt on the written score (10 rounds, random-proposal placebo) gave development AUC 0.7105 for the model's best proposal against 0.7471 unedited and 0.7629 for the random loop. Not carried forward.
- 11-19: one agent edited the kit head (18 of 50 scoring calls), scored once on unseen development kits.
- 11-19: learning suite v1 (lessons and no-lessons arms). The two Flowers-102 tasks were saturated (start head at accuracy 1.0) and excluded from the reported 8 tasks; DTD produced no tasks (too few images per class).
- 11-19: learning suite v2 (11 tasks, lessons arm only; the no-lessons arm was not repeated). Tasks whose start head already reached 95% on the feedback split were skipped.

## 2026-11-20 — Repository build
- Code cleaned from the run versions: server paths replaced by arguments and environment variables, registry checks and launch scripts removed, banner comments removed, imports renamed to `vision_lab.*`.
- Checks against the original code: the programmed head (ch06) reproduces its table on the original cached features; the closed-form video head (ch08) reproduces the +2.0-point comparison on saved features; the Mirage preflight scorer (ch09) reproduces 1/17/475 from stored output; ch11–12 cleaned functions match the originals on random inputs; ch13 `plan` with the signed overrides reproduces the run plan byte for byte; ch14 scoring gives identical scores on a synthetic cache; ch15 reproduces the start, placebo and agent heads exactly; ch16 build output matches the originals on synthetic input. Nothing else was re-run on real data.
- The deletion-probe report describes the signed change as acting on the same coordinates as the mask; the code adds it to the retained coordinates. Noted in chapter 12, not changed.
- Lab page corrections found while tracing sources: the agent's feedback AUC rose from .86 to .92 (0.8639 to 0.9156), not .84 to .92; the deletion probe's "5 of 6 gates" holds at a development-fitted threshold (4 of 6 at the frozen threshold) and its 0.6-point gain is AUROC; "the false-alarm gate failed in every arm that removed more than 2%" comes from the deletion probe, not from Rounds 1–2.
