# 08. Programming weights: video heads

**Question.** Can video heads be written directly, and can a small kernel edit improve them?

**Setup.** 300 KTH clips (walking, jogging, running; 25 subjects, 4 capture scenarios, 100 per action). Model: torchvision R3D-18 pretrained on Kinetics-400, frozen, 16 frames spread over each clip, 512-d pooled features. All results are five-fold subject-grouped out-of-fold, so each clip is scored by a probe or head that never saw its subject; confidence intervals are subject-level bootstraps. The cohort is exposed: every subject was used in design.
Three experiments:

1. `kinematic_rule.py`: a motion rule written from speed, cadence and amplitude thresholds (no labels), checked on silhouette features before any weights are compiled from it.
2. `programmed_video_head.py`: a linear head (512 x 3 + 3 = 1,539 parameters) written in closed form (class means, pooled covariance, fixed ridge 1.0), against a gradient-trained logistic head, a nearest-centroid head and a random head.
3. `temporal_lowpass.py`: rewrite the time axis of all 17 temporal-kernel Conv3d tensors of R3D-18 as a low-pass filter (and a same-size random rotation and a high-pass filter as controls), refit the same probe on each variant's features. The dose ladder (a = 0.001, 0.02, 0.10) was fixed after a preflight that checks the edit moves the features.

**Result.** From `results/summary.json`.

Motion rule: the four kinematic features carry little label information: a five-fold probe on them reaches 0.4067 accuracy (chance 0.3333), the median measured cadence is 0 Hz, and the declared rule labels every clip as walking (0.3333). No weights were compiled from it.

Closed-form head (300 clips):

| Head | Accuracy | Difference from trained head (95% interval) |
| --- | ---: | --- |
| Random | 0.3400 | -32.3 points |
| Nearest centroid | 0.5333 | -13.0 points (-17.3 to -8.7) |
| Trained (logistic) | 0.6633 | |
| Closed-form LDA | 0.6833 | +2.0 points (-2.0 to +6.0) |

LDA is right on 205 clips and the trained head on 199; LDA is right on 20 clips the trained head misses and wrong on 14 the trained head gets right.

Temporal low-pass edit (17 Conv3d tensors, 32,984,621 of 33,371,472 weights touched):

| Arm | Accuracy | Change from original (95% interval) |
| --- | ---: | --- |
| Original | 0.6633 | |
| Low-pass a = 0.001 | 0.6667 | +0.33 points |
| Low-pass a = 0.02 | 0.6933 | +3.00 points (-0.33 to +6.33) |
| Low-pass a = 0.10 | 0.6500 | -1.33 points |
| Random rotation, same size as a = 0.02 | 0.6667 | +0.33 points |
| High-pass a = 0.02 | 0.6400 | -2.33 points |

The pre-registered criteria for the a = 0.02 edit were at least +5 points over the original and at least +3 over the random-rotation control; it reached +3.00 and +2.67, and its interval includes zero.

**What it means.** A head can be written in closed form and match one trained by gradient descent on the same labels here (+2.0 points, interval spanning zero), but it is fitted from examples; the attempt to compile it from a task description alone failed at the feature stage. The low-pass edit moves in the expected direction (the high-pass control is worst and a five times larger dose is destructive) but it missed its bar and its interval includes zero, so it is an unconfirmed signal, not an improvement. All 25 subjects are used, so confirmation needs new subjects or a new dataset. Robustness to pixel noise or camera shift was not tested.
Two text-encoder edits (CLIP RN50 token and projection edits on MVTec bottle; SigLIP2 token edits) were also tried in this study: the CLIP edits could not move the text embedding enough to matter on a saturated cohort, and the SigLIP2 token edit did not reach the encoder output in the runtime used, so that run was excluded as invalid. Their code is not included here.

**How to run.** Requires `torch`, `torchvision`, `opencv-python`, `scikit-learn`, `scipy` and the extracted KTH `.avi` files (see the dataset notice in the repository README). From this folder:
```
python temporal_lowpass.py cache     --clips-dir <kth> --cache-dir cache
python temporal_lowpass.py preflight --clips-dir <kth> --cache-dir cache
python temporal_lowpass.py run       --clips-dir <kth> --cache-dir cache
python programmed_video_head.py features --clips-dir <kth> --cache-dir cache
python programmed_video_head.py heads    --cache-dir cache
python kinematic_rule.py --clips-dir <kth>
```
`--r3d-weights` accepts a local torchvision `r3d_18` Kinetics-400 state dict; without it torchvision downloads the weights. One CPU thread; the paired low-pass run took about 30 minutes in the original study.

**Status.** Mixed: closed-form head matched a trained head, low-pass edit unconfirmed, motion-rule compilation not tested beyond its feature check. Code is cleaned from the run scripts (three scripts and a helper module replace two lab files and a shared core; unused stages, the wall-clock logging and the experiment ledger were dropped, paths became arguments, `kinematic_rule.py` keeps only the feature and check stages that produced the 0.4067 result). Verified here: `py_compile` and `ruff` pass. The cleaned scripts were run on the original data on the build server for two parts: the head comparison on the saved R3D-18 features (reproduces the accuracies and the +2.0 interval above) and `kinematic_rule.py` on the 300 clips (0.4067 probe, 0.3333 rule, median cadence 0 Hz). A synthetic-clip smoke run (30 generated AVI clips with the locally cached R3D-18 weights) ran every stage of all three scripts end to end; on those clips the dose-0.10 edit's weight accounting matched the original run (delta L2 57.45, max per-tensor relative change 0.205). The full low-pass run on the real clips was not repeated here; its table is extracted from the original run record.