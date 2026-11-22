# 13. Round 3: mask the object, read the object

**Question.** Does removing the background from the input, or from the features, remove a planted shortcut where weight edits failed (chapters 11 and 12)?

**Setup.** The data is the R1 tree from chapter 11: 23,624 cached train and dev views of 9,453 source images (MVTec LOCO structural cases and VisA), each with a 12 px magenta cross planted in the synthetic padding around the image. In training the cross is on 90% of the defect images and 10% of the good ones, so a plain head learns it as the signal. Development data only; the sealed set was not opened.

Three methods ("slots"), each with a fixed recipe declared before any result was read:

| Slot | Method | Placebo |
|---|---|---|
| 1 | SAM object mask, everything outside the mask replaced by the encoder's mean colour, head refit on the masked views | random-area masks of equal area that spare the padding (3 draws) |
| 2 | Object-only patch features, weighted by SAM mask coverage | within-source permutation of the coverage weights (3 draws) |
| 3 | Delete 5% of the first-layer weights of an MLP head, learning which entries to delete with a clean-score preservation term | random deletion of the same count (3 draws) |

Nine fixed runs per encoder: three slots at three sample sizes (k = 8, 32, 128 source images, each with its clean and marked view, available to adapt on; k = 32 is primary), three seeds each (20261112 to 20261114), alpha .01. Two encoders, SigLIP2 base (224 px, primary) and DINOv2 ViT-B/14 with registers (added arm), are reported separately and never pooled. Other controls: padding-only trimming, test-time-only masking, uniform-weight aggregation, no-op controls (six function-preserving edits for slot 3), and warm and fresh retrainers on the same k examples. Gates 1a to 6 are carried over unchanged from chapter 11, plus gate 5 (retraining advantage) and an exact feasibility test on clean recall and false-positive rate. The threshold is frozen per pipeline on a calibration slice of 384 good and 128 defect sources that no fit touches (`src/vision_lab/wsp/calibration.py`). A pass requires every gate on every seed; the report also says whether the retrained reference passes the same gates, because if it does not the label is inconclusive rather than negative.

**Result.** No arm passed. All 18 attempts ended `NULL_OR_NEGATIVE`, and the no-op integrity check held in every run (`results/summary.json`, derived from the attempt files of the run, which are not published). Gates passed on the worst seed, out of 8 (k = 32; the full table is in the summary):

| Encoder | Slot 1 (SAM mean fill) | Slot 2 (object-only features) | Slot 3 (5% deletion) |
|---|---|---|---|
| SigLIP2 | 6 of 8 | 2 of 8 | 4 of 8 |
| DINOv2 | 7 of 8 | 1 of 8 | 4 of 8 |

- DINOv2, slot 1 passed marker reduction, placebo margin, recall, false-positive rate, conflict reduction and same-direction-on-all-seeds at every k, and the feasibility test (worst recall margin +0.1368). It failed only gate 5: the candidate's marker flip rate fell from 0.7308, 0.7342 and 0.7325 (plain head, three seeds) to 0.0, but a retrainer fitted on the same masked views also reaches 0.0, so the advantage is 0.0 against the required 0.05.
- SigLIP2, slot 1 removed the marker as well (flip rate 0.0) but failed the placebo margin (worst seed -0.1212) in addition to gate 5.
- In DINOv2 the padding-only reference (trim the padding, no SAM) also had flip rate 0.0 on seed 20261112 (`marker_flip_rate_by_arm` in the summary).
- Slot 2 failed most gates on both encoders; its flip rate stayed near the plain head's (SigLIP2, seed 20261112: 0.7020 against 0.7410), so the object-only aggregate still carries the marker.
- Slot 3 passed between 2 and 4 of 8 gates on both encoders, at every k.

**What it means.** Masking the input kills the marker's effect on the features, but so does training on the same examples with the masked inputs, so the experiment does not show that the input fix adds anything over retraining. The open question is that one gate: whether the input fix adds anything once retraining is allowed, and the next step is to combine them. The limit: development data only, one planted marker type, two encoders, three seeds drawn from the same sources (not independent datasets).

**How to run.** From the repo root, with the chapter 11 data tree and local model weights (no downloads happen here):

```
export PYTHONPATH=src
export VISION_R1_ROOT=/path/to/r1_tree        # r1/INTAKE.json, r1/cache/..., r1/masks/..., r1/data/...
export VISION_SIGLIP_DIR=/path/to/siglip2_base
export VISION_DINOV2_FILE=/path/to/dinov2_vitb14_reg4.safetensors
R3=chapters/13-round3-mask-the-object/run_r3.py
OUT=r3_run

python $R3 plan --out $OUT --content-box 24,24,360,360 --plan-overrides chapters/13-round3-mask-the-object/plan_overrides.json
python $R3 preflight --out $OUT --views-dir $VISION_R1_ROOT/r1/data --published-partition CALIBRATION_PARTITION.json
python $R3 extract --out $OUT --views-dir $VISION_R1_ROOT/r1/data
for k in 32 8 128; do for s in slot1 slot2 slot3; do
  python $R3 attempt --out $OUT --views-dir $VISION_R1_ROOT/r1/data --slot $s --k $k; done; done
python $R3 report --out $OUT
```

Repeat preflight, extract, attempt and report with `--arm dinov2` for the second encoder. `plan` prints the plan's sha256; the run reported here used `d1e0a9b1549ab648c778cb8a8f1dd6e04c114d6c6dea4afb7ac285afac670741` (`results/R3_PLAN.json`). Extraction is the expensive step (hours on 3 CPU cores); everything else takes minutes.

Synthetic end-to-end run (seconds, no models, no data): `python $R3 dry-run --out /tmp/r3_dry`.
Condense the attempt files: `python chapters/13-round3-mask-the-object/summarize_results.py --run-dir $OUT`.
Tests: `PYTHONPATH=src python -m pytest -q chapters/13-round3-mask-the-object/tests`. Set `R1_SCORER_DIR=src/vision_lab/wsp/legacy` to run the scorer and gate parity tests (64 pass). The ordering parity test needs `ordered.py` from the Round 1 run, which is not published, and stays skipped; so does one metrics test that loads `probe_lib.py` from the same folder (that file is in chapter 12). Set `VISION_DINOV2_FILE` for the DINOv2 encoder tests.

**Status.** Done, negative. The library in `src/vision_lab/wsp/` and `run_r3.py` are the code that produced the results, with these cleanups: imports renamed to `vision_lab.wsp`, hard-coded server paths replaced by arguments and `VISION_*` environment variables, banner comments and internal notes removed, one-line statements split; no logic changed. The SigLIP mean-patch and chunked-forward edits made on 2026-11-18, before the preflight, are included. Verified here: `py_compile` and `ruff check --select E,F,W,I --ignore E501` clean; `plan` with `plan_overrides.json` and `--content-box 24,24,360,360` reproduces the run's `R3_PLAN.json` byte for byte (same sha256 above); `dry-run` on the synthetic tree completes both arms; the test suite gives 61 passed and 14 skipped without `R1_SCORER_DIR`, 64 passed and 11 skipped with `R1_SCORER_DIR=src/vision_lab/wsp/legacy` (counts from 2026-11-22, after the test for a gate missing from a seed was added), and gave 65 passed and 9 skipped with the original Round 1 scorer modules (not published) before that test was added. Not run here: preflight, extraction and attempts on real data, the DINOv2 and SigLIP encoder tests (skipped; no weights set), and any re-run of the experiment. The slot-1 placebo choice and the added encoder arm are recorded in the repo CHANGELOG.
