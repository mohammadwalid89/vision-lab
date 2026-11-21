# 03. Surface parts, sealed test

**Question.** Does a frozen WRN-50 PatchCore score with a per-category rule release more good surface parts than the original pipeline, without releasing defects?

**Setup.** Memory banks fit on MVTec AD training normals; five-fold out-of-fold calibration. 300 seeded development half-splits chose one policy. The policy and the scores were hash-frozen before a fresh set of 100 good and 200 defective images was opened once. Controls: the original pipeline and a threshold-only version, scored on the identical images. The plan, with two amendments made before the fresh set was opened, is in `PREREGISTRATION.md`. The same run also scored a kit family (MVTec LOCO); that result belongs to chapter 04.

**Result.** From `results/p2_final_results.json` (the one-time evaluation, systems `ad|v001`, `ad|v001_tuned`, `ad|phase2`) and `results/p2_frozen_policy.json` (the policy frozen before it):

| Arm (same 100 good, 200 defective images) | Good released | Defects released | Modeled person reviews per 1,000 parts |
| --- | ---: | ---: | ---: |
| Original pipeline | 4 of 100 | 0 of 200 | 956.8 |
| Threshold-only | 4 of 100 | 0 of 200 | 939.9 |
| WRN-50 stack | 84 of 100 | 0 of 200 | 162.9 |

The release rate is 84% (exact 95% interval 75.3 to 90.6%) against 4% (1.1 to 9.9%). By category, an in-sample subgroup view of the same 100 good images (good parts released): bottle 7 of 7, cable 22 of 25, hazelnut 13 of 15, tile 15 of 15, transistor 27 of 27, pill 0 of 11. Zero defects released out of 200 bounds the escape rate at 1.49% (one-sided 95% Clopper-Pearson), not at zero.

From the run report (not published), computed by `code/p2_report.py` from per-image score files that are not in this repo: 80 good parts gained release and none lost it (exact paired test p about 1.7e-24), and scoring time, median and 95th percentile, rose from 145 and 278 ms to 407 and 2,531 ms.

**What it means.** On this public benchmark split a calibrated per-category gate released far more good parts at no observed escapes. It does not cover every category (pill released none), it is not a factory lot test, and the review costs are modeled from assumed prices and a 2% defect rate, not measured.

**How to run.** The code in `code/` is the run's code. The scripts read and write `code/out/` and `code/models/`. `code/system_one/` holds the parts of the original pipeline (the v001 application) that the control arms import: `config.py` (paths, categories, the deployed thresholds), `reference.py` (builds and loads the calibration references and category gate in `state/reference.json`), `model.py` (the deployed decision rule, the `v001` arm) and `specialists.py` (PatchCore, CLIP contrast, the kit semantic L2 and the category gate).

| Step | Script | Writes |
| --- | --- | --- |
| Which images were already used (exposed) and which are fresh | `p2_inventory.py --data-root <dir with mvtec_ad/ and mvtec_loco/> --phase1-reference code/state/reference.json` | `out/inventory.json` |
| WRN-50 PatchCore banks, five-fold out-of-fold calibration, scores | `p2_patchcore_wrn.py [categories]` | `out/wrn_scores.jsonl`, `out/wrn_calibration.json` |
| Kit component verifier (chapter 04's Phase 2 arm) | `p2_components.py [kits]` | `out/comp_scores.jsonl`, `out/comp_calibration.json` |
| Original pipeline scores (the control arms) | `p2_baseline_scores.py` | `out/base_scores.jsonl` |
| Seal inputs, check the vectorised selector, select over 300 splits, freeze | `p2_policy_fast.py --regression`, then `p2_policy_fast.py 300` (selection rule in `p2_policy.py`) | `out/p2_frozen_policy.json` |
| Open the fresh set once | `p2_final.py` (refuses if a result already exists) | `out/p2_final_results.json` |
| Paired test, latency, report | `p2_report.py` | report and manifest |
| OWLv2 part-counting probe (chapter 04) | `p2_owlv2_probe.py` (`OWLV2_DIR`) | `out/owlv2_probe.json` |

`run_sealed.sh score` then `run_sealed.sh seal` runs the steps in the order they ran. Environment:

- `VISION_V31_ROOT`: the v3.1 build tree. Chapter 02's loaders read its score tables (to mark exposed images and to build the references), and `system_one` loads its `cascade` package (PatchCore loader, kit encoder and refinement) and its model files from it: the per-category PatchCore banks under `results/v31/local/`, the kit semantic models under `results/v31/local_kit_semantic/`, and the CLIP RN50 and ResNet-18 weights under `model_cache*/`. None of this is in the repository.
- `VISION_OUT`: chapter 02's output folder (default `out`), with its `heldout_ad_scores.jsonl` and, under `patchcore/`, the PatchCore banks of the nine held-out categories. `reference.build` and `p2_inventory.py` need `heldout_ad_scores.jsonl`. If it is missing, neither raises an error: `reference.build` returns a reference in which the nine held-out categories have no calibration images and no statistics, and `p2_inventory.py` marks as fresh the held-out images that only that file lists.
- `SYSTEM_ONE_STATE`: the folder holding `reference.json` (default `code/state/`). The run's file was built by the application's `bench/build_state.py` (not published), which called `system_one.reference.build(<gate embeddings>, gate_margin=3.0)` and saved the returned dictionary as JSON to `state/reference.json`. `build` writes no file itself, and its default `gate_margin` is 1.5, which halves every category gate radius; a gate refusal sends the image to a person in every arm. The gate embeddings came from the application's bench scripts, which are not published, and the file itself is a run output.
- `SYSTEM_ONE_APP`: only to use another copy of `system_one` (default: `code/`).
- `SYSTEM_ONE_THREADS`, `TORCH_HOME` (torchvision weights).

MVTec AD and LOCO are CC BY-NC-SA 4.0 and are not redistributed.

Without data, check that everything imports: `cd code && python -c "import system_one.model, system_one.reference, system_one.specialists, p2_policy, p2_baseline_scores, p2_final, p2_report"`.

**Status.** Done. Code restored from the run archive on 2026-11-21. Cleaned from the run version: hard-coded server paths replaced by arguments, environment variables or paths relative to this folder; module-level scripts given a `main()`; unused imports and one unused timer removed; ambiguous one-letter names renamed; the parts of `p2_report.py` that described and hashed the private preview service removed; `p2_report.py` now hashes `../PREREGISTRATION.md`, because in this repo the pre-registration sits in the chapter folder rather than next to the code (its manifest key is still `PREREGISTRATION.md`); code reformatted with ruff. No scoring, selection or threshold logic was changed. Four launcher scripts with server paths were replaced by `run_sealed.sh`. Left out: design-time analysis scripts, the app preview and its report page.

The v001 application subset in `code/system_one/` was added later the same day from a second owner restore. It holds only the four modules the control arms import. The server, learning loop, pipeline, bench scripts and tests are left out, and so are the AnomalyCLIP wrapper, the model-versioning helper and the server's upload limits, which no chapter uses. Cleaned from the archive copy:
- the default server path became `VISION_V31_ROOT`;
- `reference.build` reads chapter 02's `features.py`, the published copy of the module the run used, instead of the research folder;
- the PatchCore banks of the nine held-out categories are read from chapter 02's output folder (`VISION_OUT`), where chapter 02 writes them, instead of the research folder (corrected after review);
- one unused constant and one unused import were removed;
- comments that cited unpublished reports or files were shortened, and a comment's validation figures, which could not be checked, were removed (corrected after review);
- banner comments were removed and the code reformatted with ruff.

The decision rule, the references and the scoring code are unchanged (syntax-tree comparison). With `VISION_OUT` set as above, the held-out bank paths resolve to the same files the run used. The scripts now find `system_one` in `code/` by default.

Runnable from this repo plus private inputs: every script imports from the repository alone. Running the steps still needs the MVTec images, the v3.1 build tree and `state/reference.json`, and none of these is published. Re-running `p2_final.py` against `results/p2_frozen_policy.json` would also need the per-image score files, which are run outputs and not published.

Verified here:
- py_compile and ruff pass.
- On synthetic input, the vectorised selector chose the same policy as the original rule in 16 of 16 cases (the synthetic case of the run's own regression check). The component features and the coreset and nearest-neighbour code ran.
- With `system_one` in place:
  - every script in `code/` imports, with `system_one` resolved to `code/system_one/`;
  - the v001 decision rule gave output identical to the archive copy's on 3,000 synthetic cases;
  - on synthetic score files, `p2_policy`'s original-pipeline and threshold-only arms gave the same rows, selected policies and outcomes as the archive's `p2_policy` with the archive application;
  - `reference.build`, run on the recorded v3.1 score tables through chapter 02's loaders without gate embeddings, reproduced the archive build byte for byte;
  - with the private v3.1 tree, the category gate and PatchCore scored one synthetic image.
- Not run: the CLIP and kit semantic scorers, and anything on real images.

Every number on the Lab card was checked against `results/p2_final_results.json`. All match, except the paired test and the timings, which only the run report holds.
