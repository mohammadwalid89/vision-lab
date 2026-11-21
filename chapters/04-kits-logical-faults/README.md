# 04. Kits: logical faults

**Question.** Does the surface approach carry over to kits, where faults are logical (a missing or wrong part)?

**Setup.** Phase 2 (chapter 03's run, kit family): 100 good and 200 defective sealed MVTec LOCO kits. Phase 3: two new LOCO categories (`juice_bottle`, `splicing_connectors`), 335 and 360 training normals, 54 and 60 validation normals. An exposed development subset (84 good, 149 defective) and a locked final set (129 good, 228 defective, of which 151 logical and 77 structural). Ten candidates were fixed before the final set was opened: four scores (component colour/material counts, component spatial relations, PatchCore on ResNet-18, PatchCore on WRN-50) and six min-p fusions of them. The eligibility rule, set in advance, was a mean development escape rate of 1% or less over 100 deterministic half-splits; if no candidate qualified, every kit would go to a person.

`PREREGISTRATION.md` was written before any new evaluation image or score was loaded. It holds the data and its source (with a change of data source made before any score was read), the 40/60 split, the four candidate types, a zero-development-escape constraint and a 2.5% bound on the final escape rate. It does not hold the 1% rule, the ten candidates or the all-to-a-person fallback. Those were fixed in the selection code (`code/p3_select_freeze.py`) and written into `results/p3_frozen_policy.json` (fields `selection_rule`, `fallback` and `candidates`, with the SHA256 of the selection code), frozen at 20:54:17 UTC on 2026-10-28. The final evaluation refuses to run unless that file and its hash are in place, and it completed at 21:01:41 UTC (`results/p3_final_results.json`). The 1% rule is the one Phase 2's Amendment 2 wrote down (`../03-surface-sealed-test/PREREGISTRATION.md`).

**Result.** From `results/p3_frozen_policy.json` (development selection, frozen before the final set was opened) and `results/p3_final_results.json` (the one-time final evaluation), both written by the run, and from `results/audit_records.json` and `results/audit_frontier.json`, written later by `audit_phase3.py` from the frozen records:

- No candidate qualified. Mean development escape rates ranged from 1.82% (component only) to 2.93% (component + ResNet), so the fallback applied: all kits to a person, which gives 0 of 129 good kits released and 0 of 228 escapes by construction.
- The locked final set, shown for reference only (the candidates were not selected on it):

| Candidate | Good released (of 129) | Defects escaped (of 228) | 95% upper bound on escape rate |
| --- | ---: | ---: | ---: |
| WRN only | 27 | 3 | 3.37% |
| Component + WRN | 17 | 0 | 1.31% |
| WRN + ResNet | 14 | 1 | 2.06% |
| Relational + WRN | 11 | 3 | 3.37% |
| Relational + ResNet | 10 | 7 | 5.69% |
| Component + ResNet | 6 | 0 | 1.31% |
| ResNet only | 4 | 3 | 3.37% |
| Four-way fusion | 4 | 0 | 1.31% |
| Component only | 1 | 0 | 1.31% |
| Relational only | 0 | 0 | 1.31% |

Relational + ResNet let through 2 logical and 5 structural defects. An in-sample sweep of the release threshold (optimistic, chosen on the rows it scores) found at most 7 of 84 good development kits releasable within a 1% escape budget (ResNet only, and Relational + ResNet).

Phase 2 released 0 of 100 good kits and 0 of 200 defects (`../03-surface-sealed-test/results/p2_final_results.json`, system `loco|phase2`; the frozen kit winner was the threshold-only control). From the Phase 2 run report (not published): an OWLv2 part counter (`../03-surface-sealed-test/code/p2_owlv2_probe.py`) gave wrong counts at 33 to 43 s per image on the CPU.

**What it means.** Image anomaly scores did not separate logical faults from normal variation tightly enough to meet a 1% escape bar, so every kit goes to a person. The audit also shows why: the development "zero escapes" table is circular (the release threshold is the largest defect p-value), and the candidates were selected on a population that was one third structural. This is a result for these ten candidates, not a proof that kits cannot be automated. Chapter 14 tries written part checks instead.

**How to run.** `code/` holds the Phase 3 run. It imports chapter 03's component verifier (`P2_CODE_DIR`, default `../03-surface-sealed-test/code`) and reads and writes `code/out/`, `code/models/` and `code/raw/`.

| Step | Script |
| --- | --- |
| Download the two categories from the pinned public mirror the run used (official source: MVTec) | `download_new_categories.py` |
| Fix the 40/60 development / final split from file paths alone, before any model runs | `build_split.py` |
| Fit on training normals, calibrate on validation normals, score development images | `p3_models.py --model component\|resnet\|wrn --category <kit>` |
| Check the final-scoring code on validation normals | `p3_preflight.py` |
| Select on development only, freeze and hash | `p3_select_freeze.py` |
| Open the locked final set once | `p3_final.py` (refuses if it was opened before) |
| Report | `p3_report.py` |

`code/run_phase3.sh` runs them in order. The audit re-reads the frozen records:

```
python chapters/04-kits-logical-faults/audit_phase3.py records  --phase3-dir <records> --out audit_records.json
python chapters/04-kits-logical-faults/audit_phase3.py frontier --phase3-dir <records> --out audit_frontier.json
```

MVTec LOCO is CC BY-NC-SA 4.0 and is not redistributed.

**Status.** Done. The Phase 3 code was restored from the run archive on 2026-11-21. Cleaned from the run version: server paths replaced by paths relative to this folder or `P2_CODE_DIR`, module-level scripts given a `main()`, unused imports and an unused constant removed, `O` renamed `OUT`, code reformatted with ruff; four launcher scripts replaced by `run_phase3.sh`. No scoring, split, selection or threshold logic was changed. Two consequences: `p3_final.py` checks the hash of `p3_select_freeze.py` against the frozen policy, so it will not reopen the published policy with this cleaned file (a fresh run freezes its own hash); and, as run, the byte-overlap check in `build_split.py` can never fire, because the line that records each hash sits inside the branch that raises (kept as run and marked in the code). The run's docstrings and the report text written by `p3_report.py` call the selection and the fallback "preregistered"; that wording is kept as run so the report still rebuilds line for line, and the Setup above says which parts are in `PREREGISTRATION.md` and which are in the frozen policy.

Verified here: py_compile and ruff pass. `p3_report.py` run on `results/p3_frozen_policy.json` and `results/p3_final_results.json` rebuilt the archived Phase 3 report line for line (84 of 84 lines, including the SHA256 of both files), so the published JSON files are the ones the run wrote. `p3_select_freeze.evaluate` ran on synthetic rows (zero development escapes by construction, as designed). Every number on the Lab card was checked against the two JSON files; all match, except the OWLv2 timing and the training and validation counts, which only the run reports hold. The audit ran on a toy record folder (round 1). Not re-run on real data: the per-model development score files and the split manifests are run outputs and are not in this repo.
