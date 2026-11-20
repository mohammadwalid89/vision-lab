# 04. Kits: logical faults

**Question.** Does the surface approach carry over to kits, where faults are logical (a missing or wrong part)?

**Setup.** Phase 2: 100 good and 200 defective sealed MVTec LOCO kits. Phase 3: two new LOCO categories (`juice_bottle`, `splicing_connectors`), 335 and 360 training normals, 54 and 60 validation normals. An exposed development subset (84 good, 149 defective) and a locked final set (129 good, 228 defective, of which 151 logical and 77 structural). Ten candidates were registered before the final set was opened: four scores (component colour/material counts, component spatial relations, PatchCore on ResNet-18, PatchCore on WRN-50) and six min-p fusions of them. The eligibility rule, set in advance, was a mean development escape rate of 1% or less over 100 deterministic half-splits; if no candidate qualified, every kit would go to a person.

**Result.** From `results/audit_records.json` and `results/audit_frontier.json`, written by `audit_phase3.py` from the frozen Phase 3 records:

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

From the research atlas and the Phase 2 report (not published): Phase 2 released 0 of 100 good kits; an OWLv2 part counter gave wrong counts at 33 to 43 s per image on the CPU.

**What it means.** Image anomaly scores did not separate logical faults from normal variation tightly enough to meet a 1% escape bar, so every kit goes to a person. The audit also shows why: the development "zero escapes" table is circular (the release threshold is the largest defect p-value), and the candidates were selected on a population that was one third structural. This is a result for these ten candidates, not a proof that kits cannot be automated. Chapter 14 tries written part checks instead.

**How to run.** The audit needs the original Phase 3 record folder (frozen policy, development scores, final results), which is not in this repo:

```
python chapters/04-kits-logical-faults/audit_phase3.py records  --phase3-dir <records> --out audit_records.json
python chapters/04-kits-logical-faults/audit_phase3.py frontier --phase3-dir <records> --out audit_frontier.json
```

MVTec LOCO is CC BY-NC-SA 4.0 and is not redistributed.

**Status.** Partial. The code that scored the ten candidates and froze the policy is in the archive (its folder was moved there after the run), so it is not here. What is here is the later audit script that re-reads those frozen records, with the result files it wrote. Cleaned from the audit run: the two audit scripts were merged into one with a `--phase3-dir` argument, paths and a recorded source path were removed, and the hashes of three report files were dropped from the output. Verified here: `py_compile` and `ruff` pass; both subcommands ran on a toy record folder made of random numbers (exit 0), not on the real records. The Phase 2 and OWLv2 figures come only from the atlas.
