# 12. Two probes: deletion and threshold

**Question.** Why did no arm in chapter 11 pass: the edit, or the threshold it was scored with?

**Setup.** Two exploratory probes on development (DEV) data, run after the Round 2 campaign. Neither is a new registered attempt; each fixed its arms and budgets in a preregistration file before running. Both use the chapter 11 setting: cached SigLIP2 features (1,536 per image), the Round 1 iteration-1 linear head A, three seeds, k = 32 labelled source pairs (64 views) for every edit, the same six gates, and the same 9,436 DEV views.

1. **Deletion probe** (`run_deletion_probe.py`). Which factor lets a weight edit remove the marker shortcut without false alarms: (F1) deletion only, or (F2) how many weights may change? Two heads: the linear head (1,536 weights) and a 1536-32-1 MLP trained on the confounded training views, edited in its first layer (49,152 weights). Arms: a learned deletion mask (DEL-MASK), a learned signed change (SIGNED-SPARSE), three random deletions as placebo, a retrained head warm-started from A, a retrained head from scratch, and A itself. Budgets of 1, 5, 10, 25 and 50% of the maskable weights; 10% is primary. Masks are learned with a straight-through sigmoid gate over 300 full-batch epochs (Adam, temperature 1.0 to 0.2).
2. **Threshold re-anchor probe** (`run_reanchor_probe.py`). Each arm is scored at three operating points: its family's original frozen threshold (a); a threshold re-fit on a shared TRAIN-only calibration slice of 512 sources, 128 defective and 384 good (b); and a threshold re-fit on clean DEV (dfit, a diagnostic only). It also checks, without any threshold, whether the recall and false-alarm gates can be met together. Cells: the linear head with its arms and the Round 1 V1 edit, the MLP with its arms, and the Round 2 V1 edit at k = 32.

**Result.** Numbers are from the two probe reports (not published) and the per-seed tables in `results/`. Gate values are the worst of three seeds.

*Deletion probe.*
- The false-alarm gate (FPR margin) fails in every one of the 42 arm cells whose marker reduction is above 2%, and in every one of the 37 cells above 5%, including the retrained head. It passes only for A itself and for random deletion at 1%, which changes nothing (marker reduction -0.04 to +0.01).
- With a frozen negative threshold, deletion's marker-flip reduction is mostly score shrinkage: linear DEL-MASK at 10% shows a 33% reduction at the frozen threshold and about 3% after the threshold is re-fit.
- At matched support a signed change removes far more marker response than zeroing weights does, but it spends its norm budget reshaping scores (linear AUROC 0.848 to 0.794 for SIGNED-SPARSE at 1%). The only harmless sparsity window is about 1% of weights, which is also where nothing happens.
- The closest arm is the retrained MLP head: 5 of 6 gates, marked-good false positives -36.5% and marker flips -81.5% in the worst seed, failing only the false-alarm gate (-0.164).

*Threshold re-anchor probe.*
- Moving each arm to its own DEV-fit threshold removes 62 to 91% of the false-alarm deficit for the weight-edit arms. The TRAIN-slice re-anchor, the registered rule, removes 28 to 83% for most arms and makes the Round 2 V1 edit slightly worse.
- No arm passes all six gates under any of the three rules; the best is 5 of 6 (at the frozen threshold, the retrained MLP head; at the DEV-fit point, also the MLP DEL-MASK at 5% and the Round 2 V1 edit).
- Threshold-free, every arm fails the recall and false-alarm gates together at every threshold. MLP DEL-MASK at 5% is closest, with a false-alarm margin of -0.052 at matched recall and a recall margin of -0.043 at matched false-alarm rate (bars -0.02).
- The best deletion arm: MLP DEL-MASK at 5% (2,458 of 49,152 first-layer weights) removes 64.3% of the marker flips at its DEV-fit threshold, with AUROC up 0.6 points (0.894 against 0.888), recall +0.013 and marked-good false positives -39.9%, and passes 5 of 6 gates there. It fails the false-alarm gate (-0.114). At the frozen threshold the same arm passes 4 of 6.
- A TRAIN-only calibration slice is a poor proxy for the deployment threshold when an edit moves the score scale: mean gap 0.06 to 0.17 logits for A and the linear arms, 0.45 to 1.0 for the MLP arms and retrained heads.

**What it means.** The single frozen threshold sets the size of the reported false-alarm failure, and for deletion on a linear head it explains most of the apparent effect; the edited weights decide whether any threshold can pass, and none could. The best starting point for a follow-up is deletion in a hidden layer. The limits: head-level only (the edited layer sits on frozen features, not inside the encoder); fixed k = 32 with three seeds and no confidence intervals; 56 arm-by-budget cells with no multiplicity correction; the DEV-fit numbers are in-sample and optimistic. Exploratory, not confirmatory.

**How to run.** The probes read the Round 1/2 experiment tree (cached features, fitted heads, intake and manifests), which is not published.

```
export PYTHONPATH=src            # from the repository root
python chapters/12-deletion-threshold-probes/run_deletion_probe.py \
    --wsp-root <round-1/2 tree> --out-dir out/deletion      # about 13 minutes on two cores
python chapters/12-deletion-threshold-probes/run_reanchor_probe.py \
    --wsp-root <round-1/2 tree> --deletion-results out/deletion/results_raw.json --out-dir out/reanchor
python -m unittest discover -s chapters/12-deletion-threshold-probes/tests -v
RUN_PROBE_SMOKE=1 python -m unittest discover -s chapters/12-deletion-threshold-probes/tests -v   # synthetic tree, about 5 minutes
```

`results/` holds the per-seed and worst-seed tables of both probes as the runs wrote them (`*_per_seed.csv`, `*_agg.csv`).

**Status.** Done, exploratory. Code is the probe runners and their shared library, cleaned from the run version (paths and the input-hash table replaced by arguments, dead code and an unused pilot removed, reformatted); the arithmetic is unchanged. Shared code (scorer, gate arithmetic, source ordering, subspace edit) is in `src/vision_lab/wsp/legacy/`. Not included: the TRAIN-only pilot that chose the mask optimiser settings, the integrity-check scripts, and the agent launch scripts. Verified here: `py_compile` and `ruff` clean; 9 synthetic unit tests pass; both runners' arm construction, gate evaluation and aggregation ran end to end on a synthetic tree (the deletion probe through `main()` with `--skip-base-check`, the re-anchor probe through its arm, gate and aggregation functions, not through `main()`, which also needs the Round 2 artifacts). Not run here: either probe on the real data.

Open point. As coded, the SIGNED-SPARSE delta is added to the weights the mask keeps (all but the r lowest-scoring), and the r selected weights keep their original value. The original probe report describes the signed change as acting on the same r coordinates. The SIGNED-SPARSE rows should be read with the code's definition; DEL-MASK, which carries the headline number, is unaffected.
