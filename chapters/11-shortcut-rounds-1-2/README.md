# 11. Removing a planted shortcut: Rounds 1 and 2

**Question.** Can a visual shortcut be removed from a model's weights without raising false alarms?

**Setup.** The shortcut is planted: a 12 px magenta cross in the padding of every image, present on 90% of the defective training images and 10% of the good ones, so a classifier can learn "cross means defect". Data are the public VisA set plus the structural and good images of two MVTec LOCO categories (juice bottle, splicing connectors); sources are split 60/20/20 by group with seed 20261112. Development (DEV) has 2,359 source groups with 234 defects; the holdout stays sealed. Features are the last-layer pooled and mean-patch outputs of a frozen SigLIP2 base encoder (patch16-224; the lineage of the local checkpoint was not independently verified), 1,536 numbers per image; the classifier is a linear head with 1,536 weights and a bias. Three seeds (20261112 to 20261114).

Three families of edit, each with controls:

| family | edit | controls |
|---|---|---|
| V1 | project the marker direction out of the features, scaled by a dose | random directions, a direction from unrelated good images, a zero-dose sham |
| V2 | move the head weights along (B - A), where A is trained on the confounded view and B on both views; alpha scales the move (weight averaging and extrapolation) | random and shuffled deltas of the same norm, the reverse sign, a zero sham |
| OWN | the same move on a small programmed head: learned local probes feeding fixed presence, count and position logic | as V2 |

A retrained head on the same examples is the reference for V2 and OWN. The gates were signed before any run, and the worst of the three seeds decides each gate:

| gate | rule |
|---|---|
| marker reduction | relative reduction of the marker flip rate (a prediction that changes when the cross is added) at least 0.50 |
| placebo | the reduction is at least twice the best placebo's reduction |
| recall | recall within 0.02 below the unedited head |
| false alarms | false-positive rate at most 0.02 above the unedited head |
| marked-good errors | false positives on good images that carry the cross fall by at least 0.30 |
| retrain (V2) | beats the matched retrained head on marker flips by 0.05 without losing more than 0.01 AUROC |
| direction | the marker flip rate falls in every seed |

The threshold is fixed once per seed (maximum balanced accuracy for the unedited head on clean DEV) and reused for every arm.

There were three stages, 21 registered attempts in all:

1. **DEV order**: one attempt per family.
2. **Round 1**: three iterations per family (9 attempts). Each iteration changed the recipe after the previous failure: class-balanced training and a marker direction that excludes the label direction (iteration 2); longer training with a lower learning rate, stronger L2, and a direction that also excludes category means (iteration 3). The settings are in `arms.py`.
3. **Round 2**: a fixed grid of three families at three sample sizes (9 attempts). The edit may use k = 8, 32 or 128 labelled source pairs; k = 32 decides each family's primary conclusion, the other two are endpoints. Each edit and each retrainer gets 60 CPU seconds.

**Result.** None of the 21 attempts passed all gates (`results/summary.json`, 0 of 21). The failed gates at the Round 2 primary (k = 32), from the same file:

| family | failed gates |
|---|---|
| V1 | false alarms, recall |
| V2 | false alarms, recall, placebo, warm retrain, fresh retrain |
| OWN | false alarms, recall, AUROC distance to the retrained head, strict flip improvement |

Details that are in `results/summary.json`:

- V1 reduces the marker flip rate in every stage (worst-seed relative reduction 0.73 in the DEV order, 0.825 in Round 1 iteration 1, 0.861 at k = 32 in Round 2) and beats its placebos, but costs recall and false alarms. At k = 32, worst-seed recall margin is -0.056 and false-alarm margin -0.086 against bars of -0.02.
- The closest attempt is Round 1 iteration 3 of V1: it fails only the recall gate, at -0.026 against -0.02.
- V2 never beats the matched retrained heads on marker flips at k = 8, 32 or 128, and OWN never gets within 0.02 AUROC of its retrained reference.
- The false-alarm gate fails in every arm that removes more than 2% of the shortcut, including the retrained head (from the chapter 12 deletion probe report, not published here; see chapter 12).

Neither holdout was read, decoded or scored in any stage.

**What it means.** Edits at the head, deletion or weight averaging, did not remove this shortcut without trading it for missed defects or extra false alarms, and a head retrained on the same examples did no better. The limits: development data only, 21 registered attempts that reuse the same DEV split (so a pass would have been exploratory), one planted marker type, and cached features, so this says nothing about edits inside the encoder.

**How to run.**

```
export PYTHONPATH=src            # from the repository root
python -m unittest discover -s chapters/11-shortcut-rounds-1-2/tests -v
python chapters/11-shortcut-rounds-1-2/plant_marker.py --image img.png \
    --object-mask obj.npy --group-id g1 --split train --out-dir out/
python chapters/11-shortcut-rounds-1-2/summarize_verdicts.py --dev DEV_VERDICTS.json \
    --r1 r1/VERDICTS.json --r2-dir revision2/receipts --out results/summary.json
```

The last command needs the verdict files the runs wrote, which are not published; `results/summary.json` is its output. Image data come from `scripts/fetch_data.py` (public datasets only; MVTec LOCO is CC BY-NC-SA 4.0 and is never redistributed).

**Status.** Done, negative. Partial code: this chapter publishes the pieces that define the method and the gates, not the pipeline that produced the features. Published here and in `src/vision_lab/wsp/legacy/`: the heads, weight moves, feature ablation, programmed head, Round 1 recipes, Round 2 recipes (subspace edit, retrainers, controls), scorers and gate arithmetic, marker planting, and the verdict summary. Not published: the arm runners, which executed each arm under a private run registry with pinned file hashes; the data preparation (VisA intake, SAM2 object masks, SigLIP2 feature extraction); and the cached features. Cleaned from the run version: reformatted, dead code and registry checks removed, names and paths made arguments; the numerical code is unchanged. The marker planting code is the construction function without the mask model. Verified here: `py_compile` and `ruff` clean; 24 synthetic unit tests pass (heads, edits, controls, both scorers, gate arithmetic, the fast threshold scan against the exhaustive one, marker placement, summary); `summarize_verdicts.py` was run on the real verdict files to produce `results/summary.json`. Not run here: the arms on real data, `plant_marker.py` on real images.
