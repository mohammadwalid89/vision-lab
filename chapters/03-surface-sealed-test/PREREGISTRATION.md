# Phase 2 pre-registration (written 2026-10-28, before any fresh-set result was viewed)

## Question
How many good parts per 1,000 can be released automatically, at a defensible escape risk, with an improved
inspection architecture, compared with the deployed v001 system on the same images?

## Populations (by sha256; `out/inventory.json`)
- **Fit**: train good images of each product (MVTec AD train / MVTec LOCO train).
- **Calibration**: 5-fold out-of-fold scores on those train images (no image calibrates its own model).
- **Selection (dev)**: test/validation images of the 9 products that the research or Phase 1 already used
  ("exposed"). All design choices and all thresholds are chosen here only.
- **Final evaluation (fresh)**: test/validation images never used by the research or Phase 1
  (surface 200 defects / 100 good; kits 200 defects / 100 good). Scored by the jobs, sealed, opened once by
  `p2_final.py` after the policy is frozen. Still public benchmark images (MVTec AD / LOCO), same capture setup:
  this is an independent split, not a production distribution.

## Risk constraint
A release policy is acceptable only if
1. it releases **no** defect on the selection split it was chosen on, and
2. on the fresh split the one-sided 95 % Clopper–Pearson upper bound on the escape rate (released defects /
   defects) is **≤ 2.5 %** per family. (0 of 200 gives ≤ 1.5 %; 1 of 200 gives ≤ 2.3 %; 2 of 200 gives ≤ 3.1 %.)
Per-category bounds are reported but cannot be tight (25–90 defects per category).

## Selection rule (frozen before opening fresh)
Over 300 stratified half-splits of the selection set: choose (release threshold, hold threshold) on half A
minimising expected cost at a 2 % defect rate (review $0.25, false hold $2, escape $50) subject to zero
escapes on A; the deployed point is the most frequently selected one. Reported out-of-sample on half B, then
applied unchanged to fresh.

## Primary metric
Good parts released automatically per 1,000 good parts (fresh), with a 95 % Clopper–Pearson interval, at
the frozen policy, together with escapes (by structural / logical), false holds, person rate, cost per 1,000
parts at a 2 % defect rate, and CPU latency.

## Comparisons
v001 (ResNet-18 PatchCore + semantic L2, production policy) vs candidates; same images, same gate.

## Amendment 1 (2026-10-28 17:41 UTC, before any fresh-set result was viewed)
Design analysis on exposed images showed that out-of-fold *train* calibration is optimistic for the kit
verifier: exposed good kits from the LOCO validation and test sessions get p < 0.05 far more often than 5 %
(kNN score: 27 % validation, 53 % test), consistent with near-duplicate arrangements inside the train session.
Therefore, for kits, a second calibration source is added and selected by the same rule:
- **confirmed-goods calibration**: the *exposed* LOCO validation good images (`split == validation`,
  exposed) act as operator-confirmed good parts from a different session. They are removed from the kit
  selection population (they cannot both calibrate and evaluate). Fresh validation goods (22) stay in the
  fresh set.
- The calibration source (OOF-train vs confirmed goods) and the verifier score (max_z / top5_z / knn) are
  chosen on the selection population only.
Also measured: release rate as a function of the number of confirmed goods (10 / 25 / 50 / all), because
that is the form adaptation would take in a pilot.

## Amendment 2 (2026-10-28 18:18 UTC, before any fresh-set result was viewed; no policy run yet)
1. **One kit population for every variant.** The confirmed-goods pool (exposed LOCO validation goods) is removed
   from the kit selection population for *all* variants, including v001 and OOF-calibrated ones, so that every
   kit comparison is made on the same images.
2. **Choosing between architectures** (the pre-registration fixed the threshold rule but not this): per family,
   freeze the variant with the lowest mean out-of-sample cost over the 300 half-splits among variants whose mean
   out-of-sample escape rate is ≤ 1 %. Its deployed point is the policy most often selected on half A; if that
   point releases a defect on the full selection set (violating constraint 1), the full-set selection is used.
   v001 (as deployed) and v001_tuned (global thresholds, same rule) are frozen alongside as controls.
3. Component-verifier version: v1 (registered, no shape descriptors). v2 shape descriptors were worse on exposed
   breakfast boxes and are disabled (`P2_SHAPES=0`); their outputs are kept as `*_v2_shapes*`.
4. Confirmed-goods learning curve: 10 / 25 / 50 confirmed kits, 5 random draws each, 100 half-splits, exposed only.
