# Brief: program the weights of a kit-inspection head (weight-space programming)

You are the weight programmer in a pre-registered experiment. Work only inside this folder.
Do not read, list or search anything outside it. Do not install packages. Python with numpy is available.

## The task (domain)
MVTec LOCO AD, category `{CATEGORY}`. Each image is a product kit. A *logical* fault is a kit that looks
locally normal but breaks a rule: a part missing, an extra part, a part in the wrong place, the wrong
amount or size (e.g. fill level, label type, label position). Good kits are label 0, logical faults label 1.
Each image is represented by a 1,536-number feature vector: the first 768 are the SigLIP-2 base pooled
embedding, the last 768 are the mean of the last-layer patch tokens.

## The model you edit
A linear head: score = X @ coef + intercept. Higher = more likely faulty. Images above the 99th
percentile of normal calibration scores would be sent to a person. `start_head.npz` is a logistic
regression trained on `edit_fit.npz` (normal kits plus 60% of the labelled good and faulty kits).
`candidate.npz` is what you edit (it starts as a copy of start_head). It must keep the keys coef
(1,536 float64) and intercept.

## Your tools and budget
- `edit_fit.npz` and `edit_val.npz` (X, y, ids, roles). You may read and analyse both freely.
- `python score.py [file]` scores a candidate on edit_val and lists the images it gets wrong.
  It is your only forward-pass feedback and is capped at **50 calls** (logged). Plan your calls.
- You may NOT call any gradient-based optimiser or fitting routine (no sklearn, no torch, no
  np.linalg.lstsq/solve on the labels, no gradient steps you compute on the labels). The point of the
  experiment is that the weights are written by you, from your reading of the weights and the data,
  not trained. Reading statistics (means, differences, projections, correlations) to decide an edit is fine.

## What the research says (use it)
- Sparks of Weight-Space Programming (Takehi, Teoh, Zhang, Khattab, MIT CSAIL, Nov 2026): a frontier
  LLM read a network's weights, located the part causing a failure, and edited a tiny fraction
  (0.016%) to fix it; with Python weight access and a budget of 0/10/25/50 forward-pass evaluations.
  The edits were targeted at a diagnosed failure, not broad re-weighting.
- Weight-Space Programming Doesn't Need Transformers (Ok and Lee): written weights fail on raw pixels
  but work over learned foundation-model tokens (96.1% on DINOv2 tokens). Your features are learned tokens.
- Weight-Space Reading (Kocabay): readers with weight access predict edit effects better than from text,
  but over-predict change on edits that do nothing. Verify every edit with a score call.
- Our own earlier attempts: editing thresholds on hand-made part measurements did not work, because those
  measurements barely separate good from faulty kits. Random re-weighting matched or beat an LLM making
  small blind edits. Do not make small blind edits: diagnose first.

## Suggested way to work
1. Read the failures: which faulty kits score low, which good kits score high. Compare their features
   with the class means (in edit_fit) to find directions that separate them.
2. Form a hypothesis about what the head is missing (e.g. a direction for a missing part or a fill level),
   write it down, and make a targeted edit (add or rescale a direction, change a few weights).
3. Score it. Keep what helps. Write each hypothesis, edit and result to `NOTES.md` as you go.

## Goal and finish
Maximise AUC on edit_val without raising false alarms on good kits. When done (or the budget is spent),
leave the best weights in `final.npz` (keys coef, intercept) and finish NOTES.md with: the final edit,
why it works, how many weights changed, and how many score calls you used. The final head is scored once,
by someone else, on images you never see.
