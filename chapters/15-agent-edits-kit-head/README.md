# 15. Kits: an agent edits the head

**Question.** Can a frontier model, given weight access, fix the kit head where training is short of data?

**Setup.** One Claude Opus agent (model id `claude-opus-5-5`, from the run's session metadata) works in its own folder. It gets a research brief (`brief_template.md`), Python and numpy, the 1,536 weights of a linear head over SigLIP2 features (pooled embedding plus last-layer mean patch token), two feature files (`edit_fit.npz`, `edit_val.npz`) and a scoring tool (`score_tool.py`) capped at 50 calls. It may not use any fitting or gradient routine. Category: MVTec LOCO juice bottles. The starting head is a logistic regression on the fit file (normal kits plus 60% of the labelled good and faulty training kits); the other 40% of labelled training kits plus calibration normals form the feedback set (92 images, 23 faulty). The final head is scored once on development kits the agent never saw (43 faulty, 28 good). Controls: the starting head, a random-edit placebo with the same budget (`placebo.py`), and the trained linear head from chapter 14.

**Result.** AUC on unseen development kits, from `results/summary.json`:

| Head | AUC |
|---|---:|
| Start head | 0.8405 |
| Random edits, same budget | 0.8488 |
| Agent's final head | 0.7907 |
| Trained linear head (chapter 14, fitted on all training rows) | 0.8879 |

On the feedback images the agent's final head scored 0.9156 against 0.8639 for the start head and 0.8866 for the random edits; its last scored candidate reached 0.9244. It used 18 of the 50 scoring calls. The final edit is one dense direction (relative L2 change 0.876 of the weight vector, all 1,536 weights touched). The agent's own notes (not published) explain the direction as a bottle/label mismatch lying between two normal product variants; the `agent_method/` scripts rebuild that head exactly (see Status).

**What it means.** The agent's changes fit the 23 faulty kits it could see and did not carry to new ones: unseen-kit AUC fell below both the start head and random edits. With one category and 43 faulty development kits this is a single, noisy observation. The session transcript was audited by the experimenter (nothing read outside the folder, no fitting tools used); the transcript is not published.

An earlier, separate attempt used a model to propose parameters of the chapter 14 written score over 10 rounds, with a random-proposal loop as placebo. Its development AUCs (written score only) were 0.7471 for the unedited written head, 0.7105 for the model's best proposal and 0.7629 for the random loop (from the run logs, not published; that code is not included because those numbers are not part of this chapter's result and it calls a model through a command-line client).

**How to run.** The agent session itself is not reproducible from this repository (no transcript is included and no model is called by any script here). The harness and the agent's recorded method are:

```
# features and split from chapter 14 first
python prepare_sandbox.py --features ../14-kits-written-checks/work/features --category juice_bottle --out work/
#   -> work/sandbox_juice_bottle/ (give the agent this folder only)
python placebo.py --sandbox work/sandbox_juice_bottle --out work/placebo_final.npz
python dev_eval.py --features ../14-kits-written-checks/work/features --out work/DEV_RESULT.json \
    work/sandbox_juice_bottle/start_head.npz work/placebo_final.npz work/sandbox_juice_bottle/final.npz
# the agent's final edit, as a script
python agent_method/derive_variants.py --sandbox work/sandbox_juice_bottle --out work/var.npz
python agent_method/build_head.py --sandbox work/sandbox_juice_bottle --variants work/var.npz \
    --out work/final_rebuilt.npz --alpha 2.0 --protect 387,379
```

`placebo.py` and `dev_eval.py` import the `vision_lab` package, so install it or put `src` on the path first (root README, "How to run").

**Status.** Negative (the agent lost on unseen kits). Code cleaned from the run version: server paths replaced by arguments; the setup script that generated the other files at run time split into `prepare_sandbox.py`, `score_tool.py` and `brief_template.md` (the brief is unchanged except that the interpreter path was removed); `dev_eval.py` and `placebo.py` reformatted with unchanged logic. `agent_method/` holds the agent's own analysis scripts reduced to the two that produce its final head (`derive_variants.py`, `build_head.py`, reformatted; the row indices and hyperparameters are the agent's, chosen from the feedback images). Not included: the agent's conversation log, its notes, its intermediate candidates and its exploratory scripts. Verified here: `py_compile` and `ruff` clean; against the source run, `prepare_sandbox.py` reproduces `edit_fit.npz`, `edit_val.npz` and `start_head.npz`, `placebo.py` reproduces the placebo head bit for bit, the agent scripts reproduce `final.npz` exactly (max absolute difference 0.0), and `dev_eval.py` reproduces the three development AUCs above from the cached features of the source run (read-only). `score_tool.py` was compiled but not run. No model was called.
