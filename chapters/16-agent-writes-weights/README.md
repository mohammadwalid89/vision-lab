# 16. An agent that learns to write weights

**Question.** Can an agent write a model's weights by hand, and get better at it from its own failures?

**Setup.** Binary image tasks built from four public datasets: CIFAR-10 (objects), EuroSAT (satellite land use), GTSRB (traffic signs) and Oxford-IIIT Pets, as DINOv2-small features (CLS token plus mean patch token, 768 numbers). For each dataset the class pairs that a 16-shot head confuses most are picked. Each task has a fit split (16 images per class), a feedback split (up to 100 per class) and a larger test split that never enters an agent folder. The start head is a logistic regression on the 16-shot fit split. A Claude Opus agent (started through the `claude` command-line client) edits the head with numpy only: no fitting routines, no gradient steps, at most 30 scoring calls on the feedback split. It first reads a research playbook written by another agent with web search, then on every task builds at least three candidate heads, picks by cross-validation, and after each task (lessons arm) sees the true test result and writes a lesson and a scoreboard row that the next task reads. Controls: random edits with the same budget, a logistic regression trained on fit plus feedback, the same with its regularisation tuned by 5-fold cross-validation, and, in the first run, an arm with no lessons.

Two runs: v1 (`--version v1`, free-form method, lessons and no-lessons arms, 8 counted tasks) and v2 (`--version v2`, the required three-candidate method and scoreboard, lessons arm only, 11 tasks). The two Flowers-102 tasks of v1 are not counted: the start head already had accuracy 1.0 on their test split, so there was nothing to improve. DTD yielded no tasks (47 classes of about 40 images each, below the 56 images per class the pair selection requires).

**Result.** Test accuracy in points, from `results/summary_v2.json` and `results/summary_v1.json` (computed from the per-task files in `results/` by `summarize.py`):

| Run | Agent minus start | Minus random edits | Minus trained head | Minus tuned trained head |
|---|---:|---:|---:|---:|
| v2, 11 tasks, with lessons | +8.35 (better on 11 of 11) | +7.27 (11 of 11) | +0.08 (6 won, 3 tied, 2 lost) | +0.60 (9 won, 2 lost) |
| v1, 8 tasks, with lessons | +6.35 | +5.61 | -0.71 | not run |
| v1, 8 tasks, no lessons | +1.27 | +0.53 | -5.79 | not run |

In v2 the agent used 1 or 2 of its 30 scoring calls per task.

**What it means.** Handing the agent its earlier lessons is what moved it from a small gain over the start head to a result level with a head trained on the same labelled images (v1: -5.79 points behind without lessons, -0.71 with). The method the agents settled on is classical: a closed-form Gaussian (shrinkage LDA style) head computed from class means and one pooled covariance of the fit and feedback images (`agent_method/gaussian_head.py`). The finding is the agent discovering, tuning and safeguarding that method, not a new algorithm. Limits: small test sets (112 images for each Pets task in v1 and 166 for the first GTSRB task, per the run log), one task order, a few datasets, and the feedback images are part of what the final head is computed from.

**How to run.**

```
python extract_features.py work/features 5000          # downloads datasets and DINOv2-small via torchvision / Hugging Face
python suite.py build    --version v2 --workdir work/v2 --features work/features
python suite.py research --version v2 --workdir work/v2          # needs the agent client and web access
python suite.py run lessons --version v2 --workdir work/v2       # starts one agent per task
python summarize.py v2_lessons=work/v2/arm_lessons/results.jsonl
python suite.py run lessons --version v2 --workdir work/smoke --no-agent   # scoring pipeline only, no agent
```

The agent command is read from `WSP_AGENT_CMD` (default `claude -p --model opus`), so the runner needs a working client and credentials in your environment; none are stored here. Agent folders write `agent.jsonl` (the raw session stream) into the work directory; do not commit it. `suite.py` imports the `vision_lab` package, so install it or put `src` on the path first (root README, "How to run").

Isolation. Each agent session runs in a new folder outside the work directory (under `WSP_AGENT_TMP`, or the system temporary folder). A task session's folder holds only `fit.npz`, `feedback.npz`, the start head, the brief and the scoring tool; the research session's folder starts empty. The runner copies the folder back to `arm_<arm>/<task>/` or `research/` afterwards (`tests/test_isolation.py`). This is isolation by location: the agent's Bash tool can still read any path its user can read, so for a strict run use a user or container that cannot read the work directory. The published v1 and v2 runs gave each agent a folder inside the work directory, from which `../../tasks/<task>/test.npz` reached the test split, with only the prompt asking it to stay in its folder. Their 33 agent session logs (453 tool calls) were scanned on 2026-11-22: no call names the test split, the tasks folder, another arm or a parent folder (the research sessions' web sub-agents are not expanded in the logs). `extract_features.py --revision` pins the DINOv2-small commit for a new extraction; the published runs did not record which commit they used.

**Status.** Positive on the stated comparison (matched a trained head on 11 tasks; v2 beat the tuned trained head by 0.60 points). Code cleaned from the run version: the two run scripts (`suite_run.py` for v1, `suite_v2.py` for v2) are merged into `suite.py` with a `--version` switch; the agent prompts, briefs and the scoring tool are files in `prompts/` and `score_tool.py` (text unchanged except that the interpreter path was removed from the briefs); paths come from arguments; `--no-agent` was added for testing. Not included: agent session logs, the playbooks, lessons and scoreboards the agents wrote, per-task notes, the agents' per-task cross-validation scripts, features and datasets. `agent_method/gaussian_head.py` is the agent's own head function (from its `cv.py` scripts), reduced to the part that builds a final head.

Verified here: `py_compile` and `ruff` clean. On synthetic features (3 datasets, 64 dimensions), `suite.py build` for v1 and v2 produced task files and task order identical to the original `suite_run.py` and `suite_v2.py` run on the same input; `suite.py run lessons --no-agent` ran end to end for both versions. `summarize.py` reproduced the table above from the copied result files. `gaussian_head.py` rebuilt the agents' final heads from the source run's task folders: coefficients identical for gtsrb_0 (lambda 0.1), pets_2 and pets_3 (GDA); for cifar10_0 (lambda 0.75) the rebuilt head gives the same test accuracy (0.9389) but different coefficients, so the agent's final cifar10_0 head was not exactly this recipe. The other seven task heads were not checked. No agent or model was run and no dataset was downloaded; the v1 and v2 runs were not repeated.
