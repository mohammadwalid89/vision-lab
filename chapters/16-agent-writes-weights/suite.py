"""Learning suite: an agent writes the weights of a two-class image head, task after task.

build     make binary tasks from cached DINOv2-small features: for each dataset, the
          most-confused class pairs under a 16-shot head. Each task has fit (16 per class),
          feedback (up to 100 per class) and a large unseen test split. Test never enters
          an agent folder.
research  one agent with web search writes PLAYBOOK.md (shared by both arms).
run ARM   ARM is `lessons` or `fresh`. Tasks run in a fixed order; per task an agent edits
          the 16-shot head with a budget of scoring calls on the feedback split and no
          training. The runner then scores the start head, the agent's final head, a
          random-edit placebo and trained heads on the test split. In the `lessons` arm
          the agent then sees its test result and appends a lesson that the next task reads.

--version v1 is the first run (8 counted tasks, free-form method); v2 is the second run
(11 tasks, at least three candidate methods per task chosen by cross-validation, and a
running scoreboard METHODS.md). The agent is started through a command-line client given
by $WSP_AGENT_CMD (default: `claude -p --model opus`); use --no-agent to skip all agent
calls and only run the scoring pipeline.
"""
import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
PROMPTS = HERE / "prompts"
DATASETS = ["cifar10", "eurosat", "dtd", "gtsrb", "flowers102", "pets"]
SHOT, FB, BUDGET = 16, 100, 30
SEED = 20261119

# Per-version differences: pairs built per dataset, tasks kept per dataset, interleave rounds.
VERSIONS = {
    "v1": {"pairs": 2, "max_per_dataset": None, "skip_easy": False, "rounds": 2},
    "v2": {"pairs": 8, "max_per_dataset": 4, "skip_easy": True, "rounds": 4},
}


class Suite:
    def __init__(self, workdir, version, no_agent=False):
        self.root = Path(workdir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.version = version
        self.spec = VERSIONS[version]
        self.tasks = self.root / "tasks"
        self.no_agent = no_agent
        self.agent_cmd = shlex.split(os.environ.get("WSP_AGENT_CMD", "claude -p --model opus"))

    def log(self, m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(self.root / "suite.log", "a") as f:
            f.write(line + "\n")

    def prompt(self, name):
        return (PROMPTS / f"{name}_{self.version}.txt").read_text().strip()

    def agent(self, cwd, prompt, tools, minutes):
        if self.no_agent:
            return 0
        cmd = self.agent_cmd + ["--permission-mode", "acceptEdits", "--allowedTools", *tools,
                                "--output-format", "stream-json", "--verbose", "--", prompt]
        with open(cwd / "agent.jsonl", "a") as f:
            r = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                               timeout=minutes * 60)
        txt = (cwd / "agent.jsonl").read_text()[-3000:]
        if "hit your" in txt and "limit" in txt:
            raise SystemExit("usage limit hit; stopping cleanly")
        return r.returncode

    def build(self, features):
        spec = self.spec
        self.tasks.mkdir(exist_ok=True)
        order = []
        for ds in DATASETS:
            f = Path(features) / f"{ds}.npz"
            if not f.exists():
                self.log(f"build: {ds} missing, skipped")
                continue
            d = np.load(f)
            X, y = d["X"].astype(np.float64), d["y"]
            rng = np.random.default_rng(SEED)
            classes = [c for c in np.unique(y) if (y == c).sum() >= SHOT + 40]
            mu = {c: X[rng.permutation(np.where(y == c)[0])[:SHOT]].mean(0) for c in classes}
            pairs = sorted(((np.linalg.norm(mu[a] - mu[b]), a, b) for i, a in enumerate(classes)
                            for b in classes[i + 1:]))[:spec["pairs"]]
            order_ds = []
            for k, (_, a, b) in enumerate(pairs):
                name = f"{ds}_{k}"
                parts = {"fit": [], "feedback": [], "test": []}
                for lab, c in ((0, a), (1, b)):
                    idx = rng.permutation(np.where(y == c)[0])
                    n_fb = min(FB, (len(idx) - SHOT) // 3)
                    for part, sl in (("fit", idx[:SHOT]), ("feedback", idx[SHOT:SHOT + n_fb]),
                                     ("test", idx[SHOT + n_fb:])):
                        parts[part] += [(i, lab) for i in sl]
                td = self.tasks / name
                td.mkdir(exist_ok=True)
                for part, rows in parts.items():
                    ii = np.array([i for i, _ in rows])
                    yy = np.array([lab for _, lab in rows])
                    np.savez_compressed(td / f"{part}.npz", X=X[ii], y=yy)
                w, b0 = fit_head(X[[i for i, _ in parts["fit"]]],
                                 np.array([lab for _, lab in parts["fit"]]))
                if spec["skip_easy"]:
                    fbi = np.array([i for i, _ in parts["feedback"]])
                    fby = np.array([lab for _, lab in parts["feedback"]])
                    easy = np.mean(((X[fbi] @ w + b0) > 0) == (fby == 1)) >= 0.95
                    if easy or len(order_ds) >= spec["max_per_dataset"]:
                        continue
                order_ds.append(name)
                np.savez_compressed(td / "start_head.npz", coef=w, intercept=np.float64(b0))
                meta = {"task": name, "dataset": ds, "classes": [int(a), int(b)],
                        "n": {p: len(r) for p, r in parts.items()}}
                (td / "meta.json").write_text(json.dumps(meta))
                order.append(name)
                self.log(f'build: {name} classes {a} vs {b} n={meta["n"]}')
        # interleave datasets: task 0 of every dataset, then task 1, ...
        by = {}
        for t in order:
            by.setdefault(t.rsplit("_", 1)[0], []).append(t)
        order = [ts[r] for r in range(spec["rounds"]) for ts in by.values() if r < len(ts)]
        (self.root / "order.json").write_text(json.dumps(order))

    def research(self):
        rd = self.root / "research"
        rd.mkdir(exist_ok=True)
        self.agent(rd, self.prompt("research"), ["WebSearch", "WebFetch", "Write", "Read"], 30)
        self.log("research done" if (rd / "PLAYBOOK.md").exists() else "research: no PLAYBOOK.md")

    def run(self, arm):
        order = json.loads((self.root / "order.json").read_text())
        ad = self.root / f"arm_{arm}"
        ad.mkdir(exist_ok=True)
        lessons = ad / "LESSONS.md"
        if arm == "lessons" and not lessons.exists():
            lessons.write_text("# Lessons from earlier tasks (test results you could not see at the time)\n")
        res_file = ad / "results.jsonl"
        done = {json.loads(line)["task"] for line in open(res_file)} if res_file.exists() else set()
        brief = (PROMPTS / f"brief_{self.version}.md").read_text()
        v2 = self.version == "v2"
        for i, task in enumerate(order):
            if task in done:
                continue
            td, wd = self.tasks / task, ad / task
            if wd.exists():
                shutil.rmtree(wd)
            wd.mkdir()
            meta = json.loads((td / "meta.json").read_text())
            for f in ("fit.npz", "feedback.npz", "start_head.npz"):
                shutil.copy(td / f, wd / f)
            shutil.copy(td / "start_head.npz", wd / "candidate.npz")
            if (self.root / "research" / "PLAYBOOK.md").exists():
                shutil.copy(self.root / "research" / "PLAYBOOK.md", wd / "PLAYBOOK.md")
            if v2 and (ad / "METHODS.md").exists():
                shutil.copy(ad / "METHODS.md", wd / "METHODS.md")
            shutil.copy(HERE / "score_tool.py", wd / "score.py")
            ll = "."
            if arm == "lessons":
                shutil.copy(lessons, wd / "LESSONS.md")
                ll = ", then LESSONS.md: what earlier edits did on unseen test images. Use those lessons."
            (wd / "BRIEF.md").write_text(brief.format(
                task=task, dataset=meta["dataset"], a=meta["classes"][0], b=meta["classes"][1],
                budget=BUDGET, lessons_line=ll))
            self.log(f"{arm} task {i + 1}/{len(order)} {task}: agent start")
            self.agent(wd, "Read BRIEF.md in this folder and do the task. Stay inside this folder.",
                       ["Bash", "Read", "Write", "Edit"], 40)

            c = np.load(td / "start_head.npz")
            heads = {"start": (c["coef"], float(c["intercept"]))}
            if (wd / "final.npz").exists():
                f = np.load(wd / "final.npz")
                heads["agent"] = (f["coef"].astype(np.float64), float(f["intercept"]))
            heads["placebo"] = placebo(td)
            fx, fb = np.load(td / "fit.npz"), np.load(td / "feedback.npz")
            pool_X, pool_y = np.vstack([fx["X"], fb["X"]]), np.concatenate([fx["y"], fb["y"]])
            heads["trained_fit_plus_feedback"] = fit_head(pool_X, pool_y)
            if v2:
                heads["trained_tuned"] = fit_head(pool_X, pool_y, tune=True)
            sc = test_scores(td, heads)
            calls = sum(1 for _ in open(wd / "calls.jsonl")) if (wd / "calls.jsonl").exists() else 0
            row = {"arm": arm, "i": i, "task": task, "calls": calls, "test": sc}
            with open(res_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            self.log(f"{arm} {task}: {json.dumps(sc)}")
            if arm == "lessons" and "agent" in sc:
                (wd / "TEST_RESULT.json").write_text(json.dumps(sc, indent=1))
                self.agent(wd, self.prompt("lesson"), ["Read", "Edit", "Write"], 10)
                shutil.copy(wd / "LESSONS.md", lessons)
            if v2 and (wd / "METHODS.md").exists():
                shutil.copy(wd / "METHODS.md", ad / "METHODS.md")
        self.log(f"{arm}: ALL TASKS DONE")


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def fit_head(X, y, tune=False):
    if tune:
        from sklearn.linear_model import LogisticRegressionCV
        c = LogisticRegressionCV(Cs=10, cv=5, max_iter=4000, class_weight="balanced").fit(X, y)
    else:
        c = LogisticRegression(C=1.0, max_iter=4000, class_weight="balanced").fit(X, y)
    return c.coef_.ravel().astype(np.float64), float(c.intercept_[0])


def placebo(td):
    """Random edits with the agent's call budget; a draw is kept if feedback accuracy rises."""
    v = np.load(td / "feedback.npz")
    c = np.load(td / "start_head.npz")
    w, b = c["coef"].copy(), float(c["intercept"])
    X, y = v["X"], v["y"]
    best = np.mean(((X @ w + b) > 0) == (y == 1))
    rng = np.random.default_rng(7)
    for _ in range(BUDGET):
        w2 = w.copy()
        k = rng.choice(len(w), 40, replace=False)
        w2[k] *= np.exp(rng.normal(0, .5, 40))
        w2 += rng.normal(0, 1, len(w)) * np.linalg.norm(w) / np.sqrt(len(w)) * .1
        a = np.mean(((X @ w2 + b) > 0) == (y == 1))
        if a > best:
            best, w = a, w2
    return w, b


def test_scores(td, heads):
    t = np.load(td / "test.npz")
    out = {}
    for name, (w, b) in heads.items():
        s = t["X"] @ w + b
        out[name] = {"acc": round(float(np.mean((s > 0) == (t["y"] == 1))), 4),
                     "auc": round(auc(s, t["y"]), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "research", "run"])
    ap.add_argument("arm", nargs="?", choices=["lessons", "fresh"])
    ap.add_argument("--version", choices=sorted(VERSIONS), required=True)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--features", help="folder with <dataset>.npz from extract_features.py (build)")
    ap.add_argument("--no-agent", action="store_true", help="skip every agent call")
    a = ap.parse_args()

    suite = Suite(a.workdir, a.version, a.no_agent)
    if a.command == "build":
        if not a.features:
            ap.error("build needs --features")
        suite.build(a.features)
    elif a.command == "research":
        suite.research()
    else:
        if not a.arm:
            ap.error("run needs an arm")
        suite.run(a.arm)


if __name__ == "__main__":
    sys.exit(main())
