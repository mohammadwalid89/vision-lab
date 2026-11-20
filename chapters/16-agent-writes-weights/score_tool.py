"""Forward-pass tool given to the agent (copied into its task folder as score.py).

Scores a candidate head (coef, intercept) on feedback.npz, logs each call to calls.jsonl
and refuses after BUDGET calls (keep in sync with BUDGET in suite.py).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BUDGET = 30
D = Path(__file__).resolve().parent
log = D / "calls.jsonl"
n = sum(1 for _ in open(log)) if log.exists() else 0
if n >= BUDGET:
    sys.exit("budget spent")
c = np.load(D / (sys.argv[1] if len(sys.argv) > 1 else "candidate.npz"))
v = np.load(D / "feedback.npz")
s = v["X"] @ c["coef"] + float(c["intercept"])
y = v["y"]
p, q = s[y == 1], s[y == 0]
auc = float(((p[:, None] > q[None, :]).sum() + .5 * (p[:, None] == q[None, :]).sum()) / (len(p) * len(q)))
acc = float(np.mean((s > 0) == (y == 1)))
wrong = [{"i": int(k), "label": int(y[k]), "score": round(float(s[k]), 3)}
         for k in np.argsort(-np.abs(s)) if (s[k] > 0) != (y[k] == 1)]
out = {"call": n + 1, "auc": round(auc, 4), "accuracy": round(acc, 4), "wrong": wrong[:40]}
with open(log, "a") as f:
    f.write(json.dumps({"t": time.time(), "call": n + 1, "auc": out["auc"], "accuracy": out["accuracy"]}) + "\n")
print(json.dumps(out))
