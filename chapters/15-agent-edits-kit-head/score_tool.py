"""Forward-pass tool given to the agent (copied into its sandbox as score.py).

Scores a candidate head (coef, intercept) on edit_val.npz. Budget: 50 calls, each
logged to calls.jsonl. Prints AUC, recall at a 1% false-alarm rate on calibration
normals, the false-alarm rate on good kits, and the validation images it gets wrong.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BUDGET = 50
D = Path(__file__).resolve().parent
log = D / "calls.jsonl"
n = sum(1 for _ in open(log)) if log.exists() else 0
if n >= BUDGET:
    sys.exit(f"budget spent: {BUDGET} of {BUDGET} scoring calls used")
c = np.load(D / (sys.argv[1] if len(sys.argv) > 1 else "candidate.npz"))
v = np.load(D / "edit_val.npz")
s = v["X"].astype(np.float64) @ c["coef"] + float(c["intercept"])
y = v["y"]
pos, neg = s[y == 1], s[y == 0]
auc = float(((pos[:, None] > neg[None, :]).sum() + .5 * (pos[:, None] == neg[None, :]).sum())
            / (len(pos) * len(neg)))
normals = s[(y == 0) & (v["roles"] == "calib_normal")]
thr = float(np.quantile(normals, .99))
rec = float(np.mean(pos > thr))
fa = float(np.mean(s[(y == 0) & (v["roles"] == "eval_good")] > thr))
wrong = [{"i": int(k), "id": str(v["ids"][k]), "label": int(y[k]), "score": round(float(s[k]), 3)}
         for k in np.argsort(-np.abs(s - thr)) if (s[k] > thr) != (y[k] == 1)]
out = {"call": n + 1, "auc": round(auc, 4), "recall_at_1pct": round(rec, 4),
       "false_alarm_good": round(fa, 4), "threshold": round(thr, 4), "wrong": wrong}
with open(log, "a") as f:
    f.write(json.dumps({"t": time.time(), **{k: out[k] for k in
            ("call", "auc", "recall_at_1pct", "false_alarm_good")}}) + "\n")
print(json.dumps(out))
