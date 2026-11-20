"""Aggregate suite results (results.jsonl files written by suite.py) into the table in README.md.

For each file: mean accuracy gain of the agent's head over the start head and over the
comparison heads, in percentage points, and wins / ties / losses against each.
"""
import argparse
import json
from pathlib import Path

import numpy as np

COMPARE = ["placebo", "trained_fit_plus_feedback", "trained_tuned"]


def summarize(path, exclude):
    rows = [json.loads(line) for line in open(path)]
    rows = [r for r in rows if r["task"].split("_")[0] not in exclude and "agent" in r["test"]]
    acc = {k: np.array([r["test"][k]["acc"] for r in rows])
           for k in ["start", "agent", *COMPARE] if k in rows[0]["test"]}
    out = {"n_tasks": len(rows), "tasks": [r["task"] for r in rows],
           "scoring_calls_per_task": [r["calls"] for r in rows],
           "agent_minus_start_points": round(100 * float((acc["agent"] - acc["start"]).mean()), 2),
           "agent_beats_start_on": int((acc["agent"] > acc["start"]).sum())}
    for k in COMPARE:
        if k in acc:
            d = 100 * (acc["agent"] - acc[k])
            out[f"agent_minus_{k}_points"] = round(float(d.mean()), 2)
            out[f"agent_vs_{k}_win_tie_loss"] = [int((d > 1e-9).sum()),
                                                  int((abs(d) <= 1e-9).sum()),
                                                  int((d < -1e-9).sum())]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="results.jsonl files, as name=path")
    ap.add_argument("--exclude", nargs="*", default=[], help="dataset names to leave out")
    ap.add_argument("--out")
    a = ap.parse_args()

    summary = {}
    for item in a.results:
        name, path = item.split("=", 1)
        summary[name] = summarize(path, set(a.exclude))
    text = json.dumps(summary, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
