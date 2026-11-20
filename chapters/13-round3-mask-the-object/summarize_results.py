"""Condense the Round 3 attempt files into one small summary.json.

Reads <run-dir>/attempts[_dinov2]/<slot>_k<k>.json (written by ``run_r3.py attempt``) and keeps, per
arm, slot and k: the status, the worst-seed value and verdict of every gate, the feasibility margin,
no-op integrity, and per-seed marker flip rates for A, the candidate and the matched retrainers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ARMS = {"siglip": "attempts", "dinov2": "attempts_dinov2"}


def summarize_attempt(d: dict) -> dict:
    rep = d["report"]
    rule = rep["binding_rule"]
    rr = rep["rules"][rule]
    (cand, c), = rr["candidates"].items()
    seeds = {}
    for seed, s in d["per_seed"].items():
        r = s["rules"][rule]
        seeds[seed] = dict(
            A_marker_flip_rate=r["A"]["marker_flip_rate"],
            candidate_marker_flip_rate=r["arms"][cand]["metrics"]["marker_flip_rate"],
            marker_flip_rate_by_arm={n: a["metrics"]["marker_flip_rate"] for n, a in r["arms"].items()},
        )
    return dict(
        status=rep["status"],
        candidate=cand,
        gate_verdict_worst_seed=c["gates"]["bars"],
        gate_value_worst_seed=c["gates"]["worst"],
        gates_passed=sum(v == "PASS" for v in c["gates"]["bars"].values()),
        gates_total=len(c["gates"]["bars"]),
        feasibility_all_seeds=c["feasibility_all_seeds"],
        feasibility_worst_recall_margin=c["feasibility_worst_recall_margin"],
        noop_integrity=rr["noop_integrity"],
        per_seed=seeds,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", default="results/summary.json", type=Path)
    a = ap.parse_args()
    plan = json.loads((a.run_dir / "R3_PLAN.json").read_text())
    out = dict(
        note="Derived from the attempt files of the Round 3 run (development data only, exploratory).",
        plan=dict(seeds=plan["seeds"], ks=plan["ks"], primary_k=plan["primary_k"], alpha=plan["alpha"],
                  binding_rule=plan["binding_rule"], slot1_placebo=plan["slot1_placebo"]),
        arms={},
    )
    for arm, folder in ARMS.items():
        att = {}
        for p in sorted((a.run_dir / folder).glob("slot*_k*.json")):
            d = json.loads(p.read_text())
            att[f"{d['report']['slot']}_k{d['report']['k']}"] = summarize_attempt(d)
        out["arms"][arm] = att
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {a.out} ({sum(len(v) for v in out['arms'].values())} attempts)")


if __name__ == "__main__":
    main()
