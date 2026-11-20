"""Task-required L2 checks: actually run vs merely have the feature.

LOCO has explicit structural and logical anomalies.  The v3.1 task policy
requires an L2 check, but the prior CPU experiment counted feature availability
(l2_checked=592) without ever producing an L2 verdict.  This script builds
calibrated L2 verdicts, charges their observed CPU cost, and compares:
  * L1 only
  * L2 only
  * L1+L2 fusion (release only if both normal; hold if either anomalous)
  * a gate that runs L2 only when L1 has not already held/released
"""
from __future__ import annotations

import itertools
import json

import numpy as np
from common import (
    auc,
    average_precision,
    bootstrap_ci,
    clopper_pearson,
    write_csv,
    write_json,
)
from features import attach_features
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def conformal(score, ref):
    ref = np.sort(np.asarray(ref, float))
    return (len(ref) - np.searchsorted(ref, score, side="left") + 1.0) / (len(ref) + 1.0)


def pvalues(rows, score_key):
    ref = {}
    for r in rows:
        if r["role"] == "calibration":
            ref.setdefault(r["cal_group"], []).append(float(r[score_key]))
    out = {}
    for r in rows:
        p = conformal(float(r[score_key]), ref.get(r["cal_group"], []))
        out[r["sha256"]] = p
    return out


def fit_l2(rows, keys):
    tr = [r for r in rows if r["role"] == "router_train"]
    X = np.array([[r[k] for k in keys] for r in tr], float)
    y = np.array([r["truth_bin"] for r in tr])
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5))
    m.fit(X, y)
    out = {}
    for r in rows:
        x = np.array([[r[k] for k in keys] for r in [r]], float)
        out[r["sha256"]] = float(m.predict_proba(x)[0, 1])
    return out


def decisions(rows, p1, p2, ar, ah, fusion, cost_key="l2_sem_cpu"):
    recs = []
    for r in rows:
        a, b = p1[r["sha256"]], p2[r["sha256"]]
        run_l2 = True
        if fusion == "l1":
            b = None
            run_l2 = False
            decision = "hold" if a < ah else ("release" if a > ar else "human")
        elif fusion == "l2":
            a = None
            decision = "hold" if b < ah else ("release" if b > ar else "human")
        elif fusion == "and_release_or_hold":
            decision = "hold" if (a < ah or b < ah) else (
                "release" if (a > ar and b > ar) else "human")
        elif fusion == "l2_gate":
            # L1 strongly anomalous -> hold without L2; otherwise L2 must run
            # before any release or hold.  Release requires both normal.
            if a < ah:
                decision = "hold"
                run_l2 = False
            else:
                run_l2 = True
                decision = "hold" if b < ah else (
                    "release" if (a > ar and b > ar) else "human")
        elif fusion == "l2_gate_release":
            # Mandatory only before release: hold on L1 alone, release on L1+L2.
            if a < ah:
                decision = "hold"
                run_l2 = False
            elif a > ar:
                run_l2 = True
                decision = "release" if b > ar else "human"
            else:
                decision = "human"
                run_l2 = False
        else:
            raise ValueError(fusion)
        cpu = r["l1_cpu"] + (r[cost_key] if run_l2 and r[cost_key] is not None else 0.0)
        recs.append({"row": r, "decision": decision, "run_l2": run_l2, "cpu": cpu,
                     "p1": a, "p2": b})
    return recs


def summarize(recs, ar, ah, fusion):
    c = {"release": 0, "hold": 0, "human": 0, "wrong_release": 0, "false_hold": 0,
         "true_release": 0, "anomalies_held": 0, "anomalies": 0, "good": 0,
         "l2_runs": 0, "cpu_total": 0.0}
    for x in recs:
        r = x["row"]
        truth = r["truth_bin"]
        c["anomalies" if truth else "good"] += 1
        c["l2_runs"] += int(x["run_l2"])
        c["cpu_total"] += x["cpu"]
        if x["decision"] == "release":
            c["release"] += 1
            if truth:
                c["wrong_release"] += 1
            else:
                c["true_release"] += 1
        elif x["decision"] == "hold":
            c["hold"] += 1
            if truth:
                c["anomalies_held"] += 1
            else:
                c["false_hold"] += 1
        else:
            c["human"] += 1
    n = len(recs)
    c.update(family="mvtec_loco", fusion=fusion, alpha_release=ar, alpha_hold=ah,
             n=n, human_rate=c["human"] / n, autonomous_coverage=(c["release"] + c["hold"]) / n,
             wrong_release_rate=c["wrong_release"] / c["anomalies"] if c["anomalies"] else None,
             false_hold_rate=c["false_hold"] / c["good"] if c["good"] else None,
             wrong_release_upper95=clopper_pearson(c["wrong_release"], c["anomalies"])[1],
             l2_run_rate=c["l2_runs"] / n, mean_cpu=c["cpu_total"] / n)
    return c


def main():
    rows, _ = attach_features(include_heldout=False)
    rows = [r for r in [x for x in rows if x["dataset"] == "mvtec_loco"]]
    dev = [r for r in rows if r["role"] == "dev_audit"]
    p1 = pvalues(rows, "z_patch")
    p2_sem = pvalues(rows, "z_sem")
    # Full L2 uses a supervised score over all available L2 evidence.
    full_score = fit_l2(rows, ["z_sem", "z_fused", "z_aclip"])
    for r in rows:
        r["l2_full_score"] = full_score[r["sha256"]]
    p2_full = pvalues(rows, "l2_full_score")

    grid = []
    for p2, label, cost_key in [(p2_sem, "l2_semantic", "l2_sem_cpu"),
                                (p2_full, "l2_full", "l2_full_cpu")]:
        for fusion in ["l1", "l2", "and_release_or_hold", "l2_gate", "l2_gate_release"]:
            for ar, ah in itertools.product([0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                                            [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]):
                if ah >= ar:
                    continue
                recs = decisions(dev, p1, p2, ar, ah, fusion, cost_key=cost_key)
                c = summarize(recs, ar, ah, f"{label}:{fusion}")
                grid.append(c)
    write_csv("l2_policy_grid.csv", grid)

    # Complementarity at a fixed alarm level: p<0.05 means the specialist flags anomaly.
    comp = {}
    for p2, label, cost_key in [(p2_sem, "l2_semantic", "l2_sem_cpu"),
                                (p2_full, "l2_full", "l2_full_cpu")]:
        for stratum in ["all", "structural_anomalies", "logical_anomalies", "good"]:
            sub = [r for r in dev if stratum == "all" or r["stratum"] == stratum]
            if not sub:
                continue
            l1_flag = np.array([p1[r["sha256"]] < 0.05 for r in sub])
            l2_flag = np.array([p2[r["sha256"]] < 0.05 for r in sub])
            y = np.array([r["truth_bin"] for r in sub])
            comp[f"{label}:{stratum}"] = {
                "n": len(sub), "anomalies": int(y.sum()),
                "l1_flag_rate": float(l1_flag.mean()), "l2_flag_rate": float(l2_flag.mean()),
                "l1_only": int((l1_flag & ~l2_flag).sum()),
                "l2_only": int((~l1_flag & l2_flag).sum()),
                "both": int((l1_flag & l2_flag).sum()),
                "neither": int((~l1_flag & ~l2_flag).sum()),
                "anomalies_missed_by_l1_caught_by_l2": int((y & ~l1_flag & l2_flag).sum()),
                "anomalies_missed_by_l2_caught_by_l1": int((y & l1_flag & ~l2_flag).sum()),
                "anomalies_missed_by_both": int((y & ~l1_flag & ~l2_flag).sum()),
            }
    # AUC and bootstrap CIs for the three scores.
    auc_rows = []
    y = np.array([r["truth_bin"] for r in dev])
    groups = np.array([r["category"] for r in dev])
    for name, score_map in [("l1_patch", p1), ("l2_semantic", p2_sem), ("l2_full", p2_full)]:
        s = np.array([-score_map[r["sha256"]] for r in dev])
        auc_rows.append({"method": name, "auc": auc(y, s),
                         "average_precision": average_precision(y, s),
                         "auc_ci95": bootstrap_ci(y, s, metric="auc", n_boot=1000,
                                                  seed=11, groups=groups)})
    # fusion score = min p (either specialist flags anomaly)
    s = np.array([-min(p1[r["sha256"]], p2_sem[r["sha256"]]) for r in dev])
    auc_rows.append({"method": "l1_or_l2_semantic", "auc": auc(y, s),
                     "average_precision": average_precision(y, s),
                     "auc_ci95": bootstrap_ci(y, s, metric="auc", n_boot=1000,
                                              seed=12, groups=groups)})
    s = np.array([-min(p1[r["sha256"]], p2_full[r["sha256"]]) for r in dev])
    auc_rows.append({"method": "l1_or_l2_full", "auc": auc(y, s),
                     "average_precision": average_precision(y, s),
                     "auc_ci95": bootstrap_ci(y, s, metric="auc", n_boot=1000,
                                              seed=13, groups=groups)})
    write_csv("l2_auc.csv", auc_rows)
    write_json("l2_complementarity.json", comp)

    # Operating points on the L2 policy grid.
    op = []
    for label in sorted({r["fusion"] for r in grid}):
        cand = [r for r in grid if r["fusion"] == label]
        for cname, fn in {
            "zero_wrong_release": lambda r: r["wrong_release"] == 0,
            "upper95_wr_le_1pct": lambda r: r["wrong_release_upper95"] <= 0.01,
            "wr_rate_le_5pct": lambda r: r["wrong_release_rate"] <= 0.05,
        }.items():
            ok = [r for r in cand if fn(r)]
            if not ok:
                continue
            best = max(ok, key=lambda r: (r["autonomous_coverage"], -r["human_rate"],
                                          -r["false_hold"]))
            op.append({**best, "constraint": cname})
    write_csv("l2_operating_points.csv", op)
    l1_costs = [r["l1_cpu"] for r in dev]
    l2_costs = [r["l2_sem_cpu"] for r in dev if r["l2_sem_cpu"] is not None]
    l2_full_costs = [r["l2_full_cpu"] for r in dev if r["l2_full_cpu"] is not None]
    write_json("l2_metadata.json", {
        "cost_medians_cpu_s": {
            "l1": float(np.median(l1_costs)),
            "l2_semantic_incremental": float(np.median(l2_costs)),
            "l2_full_incremental": float(np.median(l2_full_costs)),
            "l1_plus_l2_semantic": float(np.median(l1_costs) + np.median(l2_costs)),
        },
        "scope": "MVTec LOCO dev_audit (n=592, 259 anomalies, 333 good)",
        "l1": "per-category robust-z PatchCore score; conformal p-value from GOOD calibration",
        "l2_semantic": "per-category robust-z semantic representation score; conformal p-value",
        "l2_full": "logistic over semantic/fused/AnomalyCLIP z-scores, trained on router_train; "
                    "conformal p-value from GOOD calibration",
        "cost": "recorded CPU from prior runs; l1 median and l2 semantic/full medians; "
                "not a new end-to-end latency measurement",
        "unresolved": "HUMAN is not counted as autonomous or correct",
        "prior_issue": "previous task_policy_learning_results counted 592 l2_checked cases even though "
                       "no L2 verdict existed; here l2_run_rate is the actual executed fraction",
    })
    print(json.dumps({"grid": len(grid), "auc": auc_rows, "operating_points": len(op)},
                     indent=2))


if __name__ == "__main__":
    main()
