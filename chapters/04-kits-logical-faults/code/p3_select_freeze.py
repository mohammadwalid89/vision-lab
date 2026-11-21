#!/usr/bin/env python3
"""Phase 3 preregistered dev-only selection. Never imports/opens locked final records.
Frozen policies cannot be changed after P3_FINAL_OPENED is set.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from p3_models import CATS, fit_gate, gate_accept, manifest, save_json_atomic, sha
from scipy.stats import beta

D = Path(__file__).resolve().parent
OUT = D / "out"
M = D / "models"
COST = {"defect_prevalence": 0.02, "human_review_usd": 0.25, "false_hold_usd": 2.0, "escape_usd": 50.0}
MODELS = ["component", "relational", "resnet", "wrn"]
CANDIDATES = {
    "component": ["component"],
    "relational": ["relational"],
    "resnet": ["resnet"],
    "wrn": ["wrn"],
    "component_wrn": ["component", "wrn"],
    "relational_wrn": ["relational", "wrn"],
    "component_resnet": ["component", "resnet"],
    "relational_resnet": ["relational", "resnet"],
    "wrn_resnet": ["wrn", "resnet"],
    "all_four": ["component", "relational", "resnet", "wrn"],
}
HOLD_GRID = [0.0, 0.01, 0.02, 0.04, 0.08, 0.12, 0.20]


def cp_upper(x, n, alpha=0.05):
    return float(beta.ppf(1 - alpha, x + 1, n - x)) if x < n else 1.0


def cp_two_sided(x, n):
    return [float(beta.ppf(0.025, x, n - x + 1)) if x else 0.0, float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0]


def outcomes(y, typ, dec):
    g = y == 0
    a = ~g
    ng = int(g.sum())
    na = int(a.sum())
    gr = int(np.count_nonzero(g & (dec == "release")))
    esc = int(np.count_nonzero(a & (dec == "release")))
    fh = int(np.count_nonzero(g & (dec == "hold")))
    gd = int(np.count_nonzero(g & (dec == "human")))
    ad = int(np.count_nonzero(a & (dec == "human")))
    held = int(np.count_nonzero(a & (dec == "hold")))
    pi = COST["defect_prevalence"]
    ch = COST["human_review_usd"]
    cf = COST["false_hold_usd"]
    ce = COST["escape_usd"]
    er = esc / max(na, 1)
    fhr = fh / max(ng, 1)
    gdr = gd / max(ng, 1)
    adr = ad / max(na, 1)
    return {
        "n_good": ng,
        "n_defect": na,
        "good_released": gr,
        "good_release_rate": gr / max(ng, 1),
        "good_release_ci95": cp_two_sided(gr, ng) if ng else None,
        "escapes": esc,
        "escape_rate": er,
        "escape_ucb95": cp_upper(esc, na) if na else None,
        "escapes_by_stratum": {
            t: int(np.count_nonzero((typ == t) & (dec == "release")))
            for t in ["logical_anomalies", "structural_anomalies"]
        },
        "false_holds": fh,
        "good_to_person": gd,
        "defects_to_person": ad,
        "defects_held": held,
        "person_reviews_per_1000": 1000 * ((1 - pi) * gdr + pi * adr),
        "good_releases_per_1000_good": 1000 * gr / max(ng, 1),
        "usd_per_1000": 1000 * ((1 - pi) * (fhr * cf + gdr * ch) + pi * (er * ce + adr * ch)),
    }


def read_data():
    assert (OUT / "P3_DEV_MODELS_COMPLETE").exists(), "Model scoring incomplete"
    assert not (OUT / "P3_FINAL_OPENED").exists(), "Final already opened; no refreeze"
    assert not (OUT / "p3_frozen_policy.json").exists(), "Policy already frozen"
    data = {}
    files = {}
    for cat in CATS:
        v = manifest("validation", cat)
        d = manifest("dev", cat)
        gates = fit_gate(manifest("train", cat), v)
        gate = np.array([gate_accept(r["path"], gates) for r in d], bool)
        y = np.array([int(r["stratum"] != "good") for r in d])
        typ = np.array([r["stratum"] for r in d])
        out = {"rows": d, "gate": gate, "y": y, "typ": typ, "gate_params": gates, "p": {}, "latency": {}}
        for model in ["component_relational", "resnet", "wrn"]:
            file = OUT / f"p3_{model}_{cat}_dev.json"
            z = json.loads(file.read_text())
            files[file.name] = sha(file)
            expected_val = {r["sha256"] for r in v}
            expected_dev = {r["sha256"] for r in d}
            assert {r["sha256"] for r in z["validation"]} == expected_val
            assert {r["sha256"] for r in z["development"]} == expected_dev
            assert len(z["validation"]) == len(expected_val) and len(z["development"]) == len(expected_dev)
            for k in ["component", "relational"] if model == "component_relational" else [model]:
                val = np.sort(
                    np.array([r[k] if k in ("component", "relational") else r["score"] for r in z["validation"]], float)
                )
                scored = {r["sha256"]: r for r in z["development"]}
                vals = np.array(
                    [
                        scored[r["sha256"]][k] if k in ("component", "relational") else scored[r["sha256"]]["score"]
                        for r in d
                    ]
                )
                out["p"][k] = (len(val) - np.searchsorted(val, vals, side="left") + 1) / (len(val) + 1)
                out["latency"][k] = np.array([scored[r["sha256"]]["latency_ms"] for r in d])
        data[cat] = out
    return data, files


def all_rows(data, components):
    rows = []
    for cat in CATS:
        d = data[cat]
        p = np.min(np.stack([d["p"][k] for k in components]), axis=0)
        for i, r in enumerate(d["rows"]):
            rows.append(
                {
                    "cat": cat,
                    "sha": r["sha256"],
                    "stratum": r["stratum"],
                    "y": int(d["y"][i]),
                    "p": float(p[i]),
                    "gate": bool(d["gate"][i]),
                }
            )
    return rows


def policy_on(rows, ids):
    # Category-specific release threshold is the greatest p-value of gated dev defects,
    # which makes escape impossible on the selection rows (strict > threshold).
    ar = {}
    for cat in CATS:
        sub = [rows[i] for i in ids if rows[i]["cat"] == cat]
        bad = [r["p"] for r in sub if r["y"] and r["gate"]]
        ar[cat] = max(bad) if bad else 0.0
    best = None
    for ah in HOLD_GRID:
        d = decisions(rows, {"ar": ar, "ah": ah})
        r = outcomes(np.array([rows[i]["y"] for i in ids]), np.array([rows[i]["stratum"] for i in ids]), d[ids])
        key = (r["escapes"] > 0, r["usd_per_1000"], ah)
        if best is None or key < best[0]:
            best = (key, {"ar": ar, "ah": ah})
    return best[1]


def decisions(rows, pol):
    d = np.full(len(rows), "human", dtype="<U8")
    for i, r in enumerate(rows):
        if not r["gate"]:
            continue
        if r["p"] > pol["ar"][r["cat"]]:
            d[i] = "release"
        elif r["p"] <= pol["ah"]:
            d[i] = "hold"
    return d


def splits(rows, n=100, seed=20261028):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[(r["cat"], r["stratum"])].append(i)
    for _ in range(n):
        a = []
        for ids in groups.values():
            p = rng.permutation(ids)
            a.extend(p[: len(p) // 2])
        aa = np.array(sorted(a), dtype=int)
        bb = np.setdiff1d(np.arange(len(rows)), aa)
        yield aa, bb


def evaluate(rows):
    ys = np.array([r["y"] for r in rows])
    ts = np.array([r["stratum"] for r in rows])
    stats = []
    for a, b in splits(rows):
        pol = policy_on(rows, a)
        o = outcomes(ys[b], ts[b], decisions(rows, pol)[b])
        stats.append(o)
    finalpol = policy_on(rows, np.arange(len(rows)))
    full = outcomes(ys, ts, decisions(rows, finalpol))
    assert full["escapes"] == 0
    return {
        "oos_mean_cost": float(np.mean([x["usd_per_1000"] for x in stats])),
        "oos_mean_escape_rate": float(np.mean([x["escape_rate"] for x in stats])),
        "oos_any_escape_fraction": float(np.mean([x["escapes"] > 0 for x in stats])),
        "oos_mean_good_release": float(np.mean([x["good_release_rate"] for x in stats])),
        "oos_good_release_p05_p95": list(np.quantile([x["good_release_rate"] for x in stats], [0.05, 0.95])),
        "dev": full,
        "policy": finalpol,
        "oos_repetitions": len(stats),
    }


def main():
    data, files = read_data()
    variants = {}
    for name, components in CANDIDATES.items():
        rows = all_rows(data, components)
        variants[name] = evaluate(rows)
        x = variants[name]
        print(
            "DEV",
            name,
            "release",
            round(x["dev"]["good_release_rate"], 3),
            "oos escape",
            round(x["oos_mean_escape_rate"], 3),
            "oos cost",
            round(x["oos_mean_cost"], 2),
            flush=True,
        )
    acceptable = [n for n, v in variants.items() if v["oos_mean_escape_rate"] <= 0.01]
    # Precommitted fallback is refusal (all human) if no candidate has adequate dev OOS escape.
    winner = (
        min(acceptable, key=lambda n: (variants[n]["oos_mean_cost"], -variants[n]["oos_mean_good_release"], n))
        if acceptable
        else None
    )
    models = {}
    for cat in CATS:
        for model in ["component_relational", "resnet", "wrn"]:
            p = M / f"{model}_{cat}.npz" if model == "component_relational" else M / f"{model}_{cat}.pt"
            models[p.name] = sha(p)
    result = {
        "protocol": "P3 train-only fit, validation-normal calibration, deterministic 40% exposed dev selection; final never opened",
        "winner": winner,
        "fallback": None if winner else "all-human/no-release",
        "candidates": {k: list(v) for k, v in CANDIDATES.items()},
        "hold_grid": HOLD_GRID,
        "selection_rule": "lowest 100-half-split OOS cost among mean OOS escape<=1%; full-dev zero-escape release threshold",
        "variants": variants,
        "gate": {c: data[c]["gate_params"] for c in CATS},
        "dev_gate_acceptance": {
            c: {
                "good": int(np.count_nonzero(data[c]["gate"] & (data[c]["y"] == 0))),
                "defect": int(np.count_nonzero(data[c]["gate"] & (data[c]["y"] == 1))),
            }
            for c in CATS
        },
        "model_sha256": models,
        "dev_score_sha256": files,
        "selector_code_sha256": sha(__file__),
        "split_manifest_sha256": sha(OUT / "p3_split_manifest.json"),
        "locked_final_manifest_sha256": (OUT / "p3_locked_final.sha256").read_text().split()[0],
        "cost_assumptions": COST,
        "frozen_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    save_json_atomic(OUT / "p3_frozen_policy.json", result)
    (OUT / "p3_frozen_policy.sha256").write_text(sha(OUT / "p3_frozen_policy.json") + "  p3_frozen_policy.json\n")
    print("FROZEN", winner, "hash", sha(OUT / "p3_frozen_policy.json"), flush=True)


if __name__ == "__main__":
    main()
