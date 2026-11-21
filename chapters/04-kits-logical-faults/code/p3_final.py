#!/usr/bin/env python3
"""Single locked-final evaluation of all preregistered frozen Phase3 candidates.
Refuses to run if P3_FINAL_OPENED exists. No post-final policy adjustment.
"""

import json
import time

import numpy as np
import torch
from p3_models import CATS, OUT, D, Feat, M, comp_extract, embed, gate_accept, save_json_atomic, sha
from p3_select_freeze import CANDIDATES, MODELS, cp_upper, decisions, outcomes
from sklearn.neighbors import NearestNeighbors


def score_components(cat, rows):
    z = np.load(M / f"component_relational_{cat}.npz", allow_pickle=False)
    centers = z["centers"]
    cm = z["comp_med"]
    cs = z["comp_scale"]
    cz = z["comp_Z"]
    rm = z["rel_med"]
    rs = z["rel_scale"]
    rz = z["rel_Z"]
    nn = NearestNeighbors(n_neighbors=min(5, len(rz)), algorithm="brute").fit(rz)
    out = []
    for i, r in enumerate(rows):
        t = time.perf_counter()
        x, rel = comp_extract(r["path"], centers)
        a = np.clip((x - cm) / cs, -10, 10)
        d = np.sort(np.sqrt(((cz - a) ** 2).mean(1)))[:5].mean()
        zr = np.clip((rel - rm) / rs, -6, 6)
        dr = nn.kneighbors(zr[None], return_distance=True)[0][0].mean()
        out.append(
            {
                "sha256": r["sha256"],
                "component": float(d),
                "relational": float(dr),
                "latency_ms": float((time.perf_counter() - t) * 1000),
            }
        )
        if (i + 1) % 35 == 0:
            print("FINAL COMPONENT", cat, i + 1, "/", len(rows), flush=True)
    return out


def score_patch(which, cat, rows):
    net = Feat(which)
    saved = torch.load(M / f"{which}_{cat}.pt", map_location="cpu", weights_only=True)
    bank = saved["bank"].float()
    out = []
    for i, r in enumerate(rows):
        t = time.perf_counter()
        f = embed(net, [r["path"]], bs=1)[0].float()
        d = torch.cdist(f, bank).min(1).values
        out.append(
            {"sha256": r["sha256"], "score": float(d.max()), "latency_ms": float((time.perf_counter() - t) * 1000)}
        )
        if (i + 1) % 35 == 0:
            print("FINAL", which, cat, i + 1, "/", len(rows), flush=True)
    return out


def main():
    assert (OUT / "P3_DEV_MODELS_COMPLETE").exists()
    assert not (OUT / "P3_FINAL_OPENED").exists(), "Final already opened. Never rerun without explicit audit."
    assert not (OUT / "p3_final_results.json").exists()
    pol = json.loads((OUT / "p3_frozen_policy.json").read_text())
    assert sha(OUT / "p3_frozen_policy.json") == (OUT / "p3_frozen_policy.sha256").read_text().split()[0]
    assert sha(OUT / "p3_locked_final.json") == pol["locked_final_manifest_sha256"]
    assert sha(OUT / "p3_split_manifest.json") == pol["split_manifest_sha256"]
    assert sha(D / "p3_select_freeze.py") == pol["selector_code_sha256"]
    for name, expect in pol["model_sha256"].items():
        assert sha(M / name) == expect, name
    for name, expect in pol["dev_score_sha256"].items():
        assert sha(OUT / name) == expect, name
    assert set(pol["candidates"]) == set(CANDIDATES)
    assert pol["winner"] in CANDIDATES or pol["winner"] is None
    print("PRE-FINAL HASH AUDIT PASSED", time.strftime("%FT%TZ", time.gmtime()), flush=True)
    # Irreversible marker BEFORE reading any final records. Technical failure => no silent re-open.
    (OUT / "P3_FINAL_OPENED").write_text(
        "Final opened at "
        + time.strftime("%FT%TZ", time.gmtime())
        + "\nImmutable frozen policy sha256 "
        + sha(OUT / "p3_frozen_policy.json")
        + "\n"
    )
    rows = json.loads((OUT / "p3_locked_final.json").read_text())
    assert len(rows) == 357, len(rows)
    all_scores = {}
    for cat in CATS:
        rr = [r for r in rows if r["category"] == cat]
        print("FINAL CATEGORY", cat, "n", len(rr), flush=True)
        comp = score_components(cat, rr)
        wrn = score_patch("wrn", cat, rr)
        resnet = score_patch("resnet", cat, rr)
        for j, r in enumerate(rr):
            h = r["sha256"]
            assert comp[j]["sha256"] == wrn[j]["sha256"] == resnet[j]["sha256"] == h
            all_scores[h] = {
                "component": comp[j]["component"],
                "relational": comp[j]["relational"],
                "wrn": wrn[j]["score"],
                "resnet": resnet[j]["score"],
                "latency_ms": {
                    "component": comp[j]["latency_ms"],
                    "relational": comp[j]["latency_ms"],
                    "wrn": wrn[j]["latency_ms"],
                    "resnet": resnet[j]["latency_ms"],
                },
            }
        print("FINAL SCORED", cat, flush=True)
    # No final scores are exposed until all 357 have been scored.
    assert len(all_scores) == len(rows)
    vals = {}
    for cat in CATS:
        for model in ["component_relational", "wrn", "resnet"]:
            z = json.loads((OUT / f"p3_{model}_{cat}_dev.json").read_text())
            for k in ["component", "relational"] if model == "component_relational" else [model]:
                vals[(cat, k)] = np.sort(
                    [r[k] if k in ("component", "relational") else r["score"] for r in z["validation"]]
                )
    p = {}
    for r in rows:
        h = r["sha256"]
        cat = r["category"]
        p[h] = {}
        for model in MODELS:
            v = vals[(cat, model)]
            p[h][model] = float((len(v) - np.searchsorted(v, all_scores[h][model], side="left") + 1) / (len(v) + 1))
    final_variants = {}
    for name, components in CANDIDATES.items():
        rr = []
        for r in rows:
            rr.append(
                {
                    "cat": r["category"],
                    "sha": r["sha256"],
                    "stratum": r["stratum"],
                    "y": int(r["stratum"] != "good"),
                    "p": min(p[r["sha256"]][k] for k in components),
                    "gate": gate_accept(r["path"], pol["gate"][r["category"]]),
                }
            )
        decision = decisions(rr, pol["variants"][name]["policy"])
        y = np.array([r["y"] for r in rr])
        typ = np.array([r["stratum"] for r in rr])
        overall = outcomes(y, typ, decision)
        per_category = {}
        for cat in CATS:
            ix = np.array([i for i, r in enumerate(rr) if r["cat"] == cat])
            per_category[cat] = outcomes(y[ix], typ[ix], decision[ix])
        by_stratum = {}
        for typname in ["logical_anomalies", "structural_anomalies"]:
            ix = np.where(typ == typname)[0]
            esc = int(np.count_nonzero(decision[ix] == "release"))
            by_stratum[typname] = {"n": len(ix), "escapes": esc, "escape_ucb95": cp_upper(esc, len(ix))}
        lat = []
        for r in rows:
            # Both component and relational reuse the same extraction, not two separate passes.
            keys = set("component" if x in ("component", "relational") else x for x in components)
            lat.append(sum(all_scores[r["sha256"]]["latency_ms"][k] for k in keys))
        timing = {
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "n": len(lat),
            "method": "measured sequential CPU model scoring; excludes gate check and UI",
        }
        final_variants[name] = {
            "overall": overall,
            "per_category": per_category,
            "by_defect_stratum": by_stratum,
            "cpu_latency": timing,
        }
        print(
            "FINAL RESULT",
            name,
            "release",
            overall["good_released"],
            "/",
            overall["n_good"],
            "escapes",
            overall["escapes"],
            "/",
            overall["n_defect"],
            "cost",
            round(overall["usd_per_1000"], 2),
            flush=True,
        )
    winner = pol["winner"]
    if winner:
        primary = final_variants[winner]
        acceptable = primary["overall"]["escapes"] == 0 and primary["overall"]["escape_ucb95"] <= 0.025
    else:
        primary = {"all_human": True}
        acceptable = False
    res = {
        "protocol": "Frozen Phase3 one-time evaluation; previously unused LOCO categories",
        "frozen_policy_sha256": sha(OUT / "p3_frozen_policy.json"),
        "final_manifest_sha256": sha(OUT / "p3_locked_final.json"),
        "selected_on_development": winner,
        "winner_final": primary,
        "all_candidates_frozen_final": final_variants,
        "safety_acceptance_rule_met": bool(acceptable),
        "reporting_warning": "Public noncommercial benchmark only; not factory safety or deployment evidence. Prior Phase2 kits are different categories and not a paired comparison.",
        "n_final": len(rows),
        "completed_utc": time.strftime("%FT%TZ", time.gmtime()),
    }
    save_json_atomic(OUT / "p3_final_results.json", res)
    (OUT / "p3_final_results.sha256").write_text(sha(OUT / "p3_final_results.json") + "  p3_final_results.json\n")
    print("P3 FINAL COMPLETE", winner, "acceptance", acceptable, "hash", sha(OUT / "p3_final_results.json"), flush=True)


if __name__ == "__main__":
    main()
