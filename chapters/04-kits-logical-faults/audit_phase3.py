"""Read-only audit of the ten Phase 3 kit candidates from their frozen records.

Two subcommands, both reading the stored per-image scores and the frozen policy
(thresholds and category gate) of the original run and scoring no image with a model:

  records   rebuild the exposed-development decision table from the raw per-image
            scores, check it against the published frozen record, and collect the
            locked-final diagnostics published after the single post-freeze opening
  frontier  sweep the release threshold over the stored development p-values to see
            how much good-kit release any candidate could have bought at a 1% escape
            budget. The threshold is chosen on the rows it scores, so the numbers
            are optimistic by design.

The Phase 3 record folder (p3_frozen_policy.json, p3_final_results.json, p3_dev.json,
p3_<model>_<category>_dev.json) is an input; it is not part of this repository.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import beta

CATS = ["juice_bottle", "splicing_connectors"]
CANDIDATES = {
    "component": ["component"], "relational": ["relational"],
    "resnet": ["resnet"], "wrn": ["wrn"],
    "component_wrn": ["component", "wrn"],
    "relational_wrn": ["relational", "wrn"],
    "component_resnet": ["component", "resnet"],
    "relational_resnet": ["relational", "resnet"],
    "wrn_resnet": ["wrn", "resnet"],
    "all_four": ["component", "relational", "resnet", "wrn"],
}
COST = {"defect_prevalence": 0.02, "human_review_usd": 0.25, "false_hold_usd": 2.0, "escape_usd": 50.0}
ESCAPE_BUDGET = 0.01
STRATA = ["logical_anomalies", "structural_anomalies"]


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path: Path):
    return json.loads(Path(path).read_text())


def cp_upper(x: int, n: int, alpha: float = 0.05) -> float:
    return float(beta.ppf(1 - alpha, x + 1, n - x)) if x < n else 1.0


def cp_two_sided(x: int, n: int) -> list[float]:
    return [float(beta.ppf(0.025, x, n - x + 1)) if x else 0.0,
            float(beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0]


def rgb_mean(path: str) -> np.ndarray:
    with Image.open(path) as im:
        resized = im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.float32).reshape(-1, 3).mean(0) / 255


def outcomes(y, typ, dec) -> dict:
    good = y == 0
    ng, na = int(good.sum()), int((~good).sum())
    gr = int(np.count_nonzero(good & (dec == "release")))
    esc = int(np.count_nonzero((~good) & (dec == "release")))
    fh = int(np.count_nonzero(good & (dec == "hold")))
    gd = int(np.count_nonzero(good & (dec == "human")))
    ad = int(np.count_nonzero((~good) & (dec == "human")))
    held = int(np.count_nonzero((~good) & (dec == "hold")))
    pi, ch, cf, ce = (COST["defect_prevalence"], COST["human_review_usd"],
                      COST["false_hold_usd"], COST["escape_usd"])
    er, fhr = esc / max(na, 1), fh / max(ng, 1)
    gdr, adr = gd / max(ng, 1), ad / max(na, 1)
    return {
        "n_good": ng, "n_defect": na,
        "good_released": gr, "good_release_rate": gr / max(ng, 1),
        "good_release_ci95": cp_two_sided(gr, ng) if ng else None,
        "escapes": esc, "escape_rate": er,
        "escape_ucb95": cp_upper(esc, na) if na else None,
        "escapes_by_stratum": {t: int(np.count_nonzero((typ == t) & (dec == "release"))) for t in STRATA},
        "false_holds": fh, "good_to_person": gd, "defects_to_person": ad,
        "defects_held": held,
        "person_reviews_per_1000": 1000 * ((1 - pi) * gdr + pi * adr),
        "usd_per_1000": 1000 * ((1 - pi) * (fhr * cf + gdr * ch) + pi * (er * ce + adr * ch)),
    }


def load_dev(out: Path, policy: dict) -> dict:
    """Per category: dev rows, category-gate flags and per-family validation-normal p-values."""
    data = {}
    for cat in CATS:
        dev = [r for r in load(out / "p3_dev.json") if r["category"] == cat]
        g = policy["gate"][cat]
        med, scale = np.asarray(g["median"]), np.asarray(g["scale"])
        gate = np.array([float(np.sqrt(np.sum(((rgb_mean(r["path"]) - med) / scale) ** 2))) <= g["threshold"]
                         for r in dev], bool)
        p = {}
        for model in ("component_relational", "resnet", "wrn"):
            z = load(out / f"p3_{model}_{cat}_dev.json")
            for k in (["component", "relational"] if model == "component_relational" else [model]):
                key = k if k in ("component", "relational") else "score"
                val = np.sort(np.asarray([r[key] for r in z["validation"]], float))
                scored = {r["sha256"]: r for r in z["development"]}
                vals = np.asarray([scored[r["sha256"]][key] for r in dev])
                p[k] = (len(val) - np.searchsorted(val, vals, side="left") + 1) / (len(val) + 1)
        data[cat] = {
            "dev": dev, "gate": gate, "p": p,
            "y": np.asarray([int(r["stratum"] != "good") for r in dev]),
            "typ": np.asarray([r["stratum"] for r in dev]),
        }
    return data


def audit_records(out: Path, dest: Path) -> None:
    policy = load(out / "p3_frozen_policy.json")
    final = load(out / "p3_final_results.json")
    sha_ok = {name: sha(out / name) == h for name, h in policy["dev_score_sha256"].items()}
    data = load_dev(out, policy)

    rebuilt = {}
    for name, comps in CANDIDATES.items():
        y = np.concatenate([data[c]["y"] for c in CATS])
        typ = np.concatenate([data[c]["typ"] for c in CATS])
        gate = np.concatenate([data[c]["gate"] for c in CATS])
        pp = np.concatenate([np.min(np.stack([data[c]["p"][k] for k in comps]), 0) for c in CATS])
        ar = policy["variants"][name]["policy"]["ar"]
        ah = policy["variants"][name]["policy"]["ah"]
        cat_of = np.concatenate([[c] * len(data[c]["y"]) for c in CATS])
        dec = np.full(len(y), "human", dtype="<U8")
        for i in range(len(y)):
            if not gate[i]:
                continue
            if pp[i] > ar[cat_of[i]]:
                dec[i] = "release"
            elif pp[i] <= ah:
                dec[i] = "hold"
        rebuilt[name] = outcomes(y, typ, dec)
        pub = policy["variants"][name]["dev"]
        rebuilt[name]["matches_published_dev"] = all(
            rebuilt[name][k] == pub[k] for k in
            ("good_released", "escapes", "false_holds", "defects_held",
             "good_to_person", "defects_to_person", "n_good", "n_defect"))
        rebuilt[name]["gate_acceptance_rebuilt"] = {
            c: {"good": int(np.count_nonzero(data[c]["gate"] & (data[c]["y"] == 0))),
                "defect": int(np.count_nonzero(data[c]["gate"] & (data[c]["y"] == 1)))}
            for c in CATS}

    rows = []
    for name, comps in CANDIDATES.items():
        pub = policy["variants"][name]
        fin = final["all_candidates_frozen_final"][name]
        rows.append({
            "candidate": name, "components": comps,
            "dev_release_threshold_ar": pub["policy"]["ar"],
            "dev_hold_threshold_ah": pub["policy"]["ah"],
            "dev_full": pub["dev"],
            "dev_full_rebuilt": rebuilt[name],
            "dev_oos_mean_escape_rate": pub["oos_mean_escape_rate"],
            "dev_oos_any_escape_fraction": pub["oos_any_escape_fraction"],
            "dev_oos_mean_good_release": pub["oos_mean_good_release"],
            "dev_oos_good_release_p05_p95": pub["oos_good_release_p05_p95"],
            "dev_oos_mean_cost_usd_per_1000": pub["oos_mean_cost"],
            "final_overall": fin["overall"],
            "final_by_defect_stratum": fin["by_defect_stratum"],
            "final_per_category": fin["per_category"],
            "cpu_latency": fin["cpu_latency"],
        })
    acceptable = [r["candidate"] for r in rows if r["dev_oos_mean_escape_rate"] <= ESCAPE_BUDGET]
    record = {
        "audit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "read_files": {name: sha(out / name) for name in
                       ("p3_frozen_policy.json", "p3_final_results.json", "p3_split_manifest.json")},
        "dev_score_sha256_all_match": all(sha_ok.values()),
        "dev_score_sha256_check": sha_ok,
        "selection_rule": policy["selection_rule"],
        "selection_threshold_dev_oos_escape": ESCAPE_BUDGET,
        "candidates": rows,
        "no_candidate_qualified": acceptable == [],
        "winner": policy["winner"],
        "fallback": policy["fallback"],
        "dev_oos_escape_range": [min(r["dev_oos_mean_escape_rate"] for r in rows),
                                 max(r["dev_oos_mean_escape_rate"] for r in rows)],
        "final_n": final["n_final"],
        "rebuilt_dev_matches_published": {k: v["matches_published_dev"] for k, v in rebuilt.items()},
    }
    dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    for r in rows:
        print(f"{r['candidate']:20s} oos_esc={r['dev_oos_mean_escape_rate']:.4f} "
              f"final_good={r['final_overall']['good_released']:3d}/129 "
              f"final_esc={r['final_overall']['escapes']:3d}")


def audit_frontier(out: Path, dest: Path) -> None:
    policy = load(out / "p3_frozen_policy.json")
    data = load_dev(out, policy)
    result = {}
    for name, comps in CANDIDATES.items():
        ps = np.concatenate([np.min(np.stack([data[c]["p"][k] for k in comps]), 0) for c in CATS])
        ys = np.concatenate([data[c]["y"] for c in CATS])
        gs = np.concatenate([data[c]["gate"] for c in CATS])
        n_def = int((ys == 1).sum())
        best = None
        for thr in np.unique(np.round(ps, 6)):
            dec = gs & (ps > thr)
            esc = int(np.count_nonzero(dec & (ys == 1)))
            rel = int(np.count_nonzero(dec & (ys == 0)))
            if esc / n_def <= ESCAPE_BUDGET and (best is None or rel > best[1]):
                best = (esc / n_def, rel, esc, float(thr))
        result[name] = {
            "n_good": int((ys == 0).sum()), "n_defect": n_def,
            "max_in_sample_good_release_at_budget": best[1] if best else 0,
            "escapes_at_that_point": best[2] if best else None,
            "escape_rate_at_that_point": best[0] if best else None,
            "release_threshold_p": best[3] if best else None,
        }
        print(f"{name:18s} {result[name]['max_in_sample_good_release_at_budget']:3d}/"
              f"{result[name]['n_good']} good at <=1% escapes ({result[name]['escapes_at_that_point']} escapes)")
    record = {
        "escape_budget": ESCAPE_BUDGET,
        "method": ("release threshold swept over stored dev p-values with the frozen category gate; "
                   "chosen on the same dev rows it scores, so optimistic"),
        "frozen_policy_sha256": sha(out / "p3_frozen_policy.json"),
        "candidates": result,
    }
    dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["records", "frontier"])
    parser.add_argument("--phase3-dir", type=Path, required=True,
                        help="folder with the frozen Phase 3 records (p3_*.json)")
    parser.add_argument("--out", type=Path, required=True, help="JSON file to write")
    args = parser.parse_args()
    {"records": audit_records, "frontier": audit_frontier}[args.command](args.phase3_dir, args.out)


if __name__ == "__main__":
    main()
