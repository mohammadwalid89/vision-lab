#!/usr/bin/env python3
"""Matched pilot policy comparison from immutable scored rows; no final-set inputs."""

import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import beta

D = Path(os.environ.get("PILOT_DIR", Path(__file__).resolve().parent))  # holds pilot_results.json
CATS = ("juice_bottle", "splicing_connectors")
KEYS = ("eg2_768", "eg2_256", "resnet", "wrn", "component_relational")


def cp_upper(x, n):
    return float(beta.ppf(0.95, x + 1, n - x)) if x < n else 1.0


def main():
    p = json.loads((D / "pilot_results.json").read_text())
    rows = p["rows"]
    result = {
        "source": "pilot_results.json",
        "cohort": "Phase 3 exposed development only",
        "rule": "per-category threshold immediately below minimum calibration defect score; evaluation untouched by threshold selection",
        "methods": {},
        "all_human": {"good_releases": 0, "defect_escapes": 0, "reviews_per_1000": 1000, "modeled_usd_per_1000": 250},
    }
    for key in KEYS:
        by_cat = {}
        for cat in CATS:
            cal = [
                r[key] for r in rows if r["category"] == cat and r["split"] == "calibration" and r["stratum"] != "good"
            ]
            ev = [r for r in rows if r["category"] == cat and r["split"] == "evaluation"]
            threshold = float(np.nextafter(min(cal), -np.inf))
            good = [r for r in ev if r["stratum"] == "good"]
            defect = [r for r in ev if r["stratum"] != "good"]
            release = sum(r[key] <= threshold for r in good)
            escapes = sum(r[key] <= threshold for r in defect)
            by_cat[cat] = {
                "threshold": threshold,
                "good_releases": release,
                "good_n": len(good),
                "defect_escapes": escapes,
                "defect_n": len(defect),
                "logical_escapes": sum(r[key] <= threshold for r in defect if r["stratum"] == "logical_anomalies"),
                "structural_escapes": sum(
                    r[key] <= threshold for r in defect if r["stratum"] == "structural_anomalies"
                ),
                "escape_cp95_upper": cp_upper(escapes, len(defect)),
            }
        g = sum(x["good_releases"] for x in by_cat.values())
        gn = sum(x["good_n"] for x in by_cat.values())
        e = sum(x["defect_escapes"] for x in by_cat.values())
        en = sum(x["defect_n"] for x in by_cat.values())
        releases_per_1000 = 980 * g / gn + 20 * e / en
        reviews = 1000 - releases_per_1000
        cost = 0.25 * reviews + 50 * 20 * e / en
        result["methods"][key] = {
            "by_category": by_cat,
            "good_releases": g,
            "good_n": gn,
            "defect_escapes": e,
            "defect_n": en,
            "escape_cp95_upper": cp_upper(e, en),
            "reviews_per_1000": reviews,
            "modeled_usd_per_1000": cost,
            "auc": p["metrics"][key]["all"]["auc"],
        }
    out = D / "matched_comparison.json"
    if out.exists():
        raise SystemExit("Refusing to overwrite matched comparison")
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["methods"], indent=2))


if __name__ == "__main__":
    main()
