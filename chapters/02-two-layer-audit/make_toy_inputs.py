"""Write a toy input tree in the layout the audit scripts read, from random numbers.

Use it to check that the scripts run end to end without the real score tables:

    python make_toy_inputs.py --root toy_v31 --out toy_out
    VISION_V31_ROOT=toy_v31 VISION_OUT=toy_out python l2_complementarity.py

The values are random draws with a small anomaly shift. They carry no result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from common import AD_CATS6, AD_HELDOUT9, LOCO_CATS


def sha(rng) -> str:
    return "".join(rng.choice(list("0123456789abcdef"), 64))


def score_row(rng, dataset, category, role, truth, stratum, shift=1.0):
    anomaly = truth == "not_ok"
    base = rng.normal(0, 1) + (shift if anomaly else 0)
    return {"sha256": sha(rng), "dataset": dataset, "category": category, "role": role, "truth": truth,
            "source": f"{category}/{role}/{rng.integers(1_000_000)}.png", "stratum": stratum,
            "patch_score": float(10 + base), "clip_score": float(0.5 + 0.1 * rng.normal() + 0.05 * anomaly),
            "patch_cost": {"cpu": 0.3}, "clip_cost": {"cpu": 0.1}, "preprocess_cost": {"cpu": 0.05},
            "answer": "unknown"}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    ad_rows = []
    for cat in AD_CATS6:
        for role, n_good, n_bad in (("router_train", 6, 4), ("calibration", 20, 0), ("dev_audit", 10, 10)):
            ad_rows += [score_row(rng, "mvtec_ad", cat, role, "ok", "good") for _ in range(n_good)]
            ad_rows += [score_row(rng, "mvtec_ad", cat, role, "not_ok", "broken") for _ in range(n_bad)]
    write_jsonl(args.root / "results/v31/local/mvtec_ad/toy_scores.jsonl", ad_rows)

    loco_rows = []
    for cat in LOCO_CATS:
        for role, n_good, n_struct, n_logic in (("router_train", 8, 3, 3), ("calibration", 25, 0, 0),
                                                 ("dev_audit", 20, 10, 10)):
            loco_rows += [score_row(rng, "mvtec_loco", cat, role, "ok", "good", 0.6) for _ in range(n_good)]
            loco_rows += [score_row(rng, "mvtec_loco", cat, role, "not_ok", "structural_anomalies", 0.6)
                          for _ in range(n_struct)]
            loco_rows += [score_row(rng, "mvtec_loco", cat, role, "not_ok", "logical_anomalies", 0.6)
                          for _ in range(n_logic)]
    write_jsonl(args.root / "results/v31/local/mvtec_loco/toy_scores.jsonl", loco_rows)

    for family in ("semantic", "fused", "normal"):
        kit = []
        for r in loco_rows:
            shift = 1.0 if r["truth"] == "not_ok" else 0.0
            kit.append({"sha256": r["sha256"], "novelty_distance": float(rng.normal() + shift),
                        "cost": {"cpu": 0.7 if family == "semantic" else 7.0},
                        "clip_cost": {"cpu": 0.0}, "refinement_physical_cost": {"cpu": 0.0}})
        write_jsonl(args.root / f"results/v31/local_kit_{family}/mvtec_loco/toy_scores.jsonl", kit)
    write_jsonl(args.root / "results/v31/anomalyclip/mvtec_loco/toy.jsonl",
                [{"sha256": r["sha256"], "anomalyclip_score": float(rng.normal() + (r["truth"] == "not_ok"))}
                 for r in loco_rows])

    held = []
    for cat in AD_HELDOUT9:
        for role, n_good, n_bad in (("fit", 10, 0), ("calibration", 20, 0), ("support_anomaly", 0, 10),
                                    ("test_good", 10, 0), ("test_anomaly", 0, 15)):
            for truth, n in (("ok", n_good), ("not_ok", n_bad)):
                for _ in range(n):
                    row = score_row(rng, "mvtec_ad", cat, role, truth, "toy")
                    row["score_cost"] = {"cpu": 0.5}
                    held.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "heldout_ad_scores.jsonl", held)
    clip = {r["sha256"]: rng.normal(size=16).astype(np.float32) for r in held + ad_rows}
    np.savez_compressed(args.out / "clip_embeddings.npz", **clip)
    print(f"wrote {len(ad_rows)} AD, {len(loco_rows)} LOCO, {len(held)} held-out toy rows")


if __name__ == "__main__":
    main()
