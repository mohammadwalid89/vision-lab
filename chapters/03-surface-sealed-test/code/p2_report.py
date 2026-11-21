"""Read-only reproducibility manifest + results report from frozen one-shot Phase 2 evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).parent))
from p2_policy import Data, apply, decisions

D = Path(__file__).parent
OUT = D / "out"


def main():
    F = json.loads((OUT / "p2_frozen_policy.json").read_text())["systems"]
    R = json.loads((OUT / "p2_final_results.json").read_text())["systems"]

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    assert json.loads((OUT / "p2_selector_regression.json").read_text())["compared"] == 48
    assert json.loads((OUT / "p2_selector_regression.json").read_text())["all_identical"]
    for fam in ("ad", "loco"):
        for s in ("phase2", "v001", "v001_tuned"):
            o = R[f"{fam}|{s}"]["overall"]
            assert o["n_good"] == 100 and o["n_defect"] == 200
            assert o["escapes"] == 0 and o["escape_ucb95"] < 0.025
    pairs = {}
    for fam in ("ad", "loco"):
        D0 = Data(fresh=True)
        a = D0.arrays(fam, F[f"{fam}|phase2"]["system"])
        b = D0.arrays(fam, "v001")
        n = apply(a, F[f"{fam}|phase2"]["policy"])
        old = decisions(b, 0, 0)
        good = a["y"] == 0
        improved = int((good & (n == "release") & (old != "release")).sum())
        lost = int((good & (n != "release") & (old == "release")).sum())
        pairs[fam] = {
            "newly_released_good": improved,
            "no_longer_released_good": lost,
            "exact_two_sided_discordant_p": binomtest(improved, improved + lost).pvalue if improved + lost else None,
        }
    lat = {}
    for fname in ("base_scores", "wrn_scores"):
        rows = [json.loads(s) for s in (OUT / (fname + ".jsonl")).open() if s.strip()]
        for fam, cats in [
            ("ad", ["bottle", "cable", "hazelnut", "pill", "tile", "transistor"]),
            ("loco", ["breakfast_box", "pushpins", "screw_bag"]),
        ]:
            times = [
                r["ms"]["embed_wall"] + r["ms"]["knn_wall"] if fname == "wrn_scores" else r["ms"]["patchcore"]
                for r in rows
                if r["category"] in cats
            ]
            lat[f"{fname}|{fam}"] = {
                "n": len(times),
                "p50_ms": float(np.median(times)),
                "p95_ms": float(np.quantile(times, 0.95)),
            }
    manifest = {
        "date_utc": "2026-10-28",
        "experiment": "Vision System One Phase 2",
        "status": "fresh evaluated once; surface preview verified; kits not improved",
        "sources": {
            # the pre-registration sits in the chapter folder, one level above code/
            "PREREGISTRATION.md": sha(D.parent / "PREREGISTRATION.md"),
            **{
                str(p.relative_to(D)): sha(p)
                for p in [
                    D / "p2_policy.py",
                    D / "p2_policy_fast.py",
                    D / "p2_final.py",
                    D / "p2_patchcore_wrn.py",
                    OUT / "p2_frozen_policy.json",
                    OUT / "p2_final_results.json",
                    OUT / "p2_dev_results.json",
                    OUT / "p2_selector_regression.json",
                ]
            },
        },
        "models": {p.name: sha(p) for p in sorted((D / "models").glob("wrn_*.pt"))},
        "pretrained_backbone": {
            "source": "torchvision Wide_ResNet50_2_Weights.IMAGENET1K_V1",
            "sha256": sha(
                Path(os.environ.get("TORCH_HOME", Path.home() / ".cache/torch"))
                / "hub/checkpoints/wide_resnet50_2-95faca4d.pth"
            ),
        },
        "score_rows": {
            name: sum(1 for _ in (OUT / name).open())
            for name in ("base_scores.jsonl", "wrn_scores.jsonl", "comp_scores.jsonl")
        },
        "selector_regression": "48/48 identical policy selections (not formal proof)",
        "paired_good": pairs,
        "scoring_latency": lat,
        "policy_frozen_before_fresh": "2026-10-28 18:54:07 UTC",
        "fresh_evaluated_once": "2026-10-28 18:54:09 UTC",
        "caveats": [
            "Public MVTec AD/LOCO benchmark; no new-camera, new-lot, or factory transfer validation",
            "Fresh set was sealed before policy freeze and has since been opened once; do not tune on it",
            "Cost uses illustrative 2% defect prevalence and $0.25 review / $2 false hold / $50 escape",
            "WRN-50 scoring p95 includes CPU contention from simultaneous experiments",
            "LOCO frozen winner is v001_tuned; preview deliberately retains deployed v001, not a new kit model",
        ],
    }
    (OUT / "p2_repro_manifest.json").write_text(json.dumps(manifest, indent=2))
    lines = [
        "# Vision System One — Phase 2 frozen evaluation",
        "",
        "**Date:** 2026-10-28 UTC. **Status:** surface-family research improvement validated on one sealed MVTec fresh split; kit-family improvement **not** validated. **Deployment:** separate research preview; production v001 unchanged.",
        "",
        "## Protocol and integrity",
        "",
        "- Train-good data only for WRN-50 PatchCore memory banks and five-fold out-of-fold calibration. Thresholds selected on previously exposed development images, then frozen **before** opening the reserved fresh set.",
        "- Each family: **100 good and 200 defective fresh images**. Original v001, threshold-only v001_tuned, and Phase 2 compared on identical images; public benchmark capture distribution, not production.",
        "- 300 stratified exposed-development half-splits. Vectorized selector matched original on **48/48** paired regression checks (empirical equivalence, not formal proof). All score inputs and frozen policy SHA256-sealed.",
        "- Pre-registered safety check: zero escapes on selection half, and one-sided 95% Clopper–Pearson escape upper bound **≤ 2.5%** per family on fresh images. Both frozen families meet the statistical bound, but this does not prove factory safety.",
        "",
        "## One-time fresh evaluation",
        "",
        "| Family / policy | Good auto-release | Defect escapes | Good to person | All person reviews / 1k at 2% defects | Illustrative cost / 1k |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for fam in ("ad", "loco"):
        for sysname in ("phase2", "v001", "v001_tuned"):
            o = R[f"{fam}|{sysname}"]["overall"]
            lines.append(
                f"| {fam.upper()} / {sysname} | {o['good_released']}/100 ({100 * o['good_release_rate']:.0f}%) | {o['escapes']}/200 | {100 * o['good_to_person_rate']:.0f}% | {o['person_reviews_per_1000']:.1f} | ${o['usd_per_1000']:.2f} |"
            )
    lines += [
        "",
        "The **AD Phase 2** release estimate is **84%** (two-sided exact 95% CI **75.3–90.6%**), versus **4%** for v001. On the same 100 good images, 80 gained automatic release and none lost it (exact paired discordance p≈1.65×10⁻²⁴). At the stated illustrative cost assumptions, person reviews fall from **956.8 to 162.9 per 1,000 parts**, and cost from **$239.20 to $40.73 per 1,000 parts** (about **83%** lower). These are modeled savings, not measured business savings.",
        "",
        "Both AD and LOCO saw **0/200 escaped defects**; the one-sided 95% upper bound is **1.49% per family**, not 0%. Category-specific bounds are much wider. These results meet the preregistered 2.5% *family-level* criterion but are insufficient for a 0.5% production-risk claim.",
        "",
        "## Per-category fresh results (Phase 2 frozen systems)",
        "",
        "| Family | Category | Good auto-release | Defect escapes | Good to person |",
        "|---|---|---:|---:|---:|",
    ]
    for fam in ("ad", "loco"):
        for cat, o in R[f"{fam}|phase2"]["per_category"].items():
            lines.append(
                f"| {fam.upper()} | {cat} | {o['good_released']}/{o['n_good']} | {o['escapes']}/{o['n_defect']} | {100 * o['good_to_person_rate']:.0f}% |"
            )
    lines += [
        "",
        "## Per-category workload and cost comparison (illustrative 2% defect prevalence within each category)",
        "",
        "| Family | Category | Policy | Good release | Escapes | Person reviews / 1k | Cost / 1k |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for fam in ("ad", "loco"):
        for cat in sorted(R[f"{fam}|phase2"]["per_category"]):
            for sysname in ("phase2", "v001", "v001_tuned"):
                o = R[f"{fam}|{sysname}"]["per_category"][cat]
                lines.append(
                    f"| {fam.upper()} | {cat} | {sysname} | {o['good_released']}/{o['n_good']} | {o['escapes']}/{o['n_defect']} | {o['person_reviews_per_1000']:.1f} | ${o['usd_per_1000']:.2f} |"
                )
    lines += [
        "",
        "### Material limitation: kits (MVTec LOCO)",
        "",
        "The development-selected kit winner was **v001_tuned**, not a new WRN/component architecture. It showed roughly **20%** good-kit release on exposed development half-splits but **0/100** on the reserved fresh kits. Thus the kit work **failed to generalize**; all good kits still require manual review and all fresh defective kits also go to a person under the frozen variant. The deployed v001 baseline itself releases 0/100 fresh good kits and sends 99% of goods to a person. **No kit automation improvement should be claimed.**",
        "The rejected OWLv2 kit counter also produced incorrect counts and required 33–43 seconds/image on this CPU. The component verifier and WRN kit variants remain research artifacts; they are not enabled in the preview.",
        "",
        "### Surface limitations",
        "",
        "Pill releases **0/11** fresh good images; the other five categories account for the gain. Small per-category denominators (e.g. transistor only 7 defects) cannot certify per-product risk. The category gate can refuse an image, which always requires a person rather than being counted as a release.",
        "",
        "## CPU performance and economics",
        "",
        f"Across {lat['base_scores|ad']['n']} AD evaluation images, v001 PatchCore median/p95 score time was **{lat['base_scores|ad']['p50_ms']:.0f}/{lat['base_scores|ad']['p95_ms']:.0f} ms**. WRN-50 embedding + kNN was **{lat['wrn_scores|ad']['p50_ms']:.0f}/{lat['wrn_scores|ad']['p95_ms']:.0f} ms**; p95 was inflated by concurrent CPU experiments. In the separate idle live preview, after first-load warmup, sampled inference is approximately 0.3–0.6 seconds/image. The larger backbone is slower but the modeled reduction in human reviews dominates the illustrative cost function. **CPU cost itself was not separately priced in the Phase 2 cost metric.**",
        "",
        "## Follow-on research, not a retrofit of this result",
        "",
        "A future kit iteration requires an independently reserved **new** evaluation population (ideally different capture sessions/lots/cameras) and enough confirmed good kits per category for calibration. Do not adjust thresholds against the already opened Phase 2 fresh set and then present it as untouched. Production shadow-mode validation is required before any autonomous release.",
        "",
    ]
    (D / "FINAL_PHASE2_EVALUATION.md").write_text("\n".join(lines))
    print("wrote report", D / "FINAL_PHASE2_EVALUATION.md", "and manifest; lines", len(lines))


if __name__ == "__main__":
    main()
