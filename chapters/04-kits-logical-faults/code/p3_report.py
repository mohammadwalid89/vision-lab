#!/usr/bin/env python3
"""Generate Phase3 markdown report from frozen one-time final results."""

import json

from p3_models import OUT, D, sha


def pc(x):
    return f"{x * 100:.2f}%"


def main():
    r = json.loads((OUT / "p3_final_results.json").read_text())
    f = json.loads((OUT / "p3_frozen_policy.json").read_text())
    assert r["frozen_policy_sha256"] == sha(OUT / "p3_frozen_policy.json")
    win = f["winner"]
    lines = [
        "# Vision System One — Phase 3: Independent Kit Generalization",
        "",
        "**Completed:** " + r["completed_utc"],
        "**Development-selected policy:** " + str(win or "all-human fallback"),
        "**Acceptance criterion met on public benchmark:** " + str(r["safety_acceptance_rule_met"]),
        "",
        "## Independence and methods",
        "",
        "Pre-registered before scoring. New MVTec LOCO AD categories: juice_bottle and splicing_connectors. Pinned public mirror revision 464875d33af84257cac350ebcf2c4210343f85a3. Dataset attribution: Bergmann et al., IJCV 2022. **CC BY-NC-SA 4.0, noncommercial research only.** No raw benchmark images published.",
        "",
        "Fit only on train normals (335 juice bottles, 360 splicing connectors). Calibrate anomaly tail scores and common category gate only on validation normals (54, 60). Test image paths were hash-seeded and stratified before scoring into 40% exposed development (84 good, 149 defects) and 60% locked final (129 good, 228 defects: 151 logical, 77 structural). The earlier opened Phase 2 test set was never reused.",
        "",
        "Candidates: component verifier (color-material regions and grid counts), relational component geometry (material centroids and pairwise spatial relationships), pretrained ResNet-18 PatchCore, pretrained WRN-50 PatchCore, and six conservative score ensembles. PatchCore uses 224x224 input and seeded approximate 512-patch coresets. All candidates use identical splits and a common train/validation-normal category gate.",
        "",
        "Development-only selection: 100 deterministic stratified half-split trials. Eligible candidates have mean held-out development escape rate <=1%; select minimum modeled cost. Category-specific strict-release threshold ensures zero escapes on the full exposed development set. Hold threshold selected from precommitted grid. Candidate and policy frozen and hashed before opening locked final. The independent final results are never used for policy tuning.",
        "",
        "## Frozen selected policy — locked final",
        "",
    ]
    if win:
        s = r["winner_final"]
        o = s["overall"]
        t = s["cpu_latency"]
        lines += [
            "| Category | Good released (95% two-sided CP CI) | Escapes (one-sided 95% CP upper) | Person reviews per 1,000 | Modeled USD per 1,000 |",
            "|---|---|---|---:|---:|",
        ]
        for name, x in [("Overall", o)] + list(s["per_category"].items()):
            ci = x["good_release_ci95"]
            lines.append(
                f"| {name} | {x['good_released']}/{x['n_good']} ({pc(x['good_release_rate'])}, CI {pc(ci[0])}–{pc(ci[1])}) | {x['escapes']}/{x['n_defect']} (upper {pc(x['escape_ucb95'])}) | {x['person_reviews_per_1000']:.1f} | {x['usd_per_1000']:.2f} |"
            )
        lines += ["", "Defect-subtype bounds:"]
        for name, x in s["by_defect_stratum"].items():
            lines.append(f"- {name}: {x['escapes']}/{x['n']} escapes, one-sided 95% CP upper {pc(x['escape_ucb95'])}.")
        lines += [
            "",
            f"Measured shared-VPS CPU latency, selected policy (n={t['n']}): p50 {t['p50_ms']:.1f} ms, p95 {t['p95_ms']:.1f} ms; excludes category gate and UI. No dedicated hardware isolation.",
            f"Good-kit automatic releases: {o['good_releases_per_1000_good']:.0f}/1,000 good kits; false holds {o['false_holds']}/{o['n_good']}; good-kit manual reviews {o['good_to_person']}/{o['n_good']}; defect manual reviews {o['defects_to_person']}/{o['n_defect']}.",
            "",
        ]
    else:
        lines += [
            "No model met the development selection criterion. The preregistered fallback routes all kits to human review. At the 2% modeled defect prevalence, this means **0/129 good-kit automatic releases**, **0/228 automatic defect escapes by construction**, **1,000 human reviews per 1,000 inspections**, and **USD 250 per 1,000 inspections** under the illustrative USD 0.25 review cost. Zero escapes from refusing all automatic releases is not evidence of detection capability."
        ]
    lines += [
        "## Development generalization diagnostic",
        "",
        "Mean out-of-sample escape rates across the 100 stratified development half-splits (not independent final tests): "
        + ", ".join(k + " " + pc(v["oos_mean_escape_rate"]) for k, v in f["variants"].items())
        + ". The 1% development eligibility rule was fixed before final opening; no candidate qualified.",
        "",
        "## All frozen candidate results (not post-hoc deployment choices)",
        "",
        "| Candidate | Final good releases | Final defect escapes | Escape 95% CP upper | Human reviews / 1,000 | Modeled USD / 1,000 | CPU p50/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, x in r["all_candidates_frozen_final"].items():
        o = x["overall"]
        t = x["cpu_latency"]
        lines.append(
            f"| {name}{' (selected)' if name == win else ''} | {o['good_released']}/{o['n_good']} | {o['escapes']}/{o['n_defect']} | {pc(o['escape_ucb95'])} | {o['person_reviews_per_1000']:.1f} | {o['usd_per_1000']:.2f} | {t['p50_ms']:.0f}/{t['p95_ms']:.0f} |"
        )
    lines += [
        "",
        "The following per-category and defect-subtype counts are descriptive, not post-final selection criteria. Zero observed escapes for any non-selected candidate do not override its failed development eligibility test.",
        "",
        "| Candidate | Juice good releases | Juice escapes | Splicing good releases | Splicing escapes | Logical escapes | Structural escapes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, x in r["all_candidates_frozen_final"].items():
        j = x["per_category"]["juice_bottle"]
        s = x["per_category"]["splicing_connectors"]
        logical = x["by_defect_stratum"]["logical_anomalies"]
        structural = x["by_defect_stratum"]["structural_anomalies"]
        lines.append(
            f"| {name} | {j['good_released']}/{j['n_good']} | {j['escapes']}/{j['n_defect']} | {s['good_released']}/{s['n_good']} | {s['escapes']}/{s['n_defect']} | {logical['escapes']}/{logical['n']} | {structural['escapes']}/{structural['n']} |"
        )
    lines += [
        "",
        "For zero observed escapes, the one-sided 95% Clopper-Pearson upper bounds are 2.57% for juice_bottle (n=115 defects), 2.62% for splicing_connectors (n=113), 1.96% for logical anomalies (n=151), and 3.82% for structural anomalies (n=77). Per-candidate bounds, including nonzero-escape cases, are preserved in out/p3_final_results.json. These bounds do not imply factory readiness.",
        "",
        "All cost figures are illustrative: 2% defect prevalence, USD 0.25 per human review, USD 2.00 per false hold and USD 50.00 per escaped defect. They exclude model compute, labor scheduling, physical handling, integration and deployment. Cost savings are NOT observed customer results.",
        "",
        "## Historical Phase 2 result — NOT a paired comparison",
        "",
        "The frozen Phase 2 kit study on different categories (breakfast_box, pushpins, screw_bag) released **0/100 good kits** and observed **0/200 escapes**. Different categories, distribution, models and split mean Phase 3 cannot be interpreted as a paired causal improvement. The opened Phase 2 final set was not reused.",
        "",
        "## Limits and release governance",
        "",
        "These are public research captures, not a production distribution. Even zero observed escapes cannot prove zero risk. Clopper–Pearson bounds assume binomial sampling and may understate uncertainty with correlated captures or distribution shift. Logical and structural faults, lighting, unseen layouts and rare defects remain unresolved. This work does not certify factory safety, justify unsupervised industrial deployment, or permit commercial redistribution of CC BY-NC-SA benchmark data.",
        "",
        "## Reproducibility ledger",
        "",
        "- Pre-registration: PREREGISTRATION.md, including source revision correction before scoring.",
        "- Split SHA256: " + f["split_manifest_sha256"],
        "- Locked final manifest SHA256: " + f["locked_final_manifest_sha256"],
        "- Frozen policy SHA256: " + r["frozen_policy_sha256"],
        "- Final results SHA256: " + sha(OUT / "p3_final_results.json"),
        "- Selector code SHA256: " + f["selector_code_sha256"],
        "- Training/evaluation source: p3_models.py, p3_select_freeze.py, p3_final.py.",
        "- Logs: out/p3_dev_worker.log, out/p3_selection.log, out/p3_final.log.",
        "",
        "### Frozen model file SHA256 hashes",
        "",
    ]
    for name, h in sorted(f["model_sha256"].items()):
        lines.append("- " + name + ": " + h)
    lines.append("")
    p = D / "FINAL_PHASE3_EVALUATION.md"
    p.write_text("\n".join(lines))
    print("REPORT", p, "bytes", p.stat().st_size)


if __name__ == "__main__":
    main()
