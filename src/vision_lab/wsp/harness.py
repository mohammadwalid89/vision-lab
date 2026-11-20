"""Per-attempt evaluation harness (one slot x one k x all seeds).

Input per seed: a dict of *pipelines* ``{name: PipelineScores}`` where each pipeline has
  role       'A' | 'candidate' | 'placebo' | 'retrainer' | 'noop' | 'reference' | 'B'
  cal_scores scores on the calibration views (TRAIN rows, same transform as the pipeline)
  dev_scores scores on all DEV views (clean, marker, nuisance), in ``dev_rows`` order
  noop_ref   (no-ops only) name of the pipeline whose scores the no-op must reproduce

Steps (identical for every pipeline; nothing is chosen from DEV):
  1. threshold: per-pipeline Youden-J on clean calibration views (amended protocol) and,
     side by side, the shared-A rule (A's calibration threshold applied to every arm);
  2. DEV metrics at that frozen threshold (metrics.score_views);
  3. signed gates (gates.seed_values) vs the rebuilt A, with placebos and retrainers;
  4. exact feasibility test (metrics.feasibility) on clean DEV views;
  5. effect calls arm - A (significance AND minimum effect, source-paired);
  6. no-op integrity: max |score - reference score| <= NOOP_TOL on calibration and DEV and
     every effect call SAME; any failure -> attempt status INCONCLUSIVE (harness bug or
     non-deterministic pipeline), never a scientific result.
Aggregation over seeds: worst seed per gate (gates.evaluate); feasibility must pass on
every seed.  Status: PASS | NULL_OR_NEGATIVE | INCONCLUSIVE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from . import gates as G
from .calibration import AccessLedger, freeze_pipeline_threshold, threshold_bootstrap
from .metrics import feasibility, paired_units, score_views
from .significance import (
    ALPHA,
    MIN_EFFECT,
    noop_false_positive_rate,
    paired_metric_calls,
)

NOOP_TOL = 1e-9
ROLES = ("A", "candidate", "placebo", "retrainer", "noop", "reference", "B")


@dataclass
class PipelineScores:
    role: str
    cal_scores: np.ndarray
    dev_scores: np.ndarray
    noop_ref: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.role not in ROLES:
            raise ValueError(f"unknown role {self.role!r}")
        self.cal_scores = np.asarray(self.cal_scores, dtype=float)
        self.dev_scores = np.asarray(self.dev_scores, dtype=float)
        if self.role == "noop" and not self.noop_ref:
            raise ValueError("no-op pipelines must name their reference")


@dataclass(frozen=True)
class AttemptSpec:
    slot: str
    k: int
    seeds: Sequence[int]
    gate_families: Sequence[str] = ("CORE",)
    threshold_rules: Sequence[str] = ("per_pipeline", "shared_A")
    binding_rule: str = "per_pipeline"
    feasibility_tol: float = 0.02
    alpha: float = ALPHA
    recipe_sha256: str = ""


def _metric_view(m: Mapping) -> dict:
    keys = ("recall", "FPR", "AUROC", "AP", "marker_flip_rate", "conflict_set_FP", "conflict_set_FPR",
            "mean_marker_score_change", "marker_flip_rate_label0", "marker_flip_rate_label1", "threshold")
    return {k: m.get(k) for k in keys}


def evaluate_seed(spec: AttemptSpec, seed: int, cal_rows: Sequence[dict], dev_rows: Sequence[dict],
                  pipes: Mapping[str, PipelineScores], ledger: AccessLedger | None = None) -> dict:
    roles = {n: p.role for n, p in pipes.items()}
    if list(roles.values()).count("A") != 1:
        raise ValueError("exactly one A pipeline required")
    if "candidate" not in roles.values():
        raise ValueError("a candidate pipeline is required")
    a_name = next(n for n, r in roles.items() if r == "A")
    out: dict = dict(seed=seed, thresholds={}, rules={})
    thr = {n: freeze_pipeline_threshold(cal_rows, p.cal_scores, ledger, arm=n, seed=seed)
           for n, p in pipes.items()}
    out["thresholds"] = {n: dict(threshold=t["threshold"], cal_recall=t["recall"], cal_FPR=t["FPR"])
                         for n, t in thr.items()}
    ci = np.array([i for i, r in enumerate(dev_rows) if r["variant"] == "clean"])
    y_clean = np.array([dev_rows[i]["label"] for i in ci], dtype=int)
    cal_ci = np.array([i for i, r in enumerate(cal_rows) if r["variant"] == "clean"])
    y_cal = np.array([cal_rows[i]["label"] for i in cal_ci], dtype=int)
    out["threshold_bootstrap_nongating"] = {
        n: threshold_bootstrap(y_cal, p.cal_scores[cal_ci], n_boot=200, seed=seed, dev=(y_clean, p.dev_scores[ci]))
        for n, p in pipes.items() if p.role in ("A", "candidate")}
    for rule in spec.threshold_rules:
        tr = {n: (thr[n]["threshold"] if rule == "per_pipeline" else thr[a_name]["threshold"]) for n in pipes}
        met = {n: score_views(dev_rows, p.dev_scores, tr[n]) for n, p in pipes.items()}
        a = met[a_name]
        placebos = [met[n] for n, r in roles.items() if r == "placebo"]
        retr = {n: met[n] for n, r in roles.items() if r == "retrainer"}
        b = next((met[n] for n, r in roles.items() if r == "B"), None)
        units = {n: paired_units(dev_rows, p.dev_scores, tr[n]) for n, p in pipes.items()}
        arms = {}
        for n, p in pipes.items():
            if p.role in ("A", "noop"):
                continue
            entry = dict(role=p.role, metrics=_metric_view(met[n]))
            if p.role == "candidate":
                fam = [f for f in spec.gate_families if f != "G7" or b is not None]
                entry["gate_values"] = G.seed_values(met[n], a, placebos, retr or None, b, fam)
            entry["feasibility"] = feasibility(y_clean, p.dev_scores[ci], a["recall"], a["FPR"],
                                               spec.feasibility_tol)
            entry["effect_calls_vs_A"] = paired_metric_calls(units[n], units[a_name], spec.alpha, MIN_EFFECT, seed)
            arms[n] = entry
        noop = {}
        for n, p in pipes.items():
            if p.role != "noop":
                continue
            ref = pipes[p.noop_ref]
            d_cal = float(np.max(np.abs(p.cal_scores - ref.cal_scores)))
            d_dev = float(np.max(np.abs(p.dev_scores - ref.dev_scores)))
            calls = paired_metric_calls(units[n], units[p.noop_ref], spec.alpha, MIN_EFFECT, seed)
            noop[n] = dict(max_abs_score_change_cal=d_cal, max_abs_score_change_dev=d_dev,
                           calls={k: v["call"] for k, v in calls.items()},
                           ok=bool(d_cal <= NOOP_TOL and d_dev <= NOOP_TOL and
                                   all(v["call"] == "SAME" for v in calls.values())), _calls=calls)
        fp = noop_false_positive_rate({n: v.pop("_calls") for n, v in noop.items()})
        out["rules"][rule] = dict(A=_metric_view(a), arms=arms, noops=noop, noop_false_positive=fp,
                                  impossibility=G.impossibility_flags(a, placebos, retr or None))
    return out


def evaluate_attempt(spec: AttemptSpec, per_seed: Mapping[int, dict]) -> dict:
    """Aggregate evaluate_seed outputs over seeds (worst seed decides)."""
    if sorted(per_seed) != sorted(spec.seeds):
        raise ValueError("results must cover exactly the declared seeds")
    report = dict(slot=spec.slot, k=spec.k, seeds=list(spec.seeds), recipe_sha256=spec.recipe_sha256,
                  binding_rule=spec.binding_rule, rules={})
    integrity_ok = True
    for rule in spec.threshold_rules:
        cand_names = sorted({n for s in per_seed.values() for n, e in s["rules"][rule]["arms"].items()
                             if e["role"] == "candidate"})
        rr: dict = dict(candidates={})
        for c in cand_names:
            vals = {str(s): per_seed[s]["rules"][rule]["arms"][c]["gate_values"] for s in spec.seeds}
            ev = G.evaluate(vals)
            feas = {str(s): per_seed[s]["rules"][rule]["arms"][c]["feasibility"] for s in spec.seeds}
            feas_all = all(f["passed"] for f in feas.values())
            rr["candidates"][c] = dict(
                gates=ev, feasibility_all_seeds=feas_all,
                feasibility_worst_recall_margin=min(f["recall_margin"] for f in feas.values()),
                pass_all=bool(ev["pass_all"] and feas_all),
                effect_calls={str(s): {m: v["call"] for m, v in
                                       per_seed[s]["rules"][rule]["arms"][c]["effect_calls_vs_A"].items()}
                              for s in spec.seeds})
        noops_ok = all(v["ok"] for s in per_seed.values() for v in s["rules"][rule]["noops"].values())
        rr["noop_integrity"] = noops_ok
        rr["impossibility"] = {str(s): per_seed[s]["rules"][rule]["impossibility"] for s in spec.seeds}
        integrity_ok &= noops_ok
        report["rules"][rule] = rr
    bind = report["rules"][spec.binding_rule]
    if not integrity_ok:
        status = "INCONCLUSIVE"
    elif any(c["pass_all"] for c in bind["candidates"].values()):
        status = "PASS"
    else:
        status = "NULL_OR_NEGATIVE"
    report["status"] = status
    report["status_rule"] = ("INCONCLUSIVE if any no-op moved scores or drew a non-SAME call; PASS iff a "
                             "candidate passes every declared gate AND the feasibility test on every seed "
                             "under the binding threshold rule; otherwise NULL_OR_NEGATIVE")
    return report
