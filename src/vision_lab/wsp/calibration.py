"""Calibration partition, information-budget ledger and per-pipeline threshold freezing.

Partition rules (identical to the rules the R3 probes used to build the published
calibration partition):
  * unit = source group; both TRAIN views of a source move together;
  * excluded from the slice = union over all seeds of the LARGEST k list (k=128);
  * slice = 384 good + 128 defect sources; per-label category quotas proportional to the
    eligible pool (largest remainder, ties by category name);
  * order inside a (label, category) stratum: sha256('r3v2|calib|' + source_id), then id.

Ledger rules:
  * calibration sources may ONLY be used to freeze one threshold per pipeline/seed;
  * A, scalers, projections, attribution, feature selection, retrainers, references,
    preservation sets and pilots must not touch a calibration source;
  * counterfactual adaptation pairs must come from the seed's purchased k list;
  * nothing may be charged from DEV or holdout for fitting; holdout is never chargeable.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .metrics import youden_threshold
from .ordering import ordered_sources

CAL_N = {0: 384, 1: 128}
NAMESPACE = "r3v2|calib|"

FIT_PURPOSES = frozenset({
    "fit_A", "fit_scaler", "fit_projection", "mask_attribution", "feature_selection",
    "retrain", "reference_fit", "preserve_set", "pilot", "placebo_fit"})
ADAPT_PURPOSE = "adapt_pairs"
CALIBRATE_PURPOSE = "calibrate_threshold"


def _digest(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def partition_sources(train_rows: Sequence[dict], seeds: Sequence[int], k_largest: int = 128,
                      cal_n: Mapping[int, int] = CAL_N, namespace: str = NAMESPACE,
                      min_per_label: int = 64) -> dict:
    """Source-level split of active TRAIN into fitting pool and calibration slice."""
    by_source = {}
    for r in train_rows:
        if r.get("split", "train") != "train":
            raise PermissionError("partition takes TRAIN rows only")
        if r["source_id"] in by_source:
            raise ValueError("one row per source expected")
        by_source[r["source_id"]] = r
    excluded = sorted({r["source_id"] for s in seeds
                       for r in ordered_sources(train_rows, s, min_per_label)[:k_largest]})
    excl = set(excluded)
    eligible = [r for r in train_rows if r["source_id"] not in excl]
    if len(eligible) < sum(cal_n.values()):
        raise RuntimeError("insufficient eligible TRAIN capacity for the calibration slice")
    chosen, quotas = {}, {}
    for label, total in cal_n.items():
        pool = [r for r in eligible if r["label"] == label]
        if len(pool) < total:
            raise RuntimeError(f"label {label}: pool {len(pool)} < quota {total}")
        cats = sorted({r["category"] for r in pool})
        n = {c: sum(1 for r in pool if r["category"] == c) for c in cats}
        raw = {c: total * n[c] / len(pool) for c in cats}
        base = {c: int(math.floor(raw[c])) for c in cats}
        short = total - sum(base.values())
        for c in sorted(cats, key=lambda c: (-(raw[c] - base[c]), c))[:short]:
            base[c] += 1
        take = []
        for c in cats:
            ranked = sorted([r for r in pool if r["category"] == c],
                            key=lambda r: (_digest(namespace + r["source_id"]), r["source_id"]))
            if base[c] > len(ranked):
                raise RuntimeError("category quota exceeds pool")
            take.extend(ranked[:base[c]])
        chosen[label] = take
        quotas[str(label)] = dict(requested=total, achieved=len(take), quota_by_category=base)
    cal_ids = sorted(r["source_id"] for lab in chosen for r in chosen[lab])
    cal = set(cal_ids)
    if cal & excl:
        raise AssertionError("calibration slice overlaps a k list")
    fit_ids = sorted(s for s in by_source if s not in cal)
    body = dict(calibration_source_ids=cal_ids, fitting_source_ids=fit_ids,
                excluded_k_union=excluded, quotas=quotas,
                rules=dict(namespace=namespace, k_largest=k_largest, seeds=list(seeds),
                           cal_n={str(k): v for k, v in cal_n.items()}))
    body["partition_digest"] = _digest(json.dumps(
        dict(cal=cal_ids, fit=fit_ids, excl=excluded), sort_keys=True))
    return body


def verify_against_published(partition: Mapping, published: Mapping) -> dict:
    """Compare a recomputed partition with the probe's CALIBRATION_PARTITION.json."""
    pub = sorted(published["calibration"]["source_ids"])
    ours = sorted(partition["calibration_source_ids"])
    return dict(equal=pub == ours, n_published=len(pub), n_recomputed=len(ours),
                only_published=sorted(set(pub) - set(ours))[:10],
                only_recomputed=sorted(set(ours) - set(pub))[:10])


@dataclass
class AccessLedger:
    """Records and enforces who used which TRAIN sources for what."""
    calibration_sources: frozenset
    purchased: Mapping[int, Mapping[int, Sequence[str]]]   # seed -> k -> source ids
    source_split: Mapping[str, str] | None = None            # optional id -> split
    entries: list = field(default_factory=list)

    def __post_init__(self):
        self.calibration_sources = frozenset(self.calibration_sources)

    def charge(self, purpose: str, source_ids: Iterable[str], arm: str = "", seed: int | None = None,
               k: int | None = None) -> None:
        ids = set(source_ids)
        if self.source_split is not None:
            bad = {i for i in ids if self.source_split.get(i) != "train"}
            if bad:
                raise PermissionError(f"{purpose}: non-TRAIN sources charged ({len(bad)})")
        if purpose in FIT_PURPOSES:
            leak = ids & self.calibration_sources
            if leak:
                raise PermissionError(f"{purpose} for {arm!r} touches {len(leak)} calibration sources")
        elif purpose == ADAPT_PURPOSE:
            if seed is None or k is None:
                raise ValueError("adaptation charge needs seed and k")
            allowed = set(self.purchased[seed][k])
            if not ids <= allowed:
                raise PermissionError(f"adaptation for {arm!r} uses {len(ids - allowed)} unpurchased sources")
        elif purpose == CALIBRATE_PURPOSE:
            if not ids <= self.calibration_sources:
                raise PermissionError("threshold calibration outside the calibration slice")
        else:
            raise ValueError(f"unknown purpose {purpose!r}")
        self.entries.append(dict(purpose=purpose, arm=arm, seed=seed, k=k, n_sources=len(ids),
                                 sources_sha256=_digest("|".join(sorted(ids)))))

    def report(self) -> dict:
        by = {}
        for e in self.entries:
            by.setdefault(e["purpose"], 0)
            by[e["purpose"]] += 1
        return dict(n_entries=len(self.entries), by_purpose=by,
                    n_calibration_sources=len(self.calibration_sources), entries=self.entries)


def threshold_bootstrap(y_cal, s_cal, n_boot: int = 200, seed: int = 0, dev=None) -> dict:
    """NON-GATING: how unstable is the per-pipeline Youden threshold on this slice?

    Resamples calibration sources (one clean view each) with replacement within label
    strata, refits the R1 rule, and reports threshold quantiles.  With ``dev=(y, s)`` it also
    reports the spread of DEV recall/FPR those thresholds would induce — a description of
    calibration noise, never a way to pick a threshold.
    """
    y = np.asarray(y_cal, dtype=int)
    s = np.asarray(s_cal, dtype=float)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    ths = []
    for _ in range(n_boot):
        idx = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        ths.append(youden_threshold(y[idx], s[idx])["threshold"])
    ths = np.asarray(ths)
    out = dict(n_boot=n_boot, threshold_q05=float(np.quantile(ths, .05)), threshold_q50=float(np.median(ths)),
               threshold_q95=float(np.quantile(ths, .95)), recall_grid_step=1.0 / max(1, len(pos)))
    if dev is not None:
        yd, sd = np.asarray(dev[0], dtype=int), np.asarray(dev[1], dtype=float)
        rec = np.array([(sd[yd == 1] >= t).mean() for t in ths])
        fpr = np.array([(sd[yd == 0] >= t).mean() for t in ths])
        out.update(dev_recall_q05=float(np.quantile(rec, .05)), dev_recall_q95=float(np.quantile(rec, .95)),
                   dev_FPR_q05=float(np.quantile(fpr, .05)), dev_FPR_q95=float(np.quantile(fpr, .95)))
    return out


def freeze_pipeline_threshold(cal_rows: Sequence[dict], cal_scores, ledger: AccessLedger | None = None,
                              arm: str = "", seed: int | None = None) -> dict:
    """One frozen threshold per pipeline/seed from the clean calibration views (R1 rule)."""
    rows = list(cal_rows)
    s = np.asarray(cal_scores, dtype=float)
    if len(rows) != len(s):
        raise ValueError("rows/scores length mismatch")
    if any(r.get("split") != "train" for r in rows):
        raise PermissionError("calibration views must be TRAIN rows")
    idx = [i for i, r in enumerate(rows) if r["variant"] == "clean"]
    if not idx:
        raise ValueError("no clean calibration views")
    if ledger is not None:
        ledger.charge(CALIBRATE_PURPOSE, {rows[i]["source_id"] for i in idx}, arm=arm, seed=seed)
    y = np.array([rows[i]["label"] for i in idx], dtype=int)
    t = youden_threshold(y, s[idx])
    t.update(rule="per-pipeline Youden-J on clean calibration views, higher threshold wins ties",
             arm=arm, seed=seed, n_views=len(idx))
    return t
