"""Clean-set metrics, the R1 threshold-freeze rule and the exact R3 feasibility test.

Parity notes
------------
* ``ranking`` reproduces R1 ``scorer.ranking`` (tie-aware trapezoid AUROC, step AP).
* ``youden_threshold`` reproduces R1 ``scorer.freeze_threshold``: candidates are the unique
  scores plus ``nextafter(max, +inf)``; maximise balanced accuracy (== Youden J); the
  higher threshold wins ties.  Prediction rule everywhere is ``score >= threshold``.
* ``score_views`` reproduces the gate-relevant fields of R1 ``scorer.score`` (clean
  recall/FPR/AUROC/AP, marker flip rate over clean/marker pairs, conflict-set FP) without
  the provenance/timing plumbing.

Feasibility (R3 v2 §1, exact)
----------------------------
    max_t Recall_arm(t)  s.t.  FPR_arm(t) <= FPR_A + tol      compared with  Recall_A - tol
The sweep is over *every* distinct operating point (all unique scores and +inf), so ties
can never let the chosen threshold exceed the FPR allowance.  The copied probe helper
``roc_match`` picked an order statistic and could exceed the allowance when negative
scores tie at the cut (float16 features make ties plausible).
"""
from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

import numpy as np

EPS = 1e-12
ACTIVE_SPLITS = ("train", "dev")


def checked_arrays(y, s):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    if y.ndim != 1 or s.shape != y.shape or not len(y):
        raise ValueError("1-D scores and labels of equal, non-zero length required")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("binary labels required")
    if not np.isfinite(s).all():
        raise ValueError("finite scores required")
    return y.astype(int), s


def confusion(y, s, threshold: float) -> dict:
    y, s = checked_arrays(y, s)
    p = s >= threshold
    tp = int(((y == 1) & p).sum())
    fp = int(((y == 0) & p).sum())
    fn = int(((y == 1) & ~p).sum())
    tn = int(((y == 0) & ~p).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, n=len(y),
                recall=None if tp + fn == 0 else tp / (tp + fn),
                FPR=None if fp + tn == 0 else fp / (fp + tn))


def ranking(y, s) -> dict:
    """Tie-aware AUROC (trapezoid over distinct thresholds) and step AP, as R1 scorer."""
    y, s = checked_arrays(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[ends]
    fp = (ends + 1) - tp
    ap = None if not n1 else float(np.sum(np.diff(np.r_[0, tp]) * tp / (ends + 1)) / n1)
    auc = None if not n1 or not n0 else float(np.trapezoid(np.r_[0, tp / n1], np.r_[0, fp / n0]))
    return dict(AUROC=auc, AP=ap)


def operating_points(y, s):
    """All distinct operating points of the rule ``s >= t``.

    Returns (thresholds, tp, fp, n1, n0) with thresholds in *descending* order, starting
    with +inf (predict nothing) and then every unique score.
    """
    y, s = checked_arrays(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    order = np.argsort(-s, kind="stable")
    ys, ss = y[order], s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.r_[0, np.cumsum(ys)[ends]]
    fp = np.r_[0, (ends + 1)] - tp
    thr = np.r_[np.inf, ss[ends]]
    return thr, tp.astype(int), fp.astype(int), n1, n0


def youden_threshold(y, s) -> dict:
    """R1 freeze rule: max balanced accuracy, higher threshold wins ties."""
    y, s = checked_arrays(y, s)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        raise ValueError("both classes required for a threshold")
    order = np.argsort(s, kind="stable")
    ss, y2 = s[order], y[order]
    starts = np.r_[0, np.flatnonzero(np.diff(ss)) + 1]
    removed_pos = np.r_[0, np.cumsum(y2)][starts]
    tp = n1 - removed_pos
    fp = n0 - (starts - removed_pos)
    thresholds = np.r_[ss[starts], np.nextafter(ss[-1], np.inf)]
    ba = np.r_[(tp / n1 + 1 - fp / n0) / 2, 0.5]
    best_ba, threshold = max(zip(ba.tolist(), thresholds.tolist()))
    pred = s >= threshold
    return dict(threshold=float(threshold), balanced_accuracy=float(best_ba),
                youden_j=float(2 * best_ba - 1),
                recall=float(pred[y == 1].mean()), FPR=float(pred[y == 0].mean()),
                n_pos=n1, n_neg=n0,
                scores_sha256=hashlib.sha256(np.asarray(s, dtype="<f8").tobytes()).hexdigest())


def feasibility(y, s_arm, recall_A: float, fpr_A: float, tol: float = 0.02) -> dict:
    """Exact R3 v2 feasibility test on clean views.

    PASS iff  max_t Recall_arm(t) s.t. FPR_arm(t) <= FPR_A + tol   >=   Recall_A - tol.
    Also reports the dual (min FPR s.t. recall >= Recall_A - tol); both describe the same
    feasibility question and must agree on PASS/FAIL.
    """
    thr, tp, fp, n1, n0 = operating_points(y, s_arm)
    if n1 == 0 or n0 == 0:
        raise ValueError("both classes required")
    rec = tp / n1
    fpr = fp / n0
    ok = fpr <= fpr_A + tol + EPS
    i = int(np.flatnonzero(ok)[np.argmax(rec[ok])])  # +inf point is always ok
    # among equal max recall, prefer the lowest FPR (largest threshold)
    best_rec = rec[i]
    cand = np.flatnonzero(ok & (rec == best_rec))
    i = int(cand[np.argmin(fpr[cand])])
    margin = float(rec[i] - recall_A)
    passed = bool(rec[i] >= recall_A - tol - EPS)
    need = rec >= recall_A - tol - EPS
    if need.any():
        j_c = np.flatnonzero(need)
        j = int(j_c[np.argmin(fpr[j_c])])
        dual_fpr = float(fpr[j])
    else:  # unreachable in practice: the lowest threshold gives recall 1
        j, dual_fpr = None, float("inf")
    dual_pass = bool(dual_fpr <= fpr_A + tol + EPS)
    if dual_pass != passed:
        raise AssertionError("primal/dual feasibility disagree — bug")
    return dict(test="max_t Recall(t) s.t. FPR(t) <= FPR_A + tol vs Recall_A - tol",
                recall_A=float(recall_A), FPR_A=float(fpr_A), tol=tol,
                max_recall_at_allowed_fpr=float(best_rec), fpr_at_that_point=float(fpr[i]),
                threshold_at_that_point=float(thr[i]),
                recall_margin=margin, passed=passed,
                dual_min_fpr_at_allowed_recall=dual_fpr,
                dual_fpr_margin=float(fpr_A - dual_fpr),
                n_pos=n1, n_neg=n0)


def _guard_rows(rows: Sequence[dict], allowed: Iterable[str]):
    allowed = set(allowed)
    bad = {r.get("split") for r in rows} - allowed
    if bad:
        raise PermissionError(f"rows from split(s) {sorted(map(str, bad))} refused; allowed {sorted(allowed)}")
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate view ids")


def clean_index(rows) -> np.ndarray:
    return np.array([i for i, r in enumerate(rows) if r["variant"] == "clean"], dtype=int)


def pair_index(rows) -> tuple[np.ndarray, np.ndarray, list]:
    """Row indices of (clean, marker) views per source group, in sorted group order."""
    pairs: dict = {}
    for i, r in enumerate(rows):
        if r["variant"] in ("clean", "marker"):
            g = pairs.setdefault(r["group_id"], {})
            if r["variant"] in g:
                raise ValueError("duplicate pair state")
            g[r["variant"]] = i
    groups = sorted(pairs)
    if any(set(pairs[g]) != {"clean", "marker"} for g in groups):
        raise ValueError("incomplete counterfactual pairs")
    ci = np.array([pairs[g]["clean"] for g in groups], dtype=int)
    mi = np.array([pairs[g]["marker"] for g in groups], dtype=int)
    return ci, mi, groups


def score_views(rows: Sequence[dict], scores, threshold: float, allowed_splits=("dev",)) -> dict:
    """Gate-relevant metrics for one pipeline at one frozen threshold."""
    _guard_rows(rows, allowed_splits)
    y = np.array([r["label"] for r in rows], dtype=int)
    y, s = checked_arrays(y, scores)
    pred = s >= threshold
    out: dict = dict(threshold=float(threshold), n_views=len(rows))
    ci = clean_index(rows)
    out.update(confusion(y[ci], s[ci], threshold))
    out.update(ranking(y[ci], s[ci]))
    c_i, m_i, groups = pair_index(rows)
    if not np.array_equal(y[c_i], y[m_i]):
        raise ValueError("pair label mismatch")
    flips = pred[c_i] != pred[m_i]
    out["marker_flip_rate"] = float(flips.mean())
    out["n_pairs"] = int(len(c_i))
    out["mean_marker_score_change"] = float(np.mean(s[m_i] - s[c_i]))
    for lab in (0, 1):
        sel = y[c_i] == lab
        out[f"marker_flip_rate_label{lab}"] = float(flips[sel].mean()) if sel.any() else None
    good_marked = np.array([i for i, r in enumerate(rows) if r["variant"] == "marker" and r["label"] == 0], dtype=int)
    out["conflict_set_FP"] = int(pred[good_marked].sum()) if len(good_marked) else 0
    out["conflict_set_FPR"] = float(pred[good_marked].mean()) if len(good_marked) else None
    nuis = {}
    for v in ("brightness", "background"):
        idx = np.array([i for i, r in enumerate(rows) if r["variant"] == v], dtype=int)
        if len(idx):
            nuis[v] = confusion(y[idx], s[idx], threshold)
    out["nuisance_by_type"] = nuis
    cats = sorted({r.get("category", "") for r in rows})
    out["per_category_clean"] = {
        c: confusion(y[[i for i in ci if rows[i].get("category", "") == c]],
                     s[[i for i in ci if rows[i].get("category", "") == c]], threshold)
        for c in cats if any(rows[i].get("category", "") == c for i in ci)}
    return out


def paired_units(rows: Sequence[dict], scores, threshold: float) -> dict:
    """Per-source-group indicator arrays (the independence unit) for significance tests.

    Returns arrays aligned on sorted group ids:
      flip[g]          1 if clean/marker predictions disagree
      clean_pred[g]    prediction on the clean view;  label[g]
      conflict[g]      prediction on the marker view if label 0 else nan
      clean_score[g], marker_score[g]
    """
    y = np.array([r["label"] for r in rows], dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = s >= threshold
    c_i, m_i, groups = pair_index(rows)
    lab = y[c_i]
    conflict = np.where(lab == 0, pred[m_i].astype(float), np.nan)
    return dict(groups=groups, label=lab, flip=(pred[c_i] != pred[m_i]).astype(float),
                clean_pred=pred[c_i].astype(float), conflict=conflict,
                clean_score=s[c_i], marker_score=s[m_i])
