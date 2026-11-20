import numpy as np
import pytest

from vision_lab.wsp import metrics as M


def test_confusion_known_answer():
    c = M.confusion([0, 0, 1, 1, 1], [0.1, 0.6, 0.4, 0.7, 0.9], 0.5)
    assert (c["TP"], c["FP"], c["FN"], c["TN"]) == (2, 1, 1, 1)
    assert c["recall"] == pytest.approx(2 / 3) and c["FPR"] == pytest.approx(0.5)


def test_ranking_known_answers():
    r = M.ranking([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert r["AUROC"] == pytest.approx(0.75)
    assert r["AP"] == pytest.approx((1 + 2 / 3) / 2)
    assert M.ranking([0, 1], [1.0, 1.0])["AUROC"] == pytest.approx(0.5)   # ties -> diagonal


def test_youden_known_answer_and_tiebreak():
    assert M.youden_threshold([0, 0, 1, 1], [1, 2, 3, 4])["threshold"] == 3
    t = M.youden_threshold([0, 1], [0.0, 0.0])          # BA .5 at 0 and at nextafter(0)
    assert t["threshold"] == np.nextafter(0.0, np.inf)   # higher threshold wins ties


def test_youden_and_ranking_parity_with_r1_scorer(r1_scorer):
    rng = np.random.default_rng(1)
    for _ in range(25):
        n = int(rng.integers(5, 60))
        y = rng.integers(0, 2, n)
        y[0], y[1] = 0, 1
        s = np.round(rng.normal(size=n), int(rng.integers(0, 3)))   # many ties
        rows = [dict(id=str(i), split="dev", variant="clean", label=int(y[i])) for i in range(n)]
        ref = r1_scorer.freeze_threshold(rows, s, "h")
        assert M.youden_threshold(y, s)["threshold"] == ref["threshold"]
        assert M.ranking(y, s) == r1_scorer.ranking(y, s)


def brute_feasibility(y, s, rec_A, fpr_A, tol):
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    best = 0.0
    for t in np.r_[np.inf, np.unique(s)]:
        p = s >= t
        fpr = p[y == 0].mean()
        rec = p[y == 1].mean()
        if fpr <= fpr_A + tol + 1e-12:
            best = max(best, rec)
    return best


def test_feasibility_matches_brute_force_with_ties():
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(6, 80))
        y = rng.integers(0, 2, n)
        y[0], y[1] = 0, 1
        s = np.round(rng.normal(size=n) + y, int(rng.integers(0, 2)))
        rec_A, fpr_A = rng.uniform(0.3, 0.9), rng.uniform(0.0, 0.4)
        f = M.feasibility(y, s, rec_A, fpr_A)
        assert f["max_recall_at_allowed_fpr"] == pytest.approx(brute_feasibility(y, s, rec_A, fpr_A, .02))
        assert f["fpr_at_that_point"] <= fpr_A + .02 + 1e-12
        assert f["passed"] == (f["max_recall_at_allowed_fpr"] >= rec_A - .02 - 1e-12)


def test_feasibility_known_answer():
    # 10 negatives, 10 positives; A at recall .7 / FPR .1 -> allowance .12 -> 1 FP allowed
    neg = np.arange(10) / 10.0           # 0.0 .. 0.9
    pos = np.array([0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05])
    y = np.r_[np.zeros(10), np.ones(10)].astype(int)
    s = np.r_[neg, pos]
    f = M.feasibility(y, s, recall_A=0.7, fpr_A=0.1)
    # lowest admissible threshold 0.85 admits 1 FP (0.9); positives >= .85: .95, .85 -> recall .2
    assert f["max_recall_at_allowed_fpr"] == pytest.approx(0.2)
    assert f["threshold_at_that_point"] == pytest.approx(0.85)
    assert not f["passed"] and f["recall_margin"] == pytest.approx(-0.5)


def test_order_statistic_shortcut_violates_constraint_on_ties(r1_scorer):
    """Documents why we sweep: the copied roc_match picks neg[n0-m] and can overshoot."""
    import importlib.util

    from conftest import R1_COPY
    spec = importlib.util.spec_from_file_location("r1copy_probe_lib", R1_COPY / "probe_lib.py")
    lib = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(lib)
    except Exception:
        pytest.skip("probe_lib not importable")
    neg = np.array([0.0] * 8 + [0.9, 0.9])     # top two negatives tie
    pos = np.array([0.95, 0.9, 0.8, 0.7])
    y = np.r_[np.zeros(10), np.ones(4)].astype(int)
    s = np.r_[neg, pos]
    base = np.r_[np.array([1.0] + [0.0] * 9), pos]   # A: one negative above 0.92 -> FPR_A = .1
    clean_idx = np.arange(len(s))
    rm = lib.roc_match(s, base, clean_idx, y, base_thr=0.92)
    assert rm["base_FPR"] == pytest.approx(0.1)
    # allowance .12 -> m = 1 -> threshold = 2nd-largest negative (0.9), tied with the largest
    assert rm["fpr_at_base_fpr"] == pytest.approx(0.2)        # > .12 allowance: overshoot
    assert rm["recall_at_base_fpr"] == pytest.approx(0.5)     # overstated recall
    ours = M.feasibility(y, s, rm["base_recall"], rm["base_FPR"])
    assert ours["fpr_at_that_point"] <= 0.12 + 1e-12
    assert ours["max_recall_at_allowed_fpr"] == pytest.approx(0.25)   # only 0.95 above the tie


def _views(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows, s = [], []
    for g in range(n):
        lab = g % 3 == 0
        base = rng.normal() + 2 * lab
        for v, shift in (("clean", 0.0), ("marker", 3.0), ("brightness", 0.2)):
            rows.append(dict(id=f"{g}{v}", group_id=f"g{g}", split="dev", label=int(lab), variant=v, category="c"))
            s.append(base + shift)
    return rows, np.array(s)


def test_score_views_flip_conflict_and_holdout_refusal():
    rows, s = _views()
    m = M.score_views(rows, s, threshold=1.5)
    ci, mi, _ = M.pair_index(rows)
    pred = s >= 1.5
    assert m["marker_flip_rate"] == pytest.approx(np.mean(pred[ci] != pred[mi]))
    good_marked = [i for i, r in enumerate(rows) if r["variant"] == "marker" and r["label"] == 0]
    assert m["conflict_set_FP"] == int(pred[good_marked].sum())
    rows[0] = dict(rows[0], split="holdout")
    with pytest.raises(PermissionError):
        M.score_views(rows, s, 1.5)


def test_score_views_parity_with_r1_scorer(r1_scorer):
    rows, s = _views(30, seed=3)
    rows = [dict(r, sets=["clean"] if r["variant"] == "clean" else []) for r in rows]
    ref = r1_scorer.score(rows, s, dict(frozen=True, threshold=1.0), timing=dict(n_images=len(rows), elapsed_seconds=1))
    ours = M.score_views(rows, s, 1.0)
    for k in ("recall", "FPR", "AUROC", "AP", "marker_flip_rate", "conflict_set_FP"):
        assert ours[k] == pytest.approx(ref[k]), k
