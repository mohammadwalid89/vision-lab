import numpy as np
import pytest

from vision_lab.wsp import harness as H
from vision_lab.wsp.calibration import AccessLedger, partition_sources
from vision_lab.wsp.edits import random_deletion_keep
from vision_lab.wsp.heads import (
    Standardizer,
    linear_logits,
    mlp_forward,
    retrain_linear_fresh,
    retrain_linear_warm,
    train_linear_A,
    train_mlp_A,
)
from vision_lab.wsp.ledger import (
    AttemptLedger,
    LedgerError,
    ResourceGuard,
    ResourceStop,
)
from vision_lab.wsp.slots.hidden_deletion import (
    DeletionRecipe,
    apply_keep,
    build_slot3_arms,
    learn_deletion_mask,
    select_preserve_sources,
)
from vision_lab.wsp.synthetic import feature_world

FAST = DeletionRecipe(epochs=40, preserve_n=200)


@pytest.fixture(scope="module")
def mlp_setup():
    rows, X, _ = feature_world(n_train=300, n_dev=60, d=16, seed=1)
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    w = np.array([rows[i]["confounded_train"] for i in tr])
    y = np.array([rows[i]["label"] for i in tr])
    A, _ = train_mlp_A(X[tr], y, seed=5, hidden=8, steps=150, lr=1e-2, sample_weight=w)
    k_idx = tr[:32]
    pres = [i for i in tr[100:] if rows[i]["variant"] == "clean"][:200]
    return rows, X, A, X[k_idx], y[:32], X[pres]


def test_slot3_exact_budget_and_determinism(mlp_setup):
    _, _, A, xk, yk, xp = mlp_setup
    k1, info = learn_deletion_mask(A, xk, yk, xp, FAST, seed=1)
    k2, _ = learn_deletion_mask(A, xk, yk, xp, FAST, seed=1)
    assert int((~k1).sum()) == info["r"] == round(0.05 * A["W1"].size)
    assert np.array_equal(k1, k2)
    assert np.array_equal(apply_keep(A, k1)["b1"], A["b1"])            # only W1 entries change


def test_slot3_preservation_term_preserves_clean_scores(mlp_setup):
    """Known-direction check: with the preservation term on, clean logits on P move less
    than under (a) the same recipe with lam_pres = 0 and (b) random deletion of equal r."""
    _, _, A, xk, yk, xp = mlp_setup
    zA = mlp_forward(A, xp)
    rec = DeletionRecipe(epochs=60, fraction=0.10, lam_pres=5.0)
    k_on, info = learn_deletion_mask(A, xk, yk, xp, rec, seed=2)
    k_off, _ = learn_deletion_mask(A, xk, yk, xp, rec, seed=2, lam_pres=0.0)
    k_rand = random_deletion_keep(A["W1"].shape, info["r"], seed=9)
    mse = {n: float(np.mean((mlp_forward(apply_keep(A, k), xp) - zA) ** 2))
           for n, k in (("on", k_on), ("off", k_off), ("rand", k_rand))}
    assert mse["on"] <= mse["off"] and mse["on"] <= mse["rand"], mse


def test_preserve_set_excludes_calibration():
    ids = [f"s{i}" for i in range(50)]
    sel = select_preserve_sources(ids, 10, "ns|", calibration=frozenset(ids[:25]))
    assert len(sel) == 10 and not set(sel) & set(ids[:25])
    assert sel == select_preserve_sources(ids, 10, "ns|", calibration=frozenset(ids[:25]))


def test_build_slot3_arms_roles(mlp_setup):
    _, _, A, xk, yk, xp = mlp_setup
    arms = build_slot3_arms(A, xk, yk, xp, seed=3, recipe=FAST)
    roles = {n: a["role"] for n, a in arms.items()}
    assert roles["slot3_delmask"] == "candidate"
    assert sum(r == "placebo" for r in roles.values()) == 3
    assert sum(r == "noop" for r in roles.values()) == 6
    assert {n for n, r in roles.items() if r == "retrainer"} == {"warm@k", "fresh@k", "warm_preserve@k"}


# ------------------------------------------------------------------ end-to-end harness
SEEDS = (11, 12, 13)


def _linear_world():
    rows, X, info = feature_world(n_train=900, n_dev=400, d=12, seed=4)
    train_src = [dict(source_id=r["source_id"], group_id=r["group_id"], split="train", label=r["label"],
                      category=r["category"]) for r in rows if r["split"] == "train" and r["variant"] == "clean"]
    part = partition_sources(train_src, SEEDS, k_largest=32, cal_n={0: 60, 1: 30}, min_per_label=40)
    return rows, X, info, train_src, part


def _pipelines(rows, X, info, part, seed, broken_noop=False, candidate="oracle"):
    cal = set(part["calibration_source_ids"])
    tr = [i for i, r in enumerate(rows) if r["split"] == "train"]
    fit = [i for i in tr if rows[i]["source_id"] not in cal]
    cal_i = [i for i in tr if rows[i]["source_id"] in cal]
    dev_i = [i for i, r in enumerate(rows) if r["split"] == "dev"]
    st = Standardizer().fit(X[fit])
    Z = st.transform(X).astype(np.float64)
    yf = np.array([rows[i]["label"] for i in fit])
    wf = np.array([rows[i]["confounded_train"] for i in fit])
    thA, _ = train_linear_A(Z[fit], yf, seed, steps=300, lr=0.3, sample_weight=wf)
    k_src = sorted({rows[i]["source_id"] for i in fit})[:32]
    k_i = [i for i in fit if rows[i]["source_id"] in set(k_src)]
    yk = np.array([rows[i]["label"] for i in k_i])
    if candidate == "oracle":            # remove the marker input direction (index 1)
        th_c = thA.copy()
        th_c[1] = 0.0
    else:                                # candidate identical to A
        th_c = thA.copy()
    th_w, _ = retrain_linear_warm(Z[k_i], yk, thA, steps=300)
    th_f, _ = retrain_linear_fresh(Z[k_i], yk, seed, steps=300)
    pipes = {}

    def add(name, role, th, **kw):
        s = linear_logits(Z, th)
        pipes[name] = H.PipelineScores(role, s[cal_i], s[dev_i], **kw)

    add("A", "A", thA)
    add("cand", "candidate", th_c)
    for d in range(3):
        th_p = thA.copy()
        th_p[3 + d] = 0.0  # delete a noise coordinate
        add(f"placebo{d}", "placebo", th_p)
    add("warm@k", "retrainer", th_w)
    add("fresh@k", "retrainer", th_f)
    add("noop_identity", "noop", thA.copy(), noop_ref="A")
    if broken_noop:
        th_b = thA.copy()
        th_b[0] *= 1.5
        add("noop_broken", "noop", th_b, noop_ref="A")
    return [rows[i] for i in cal_i], [rows[i] for i in dev_i], pipes


def _run(broken=False, candidate="oracle"):
    rows, X, info, train_src, part = _linear_world()
    spec = H.AttemptSpec(slot="toy", k=32, seeds=SEEDS, gate_families=("CORE", "G5"))
    led = AccessLedger(frozenset(part["calibration_source_ids"]), {})
    per = {}
    for s in SEEDS:
        cal_rows, dev_rows, pipes = _pipelines(rows, X, info, part, s, broken, candidate)
        per[s] = H.evaluate_seed(spec, s, cal_rows, dev_rows, pipes, led)
    return H.evaluate_attempt(spec, per), per


def test_harness_oracle_edit_removes_shortcut():
    rep, per = _run()
    c = rep["rules"]["per_pipeline"]["candidates"]["cand"]
    assert rep["rules"]["per_pipeline"]["noop_integrity"]
    assert c["gates"]["bars"]["1a_marker_relative_reduction"] == "PASS"
    assert c["gates"]["bars"]["6_same_direction_all_seeds"] == "PASS"
    calls = c["effect_calls"][str(SEEDS[0])]
    assert calls["marker_flip_rate"] == "DOWN"
    assert rep["status"] in ("PASS", "NULL_OR_NEGATIVE")
    assert set(rep["rules"]) == {"per_pipeline", "shared_A"}


def test_harness_identity_candidate_is_null_but_feasible():
    rep, _ = _run(candidate="identity")
    c = rep["rules"]["per_pipeline"]["candidates"]["cand"]
    assert c["feasibility_all_seeds"] and c["feasibility_worst_recall_margin"] >= 0.0
    assert c["gates"]["bars"]["1a_marker_relative_reduction"] == "FAIL"
    assert all(v == "SAME" for v in c["effect_calls"][str(SEEDS[0])].values())
    assert rep["status"] == "NULL_OR_NEGATIVE"


def test_harness_broken_noop_makes_attempt_inconclusive():
    rep, per = _run(broken=True)
    assert not rep["rules"]["per_pipeline"]["noop_integrity"]
    assert rep["status"] == "INCONCLUSIVE"
    nb = per[SEEDS[0]]["rules"]["per_pipeline"]["noops"]["noop_broken"]
    assert nb["max_abs_score_change_dev"] > 0 and not nb["ok"]


def test_harness_refuses_calibration_leak_and_holdout():
    rows, X, info, train_src, part = _linear_world()
    cal_rows, dev_rows, pipes = _pipelines(rows, X, info, part, SEEDS[0])
    spec = H.AttemptSpec(slot="toy", k=32, seeds=SEEDS)
    led = AccessLedger(frozenset(), {})                 # calibration slice declared empty
    with pytest.raises(PermissionError):
        H.evaluate_seed(spec, SEEDS[0], cal_rows, dev_rows, pipes, led)
    bad = [dict(r, split="holdout") if i == 0 else r for i, r in enumerate(dev_rows)]
    with pytest.raises(PermissionError):
        H.evaluate_seed(spec, SEEDS[0], cal_rows, bad, pipes)


def test_attempt_ledger_and_resource_guard(tmp_path):
    led = AttemptLedger(path=tmp_path / "ledger.json")
    led.register_recipe("slot3", dict(a=1))
    with pytest.raises(LedgerError):
        led.register_recipe("slot3", dict(a=2))                         # no fallbacks
    with pytest.raises(LedgerError):
        led.begin("slot3", 16, {})                                      # undeclared k
    led.begin("slot3", 32, dict(x=1))
    with pytest.raises(LedgerError):
        led.begin("slot3", 32, dict(x=2))                               # replacement attempt
    led.finish("slot3", 32, "NULL_OR_NEGATIVE")
    with pytest.raises(LedgerError):
        led.begin("slot3", 32, dict(x=1))                               # completed
    assert AttemptLedger.load(tmp_path / "ledger.json").primary_conclusion("slot3") == "NULL_OR_NEGATIVE"
    import os
    g = ResourceGuard(tmp_path, max_cores=os.cpu_count())
    assert g.check("t")["footprint_bytes"] > 0
    with pytest.raises(ResourceStop):
        ResourceGuard(tmp_path, footprint_cap_bytes=1, max_cores=os.cpu_count()).check("t")
    with pytest.raises(ResourceStop):
        ResourceGuard(tmp_path, max_cores=0).check("t")
