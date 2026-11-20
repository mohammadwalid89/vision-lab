import numpy as np
import pytest

from vision_lab.wsp.calibration import (
    AccessLedger,
    freeze_pipeline_threshold,
    partition_sources,
)
from vision_lab.wsp.ordering import k_lists, ordered_sources

SEEDS = (20261112, 20261113, 20261114)


def train_rows(n_good=1500, n_def=400, cats=("a", "b", "c", "d")):
    rows = []
    for i in range(n_good + n_def):
        lab = int(i >= n_good)
        rows.append(dict(source_id=f"s{i:05d}", group_id=f"g{i:05d}", split="train", label=lab,
                         category=cats[(i * 7) % len(cats)]))
    return rows


def test_ordering_parity_and_prefix(r1_ordered):
    rows = train_rows()
    for s in SEEDS:
        ours = [r["source_id"] for r in ordered_sources(rows, s)]
        ref = [r["source_id"] for r in r1_ordered.ordered_sources(rows, s)]
        assert ours == ref
        assert [r["label"] for r in ordered_sources(rows, s)[:8]] == [0, 1] * 4
    kl = k_lists(rows, SEEDS, (8, 32, 128))
    assert all(kl[s][128][:32] == kl[s][32] for s in SEEDS)


def test_partition_rules():
    rows = train_rows()
    p = partition_sources(rows, SEEDS, k_largest=128)
    cal, fit, excl = set(p["calibration_source_ids"]), set(p["fitting_source_ids"]), set(p["excluded_k_union"])
    assert len(cal) == 512 and not cal & excl and not cal & fit
    assert cal | fit == {r["source_id"] for r in rows}
    lab = {r["source_id"]: r["label"] for r in rows}
    assert sum(lab[s] for s in cal) == 128
    assert partition_sources(rows, SEEDS)["partition_digest"] == p["partition_digest"]   # deterministic
    # quotas proportional to eligible pool per category (largest remainder sums exactly)
    for q in p["quotas"].values():
        assert sum(q["quota_by_category"].values()) == q["requested"] == q["achieved"]


def test_ledger_enforces_information_budget():
    rows = train_rows()
    p = partition_sources(rows, SEEDS)
    kl = k_lists(rows, SEEDS, (8, 32, 128))
    led = AccessLedger(frozenset(p["calibration_source_ids"]), kl)
    led.charge("fit_A", p["fitting_source_ids"], arm="A")
    led.charge("adapt_pairs", kl[SEEDS[0]][32], arm="cand", seed=SEEDS[0], k=32)
    with pytest.raises(PermissionError):
        led.charge("retrain", p["calibration_source_ids"][:1], arm="warm@k")
    with pytest.raises(PermissionError):
        led.charge("adapt_pairs", kl[SEEDS[0]][128], arm="cand", seed=SEEDS[0], k=32)
    with pytest.raises(PermissionError):
        led.charge("calibrate_threshold", p["fitting_source_ids"][:3])
    with pytest.raises(ValueError):
        led.charge("peek_at_dev", ["x"])
    assert led.report()["by_purpose"] == {"fit_A": 1, "adapt_pairs": 1}


def test_threshold_bootstrap_is_descriptive_and_deterministic():
    from vision_lab.wsp.calibration import threshold_bootstrap
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(384), np.ones(128)].astype(int)
    s = rng.normal(size=512) + 1.5 * y
    a = threshold_bootstrap(y, s, n_boot=50, seed=1, dev=(y, s))
    assert a == threshold_bootstrap(y, s, n_boot=50, seed=1, dev=(y, s))
    assert a["threshold_q05"] <= a["threshold_q50"] <= a["threshold_q95"]
    assert a["recall_grid_step"] == pytest.approx(1 / 128)


def test_freeze_threshold_uses_clean_train_calibration_views_only():
    rows = [dict(source_id=f"c{i}", split="train", variant=v, label=i % 2) for i in range(10) for v in ("clean", "marker")]
    s = np.array([float(r["label"]) + (5.0 if r["variant"] == "marker" else 0.0) for r in rows])
    t = freeze_pipeline_threshold(rows, s)
    assert t["threshold"] == 1.0 and t["n_views"] == 10          # marker views ignored
    with pytest.raises(PermissionError):
        freeze_pipeline_threshold([dict(rows[0], split="dev")], s[:1])
