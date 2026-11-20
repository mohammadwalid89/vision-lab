import numpy as np
import pytest

from vision_lab.wsp import harness as H
from vision_lab.wsp.calibration import AccessLedger, partition_sources
from vision_lab.wsp.ordering import k_lists
from vision_lab.wsp.pipelines import SlotData, slot_pipelines
from vision_lab.wsp.slots.hidden_deletion import DeletionRecipe
from vision_lab.wsp.synthetic import feature_world

SEEDS = (1, 2, 3)


@pytest.fixture(scope="module")
def setup():
    rows, X, info = feature_world(n_train=400, n_dev=150, d=16, seed=9)
    rng = np.random.default_rng(0)
    Xm = X.copy()
    Xm[:, 1] = 0.0  # masking removes the marker coordinate
    def tdict(Z):
        return dict(pooled=Z[:, :8], mean_patch=Z[:, 8:], object=Z[:, :8] + Z[:, 8:],
                    perm_1=Z[:, :8], perm_2=Z[:, 8:], perm_3=Z[:, :8] - Z[:, 8:], uniform=Z[:, :8] * 0.5)
    feats = {"identity": tdict(X), "keep_all": tdict(X.copy()), "sam_meanfill": tdict(Xm),
             "padding_only": tdict(Xm)}
    for d in (1, 2, 3):
        feats[f"random_area_{d}"] = tdict(X + rng.normal(0, 0.5, X.shape))
    train_src = [dict(source_id=r["source_id"], group_id=r["group_id"], split="train", label=r["label"],
                      category=r["category"]) for r in rows if r["split"] == "train" and r["variant"] == "clean"]
    part = partition_sources(train_src, SEEDS, k_largest=16, cal_n={0: 30, 1: 15}, min_per_label=20)
    kl = k_lists(train_src, SEEDS, (16,), min_per_label=20)
    D = SlotData(rows, frozenset(part["calibration_source_ids"]), kl, 16)
    acc = AccessLedger(frozenset(part["calibration_source_ids"]), kl)
    return feats, D, acc


@pytest.mark.parametrize("slot", ["slot1", "slot2", "slot3"])
def test_slot_pipelines_end_to_end(setup, slot):
    feats, D, acc = setup
    spec = H.AttemptSpec(slot=slot, k=16, seeds=SEEDS, gate_families=("CORE", "G5"))
    rec = DeletionRecipe(epochs=20, preserve_n=100)
    per = {}
    for s in SEEDS:
        pipes, comp = slot_pipelines(slot, s, feats, D, acc, rec)
        roles = [p.role for p in pipes.values()]
        assert roles.count("A") == 1 and roles.count("candidate") == 1 and roles.count("placebo") == 3
        assert roles.count("retrainer") >= 2 and roles.count("noop") >= 1
        per[s] = H.evaluate_seed(spec, s, D.cal_rows, D.dev_rows, pipes, acc)
    rep = H.evaluate_attempt(spec, per)
    assert rep["rules"]["per_pipeline"]["noop_integrity"], slot
    assert rep["status"] in ("PASS", "NULL_OR_NEGATIVE")
    if slot == "slot1":                                   # masking the marker coordinate removes the shortcut
        c = rep["rules"]["per_pipeline"]["candidates"]["slot1_sam_meanfill"]
        assert c["gates"]["bars"]["1a_marker_relative_reduction"] == "PASS"
