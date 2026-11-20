"""End-to-end runner test on a synthetic R1-shaped tree (tiny encoder, no models)."""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_r3", ROOT / "run_r3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dry(runner, tmp_path_factory):
    out = tmp_path_factory.mktemp("r3dry")
    runner.main(["dry-run", "--out", str(out)])
    return out


def test_dry_run_produces_all_attempts(dry):
    pf = json.loads((dry / "PREFLIGHT.json").read_text())
    assert pf["status"] == "PASS", pf["checks"]
    for slot in ("slot1", "slot2", "slot3"):
        d = json.loads((dry / "attempts" / f"{slot}_k16.json").read_text())
        assert d["report"]["status"] in ("PASS", "NULL_OR_NEGATIVE")
        assert d["report"]["rules"]["per_pipeline"]["noop_integrity"]
        assert "HOLDOUT_x" not in json.dumps(d)
    assert (dry / "R3_RESULTS.md").exists()
    led = json.loads((dry / "ATTEMPT_LEDGER.json").read_text())
    assert set(led["attempts"]) == {"slot1|16", "slot2|16", "slot3|16"}


def test_added_dinov2_arm_runs_separately_with_its_own_files(dry):
    pf = json.loads((dry / "PREFLIGHT_dinov2.json").read_text())
    assert pf["status"] == "PASS" and pf["arm"] == "dinov2", pf["checks"]
    assert pf["checks"]["fill_rgb_equals_encoder_mean"]["detail"]["encoder"] == [124, 116, 104]
    assert "alignment_pooled" not in pf["checks"] and "encoder_features_finite" in pf["checks"]
    assert json.loads((dry / "PREFLIGHT.json").read_text())["arm"] == "siglip"
    for slot in ("slot1", "slot2", "slot3"):
        d = json.loads((dry / "attempts_dinov2" / f"{slot}_k16.json").read_text())
        assert d["arm"] == "dinov2" and d["report"]["rules"]["per_pipeline"]["noop_integrity"]
    led = json.loads((dry / "ATTEMPT_LEDGER_dinov2.json").read_text())
    assert set(led["attempts"]) == {"slot1|16", "slot2|16", "slot3|16"}
    assert led["recipes"] == json.loads((dry / "ATTEMPT_LEDGER.json").read_text())["recipes"]    # same recipes, own ledger
    assert "dinov2" in (dry / "R3_RESULTS_dinov2.md").read_text() and (dry / "features_dinov2" / "identity").is_dir()
    plan = json.loads((dry / "R3_PLAN.json").read_text())
    assert plan["arms"]["dinov2"]["role"].startswith("ADDED ARM")


def test_undeclared_arm_and_cross_arm_preflight_refused(runner, dry):
    tree, views = dry / "fake_r1_tree", dry / "fake_r1_tree/views"
    with pytest.raises(SystemExit):
        runner.main(["extract", "--out", str(dry), "--wsp-root", str(tree), "--views-dir", str(views), "--arm", "clip"])


def test_slot1_masking_removes_planted_marker_in_dry_run(dry):
    d = json.loads((dry / "attempts" / "slot1_k16.json").read_text())
    c = d["report"]["rules"]["per_pipeline"]["candidates"]["slot1_sam_meanfill"]
    assert c["gates"]["bars"]["1a_marker_relative_reduction"] == "PASS"
    for s, seed_rep in d["per_seed"].items():
        arm = seed_rep["rules"]["per_pipeline"]["arms"]["slot1_sam_meanfill"]
        assert arm["metrics"]["marker_flip_rate"] == 0.0        # marker pixels cannot reach features


def test_completed_attempt_cannot_be_rerun(runner, dry):
    tree, views = dry / "fake_r1_tree", dry / "fake_r1_tree/views"
    with pytest.raises(Exception):
        runner.main(["attempt", "--out", str(dry), "--wsp-root", str(tree), "--views-dir", str(views),
                     "--slot", "slot1", "--k", "16"])


def test_plan_is_immutable(runner, dry):
    with pytest.raises(SystemExit):
        runner.main(["plan", "--out", str(dry)])                  # differs from the dry-run plan
