"""DINOv2-B/14-reg4 arm (added 2026-11-17): unit + known-answer tests with the real local checkpoint.

Independent checks (not tautologies): constants are compared with timm's own pretrained_cfg, features with a
direct timm forward_features / forward_head call, and the leak-proof property with a global-attention model.
The pinned numbers are REGRESSION pins (same checkpoint, same preprocessing, this CPU stack), not external truth.
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from vision_lab.wsp.encoders import DINO_SNAP, SIGLIP_PATH, DinoV2Encoder, extract
from vision_lab.wsp.r1tree import sha256_file
from vision_lab.wsp.slots import input_mask as IM
from vision_lab.wsp.synthetic import image_world

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(DINO_SNAP is None or not DINO_SNAP.exists(),
                                reason="set VISION_DINOV2_FILE to the local DINOv2 checkpoint")


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location("run_r3_dino", ROOT / "run_r3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def enc():
    return DinoV2Encoder()


def _gradient(size=384):
    yy, xx = np.mgrid[0:size, 0:size]
    return np.stack([(xx * 255 // (size - 1)), (yy * 255 // (size - 1)), ((xx + yy) * 255 // (2 * size - 2))], -1).astype(np.uint8)


def test_output_keys_and_shapes_match_siglip_contract(enc):
    o = enc([_gradient(), _gradient()[::-1].copy()])
    assert {"tokens", "pooled", "mean_patch"} <= set(o)
    assert o["tokens"].shape == (2, 256, 768) and enc.grid == 16 and enc.img == 224
    assert o["pooled"].shape == (2, 768) and o["mean_patch"].shape == (2, 768)
    assert all(np.isfinite(a).all() for a in o.values())
    np.testing.assert_allclose(o["mean_patch"], o["tokens"].mean(1), atol=1e-12)


def test_constants_equal_timm_config_and_fill_is_encoder_mean(enc):
    cfg = enc.model.pretrained_cfg
    assert tuple(cfg["mean"]) == enc.mean and tuple(cfg["std"]) == enc.std
    assert [int(v) for v in np.round(np.asarray(enc.mean) * 255.0)] == [124, 116, 104]
    # a view filled with that colour is ~0 after normalisation (the property "encoder_mean" fill relies on)
    z = (np.array([124, 116, 104]) / 255.0 - np.asarray(enc.mean)) / np.asarray(enc.std)
    assert np.abs(z).max() < 0.01


def test_matches_direct_timm_forward_and_registers_are_dropped(enc):
    from PIL import Image
    img = _gradient()
    o = enc([img])
    im = Image.fromarray(img).resize((224, 224), Image.Resampling.BICUBIC)
    x = (np.asarray(im, np.float32) / 255.0 - np.asarray(enc.model.pretrained_cfg["mean"], np.float32)) \
        / np.asarray(enc.model.pretrained_cfg["std"], np.float32)
    x = enc.torch.tensor(x.transpose(2, 0, 1)[None])
    with enc.torch.no_grad():
        f = enc.model.forward_features(x)
        head = enc.model.forward_head(f, pre_logits=True)
    assert f.shape[1] == 1 + 4 + 256 and enc.n_prefix == 5                 # CLS + 4 registers + 16x16 patches
    # atol 2e-4: float32 op-order noise between this re-implemented preprocessing and the encoder's (observed 3e-5);
    # a registers/CLS mix-up would differ by O(1)
    np.testing.assert_allclose(o["tokens"][0], f[0, 5:].double().numpy(), atol=2e-4)   # registers excluded
    np.testing.assert_allclose(o["pooled"][0], head[0].double().numpy(), atol=2e-4)    # pooled == CLS via timm's head path
    np.testing.assert_allclose(o["pooled"][0], f[0, 0].double().numpy(), atol=2e-4)


def test_deterministic_and_batch_invariant(enc):
    a, b = _gradient(), _gradient()[:, ::-1].copy()
    o1, o2 = enc([a, b]), enc([a, b])
    assert all(np.array_equal(o1[k], o2[k]) for k in o1)
    single = enc([a])
    np.testing.assert_allclose(single["pooled"][0], o1["pooled"][0], atol=1e-5)
    np.testing.assert_allclose(single["tokens"][0], o1["tokens"][0], atol=1e-5)


def test_regression_pin_on_arithmetic_image(enc):
    o = enc([_gradient()])
    np.testing.assert_allclose(o["pooled"][0, :6], [0.225532, -0.496271, -0.453353, 0.549037, 1.556556, 0.622509], atol=2e-3)
    np.testing.assert_allclose(o["mean_patch"][0, :6], [0.746130, -0.920580, -0.738628, 0.196813, 0.346389, 0.112783], atol=2e-3)
    assert np.linalg.norm(o["pooled"][0]) == pytest.approx(24.8061, rel=1e-3)
    assert np.linalg.norm(o["mean_patch"][0]) == pytest.approx(22.9019, rel=1e-3)
    np.testing.assert_allclose(o["tokens"][0, 0, :4], [0.816338, -1.146675, -1.274969, -0.019099], atol=2e-3)
    np.testing.assert_allclose(o["tokens"][0, 255, :4], [-0.432051, -0.582313, -0.423867, 0.449720], atol=2e-3)


def test_masking_is_leak_proof_for_global_attention_dinov2(enc):
    """Marked and clean views become identical after the arm's mean-fill, so DINOv2 features are identical,
    while without masking the marker changes the features (the actuator is real)."""
    w = image_world(n=3, size=32, seed=3)
    tf = IM.build_transforms(n_random=1, encoder_mean=tuple(enc.mean))
    meta = [dict(source_id=x["source_id"], object_mask=x["object_mask"], content_box=x["content_box"]) for x in w]
    clean = [x["image_clean"] for x in w]
    mark = [x["image_marker"] for x in w]
    fc = enc([tf["sam_meanfill"](im, m) for im, m in zip(clean, meta)])
    fm = enc([tf["sam_meanfill"](im, m) for im, m in zip(mark, meta)])
    assert all(np.array_equal(fc[k], fm[k]) for k in ("pooled", "mean_patch", "tokens"))
    uc, um = enc(clean), enc(mark)
    assert np.abs(uc["pooled"] - um["pooled"]).max() > 1e-3                 # unmasked: marker is visible to the encoder
    # the fill colour really is DINOv2's mean, not SigLIP's 128
    m0 = tf["sam_meanfill"](clean[0], meta[0])
    assert tuple(m0[~meta[0]["object_mask"]][0]) == (124, 116, 104)


def test_extract_integration_with_16x16_grid(enc):
    w = image_world(n=4, size=32, seed=4)
    tf = IM.build_transforms(n_random=1, encoder_mean=tuple(enc.mean))
    items = [(x["image_clean"], dict(source_id=x["source_id"], object_mask=x["object_mask"], content_box=x["content_box"])) for x in w]
    res = extract(enc, items, {"identity": tf["identity"], "sam_meanfill": tf["sam_meanfill"]}, batch=2)
    r = res["identity"]
    assert r["pooled"].shape == (4, 768) and r["object"].shape == (4, 768) and r["uniform"].shape == (4, 768)
    assert r["weights"].shape == (4, 256) and np.allclose(r["weights"].sum(1), 1.0)
    assert {"perm_1", "perm_2", "perm_3"} <= set(r)
    assert np.isfinite(res["sam_meanfill"]["pooled"]).all()


def test_make_encoder_and_plan_declare_the_added_arm(runner):
    e = runner.make_encoder("dinov2")
    assert isinstance(e, DinoV2Encoder)
    arms = runner.DEFAULT_PLAN["arms"]
    assert set(arms) == {"siglip", "dinov2"}
    d, s = arms["dinov2"], arms["siglip"]
    assert d["role"].startswith("ADDED ARM") and s["role"] == "primary"
    assert d["fill_rgb"] == [124, 116, 104] and s["fill_rgb"] == [128, 128, 128]
    assert d["r1_cache_alignment"] is False and s["r1_cache_alignment"] is True
    assert d["suffix"] == "_dinov2" and s["suffix"] == ""
    # slots, gates, k schedule, seeds are single plan-level entries: nothing arm-specific can differ
    for key in ("slots", "ks", "seeds", "gate_families", "threshold_rules", "binding_rule", "recipes", "transforms"):
        assert key in runner.DEFAULT_PLAN and key not in d


def test_declared_weights_hashes_match_files(runner):
    arms = runner.DEFAULT_PLAN["arms"]
    assert sha256_file(DINO_SNAP.resolve()) == arms["dinov2"]["weights_sha256"]
    sg = None if SIGLIP_PATH is None else SIGLIP_PATH / "model.safetensors"
    if sg is not None and sg.exists():
        assert sha256_file(sg) == arms["siglip"]["weights_sha256"]
