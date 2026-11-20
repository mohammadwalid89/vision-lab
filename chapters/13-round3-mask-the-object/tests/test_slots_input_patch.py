import numpy as np
import pytest

from vision_lab.wsp.slots import input_mask as IM
from vision_lab.wsp.slots import patch_scoring as PS
from vision_lab.wsp.synthetic import TinyPatchEncoder, image_world


@pytest.fixture(scope="module")
def world():
    return image_world(n=6)


@pytest.mark.parametrize("attention", [False, True])
def test_slot1_masking_makes_marker_unreachable(world, attention):
    """Known answer: after hard fill outside the object, clean and marked views are
    pixel-identical, so ANY encoder (even with global attention) gives identical features."""
    enc = TinyPatchEncoder(attention=attention)
    tf = IM.build_transforms()
    for w in world:
        meta = dict(object_mask=w["object_mask"], source_id=w["source_id"], content_box=w["content_box"])
        raw = np.abs(enc(w["image_clean"])["pooled"] - enc(w["image_marker"])["pooled"]).max()
        assert raw > 1e-6                                              # marker does reach unmasked features
        for name in ("sam_meanfill", "padding_only"):
            a, b = tf[name](w["image_clean"], meta), tf[name](w["image_marker"], meta)
            assert np.array_equal(a, b), name
            assert np.array_equal(enc(a)["tokens"], enc(b)["tokens"])


def test_slot1_fill_modes(world):
    w = world[0]
    with pytest.raises(IM.LeakError):
        IM.mask_fill(w["image_marker"], w["object_mask"], "image_mean")
    a = IM.mask_fill(w["image_clean"], w["object_mask"], "object_mean")
    b = IM.mask_fill(w["image_marker"], w["object_mask"], "object_mean")
    assert np.array_equal(a, b)                                        # leak-free alternative
    f = IM.mask_fill(w["image_clean"], w["object_mask"], "encoder_mean")
    assert np.all(f[~w["object_mask"]] == 128)                         # SigLIP mean -> ~0 input
    assert np.array_equal(f[w["object_mask"]], w["image_clean"][w["object_mask"]])


def test_slot1_identity_and_random_area_controls(world):
    tf = IM.build_transforms(n_random=3)
    w = world[1]
    meta = dict(object_mask=w["object_mask"], source_id=w["source_id"], content_box=w["content_box"])
    assert np.array_equal(tf["identity"](w["image_marker"], meta), w["image_marker"])
    keeps = []
    for d in (1, 2, 3):
        rk, info = IM.random_equal_area_keep(w["object_mask"], f"r3|slot1|random_area|{w['source_id']}|{d}")
        assert info["area"] == info["area_random"] == int(w["object_mask"].sum())
        keeps.append(rk)
        rep = IM.overlap_report(rk, w["object_mask"], w["marker_box"])
        assert 0 <= rep["object_pixels_removed"] <= 1 and 0 <= rep["marker_pixels_kept"] <= 1
    assert not np.array_equal(keeps[0], keeps[1])
    again, _ = IM.random_equal_area_keep(w["object_mask"], f"r3|slot1|random_area|{w['source_id']}|1")
    assert np.array_equal(again, keeps[0])                             # deterministic per source/draw


def test_padding_sparing_placebo_keeps_marker_and_risk_check(world):
    tf = IM.build_transforms(n_random=3, padding_sparing=True)
    for w in world:
        meta = dict(object_mask=w["object_mask"], source_id=w["source_id"], content_box=w["content_box"])
        out = tf["random_area_ps_1"](w["image_marker"], meta)
        y0, x0, y1, x1 = w["marker_box"]
        assert np.array_equal(out[y0:y1, x0:x1], w["image_marker"][y0:y1, x0:x1])   # marker untouched
        f = IM.padding_kept_fraction(w["object_mask"], w["content_box"], f"k|{w['source_id']}")
        assert 0.0 <= f <= 1.0


def test_resize_mask_nearest():
    m = np.zeros((4, 4), dtype=np.uint8)
    m[:2, :2] = 255
    r = IM.resize_mask(m, (8, 8))
    assert r.sum() == 16 and r[:4, :4].all()


def test_patch_coverage_and_weights():
    full = np.ones((384, 384), dtype=np.uint8) * 255
    assert np.allclose(PS.patch_coverage(full), 1.0)
    half = np.zeros((224, 224), dtype=bool)
    half[:, :112] = True
    cov = PS.patch_coverage(half).reshape(14, 14)
    assert np.allclose(cov[:, :7], 1) and np.allclose(cov[:, 7:], 0)
    w = PS.normalise_weights(np.vstack([cov.reshape(-1), np.zeros(196)]))
    assert np.allclose(w.sum(1), 1) and np.allclose(w[1], 1 / 196)


def test_linear_scoring_equals_pooling_identity():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(5, 196, 12))
    w = PS.normalise_weights(rng.random((5, 196)))
    theta = rng.normal(size=13)
    lhs = PS.token_logit_aggregate(t, w, theta)
    rhs = PS.aggregate(t, w) @ theta[:-1] + theta[-1]
    assert np.allclose(lhs, rhs)


def test_permutation_placebo_properties():
    rng = np.random.default_rng(1)
    w = PS.normalise_weights(rng.random((4, 196)))
    sids = ["a", "a", "b", "c"]                                        # two views of source a
    p1 = PS.permuted_weights(w, sids, 1)
    assert np.allclose(np.sort(p1, 1), np.sort(w, 1))                  # same multiset
    assert np.array_equal(PS.permutation_for("a", 196, 1), PS.permutation_for("a", 196, 1))
    assert not np.array_equal(PS.permutation_for("a", 196, 1), PS.permutation_for("a", 196, 2))
    w2 = w.copy()
    w2[1] = w[0]
    p = PS.permuted_weights(w2, sids, 1)
    assert np.array_equal(p[0], p[1])                                  # same permutation per source


@pytest.mark.parametrize("attention,invariant", [(False, True), (True, False)])
def test_slot2_object_aggregate_leakage_known_answer(world, attention, invariant):
    """Local tokens: object aggregate is exactly marker-invariant (zero weight on the marker
    patch).  Global attention: object tokens absorb the marker -> leakage (honest limit)."""
    enc = TinyPatchEncoder(attention=attention)
    for w in world:
        wt = PS.normalise_weights(PS.patch_coverage(w["object_mask"], grid=8, img=32))[None]
        zc = PS.aggregate(enc(w["image_clean"])["tokens"][None], wt)
        zm = PS.aggregate(enc(w["image_marker"])["tokens"][None], wt)
        leak = PS.marker_leakage(zc, zm)[0]
        assert (leak == 0.0) if invariant else (leak > 1e-6)
