"""Smoke run on a synthetic feature cache (planted parts, 196 x 32 tokens).

Builds a cache with the real layout, runs fit + dev scoring end to end, checks that
every score is finite, that scoring is deterministic, and that defects score higher.
"""
import json
import tempfile
from pathlib import Path

import kits_eval as E
import kits_head as H
import numpy as np
from kits_features import FeatureCache

G, D = 14, 32
P = G * G


def block(r0, c0, h=3, w=3):
    return [r * G + c for r in range(r0, r0 + h) for c in range(c0, c0 + w)]


def basis():
    B = np.zeros((5, D), np.float32)
    B[np.arange(5), np.arange(5)] = 1.0
    return B


def image(seed, defect=None):
    B = basis()
    rng = np.random.default_rng(seed)
    X = 2.0 * B[0] + rng.normal(0, 0.05, (P, D)).astype(np.float32)
    X[block(3, 3)] = 3.0 * B[1] + rng.normal(0, 0.05, (9, D))
    X[block(8, 9)] = 3.0 * B[2] + rng.normal(0, 0.05, (9, D))
    if defect == "missing":
        X[block(3, 3)] = 2.0 * B[0] + rng.normal(0, 0.05, (9, D))
    if defect == "duplicate":
        X[block(10, 1)] = 3.0 * B[1] + rng.normal(0, 0.05, (9, D))
    pooled = rng.normal(size=D).astype(np.float32) + (2.0 if defect else 0.0)
    return X, pooled


def build_cache(path):
    spec = ([("train", "fit_normal", None, 8), ("train", "calib_normal", None, 5),
             ("train", "eval_good", None, 5), ("train", "eval_logical", "missing", 5),
             ("dev", "eval_good", None, 4), ("dev", "eval_logical", "duplicate", 4)])
    rows, patches, pooled, layer = [], [], [], []
    seed = 0
    for split, role, defect, n in spec:
        for j in range(n):
            seed += 1
            X, p = image(seed, defect)
            rows.append({"id": f"{split}-{role}-{j}", "split": split, "role": role,
                         "category": "juice_bottle"})
            patches.append(X.astype(np.float16))
            pooled.append(p.astype(np.float16))
            layer.append(np.zeros((2, D), np.float16))
    np.save(path / "patches.npy", np.stack(patches))
    np.save(path / "pooled.npy", np.stack(pooled))
    np.save(path / "layer_mean.npy", np.stack(layer))
    (path / "index.json").write_text(json.dumps({"status": "READY", "rows": rows}))
    return FeatureCache(path)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        cache = build_cache(Path(tmp))
        frozen = E.fit_category(cache, "juice_bottle")
        rows = E.score_split(cache, frozen, "dev")
        again = E.score_split(cache, E.fit_category(cache, "juice_bottle"), "dev")
    assert len(rows) == 8
    for r in rows:
        for name in E.CANDIDATES:
            assert np.isfinite(r[name]), (name, r["id"])
        assert {k for k in r if k.startswith("check_")} == {f"check_{c}" for c in H.CHECK_ORDER}
    assert [r["programmed_a0.5"] for r in rows] == [r["programmed_a0.5"] for r in again]
    good = [r["programmed_a0.5"] for r in rows if r["y"] == 0]
    bad = [r["programmed_a0.5"] for r in rows if r["y"] == 1]
    assert np.mean(bad) > np.mean(good)
    print("smoke ok: mean score good %.3f, defect %.3f" % (np.mean(good), np.mean(bad)))


if __name__ == "__main__":
    main()
