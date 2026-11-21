#!/usr/bin/env python3
"""Phase 3: isolated train-normal models, validation-normal calibration, exposed-dev scoring.
No locked-final manifest is read by this program. Never import Phase 2 final scores.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

D = Path(__file__).resolve().parent
OUT = D / "out"
M = D / "models"
M.mkdir(exist_ok=True)
# Phase 2 component verifier, shared with chapter 03
P2 = Path(os.environ.get("P2_CODE_DIR", str(D.parents[1] / "03-surface-sealed-test" / "code")))
sys.path.insert(0, str(P2))
import p2_components as C  # noqa: E402

C.W = 192
C.SHAPES = False
torch.set_num_threads(2)
CATS = ["juice_bottle", "splicing_connectors"]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def manifest(name, cat):
    return [r for r in json.loads((OUT / ("p3_" + name + ".json")).read_text()) if r["category"] == cat]


def save_json_atomic(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, sort_keys=True))
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_dev():
    assert (OUT / "DOWNLOAD_COMPLETE").exists() and (OUT / "p3_split_manifest.json").exists()
    assert not (OUT / "p3_frozen_policy.json").exists(), "Models frozen: no development rerun"
    assert not (OUT / "P3_FINAL_OPENED").exists()


def image(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR), dtype=np.uint8)


def rgbstats(paths):
    return np.stack([image(p).astype(np.float32).reshape(-1, 3).mean(0) / 255 for p in paths])


def fit_gate(train, val):
    t = rgbstats([r["path"] for r in train])
    v = rgbstats([r["path"] for r in val])
    med = np.median(t, 0)
    scale = np.maximum(np.std(t, 0), 0.02)
    dv = np.sqrt((((v - med) / scale) ** 2).sum(1))
    return {"median": med.tolist(), "scale": scale.tolist(), "threshold": float(max(np.quantile(dv, 0.995), 0.5))}


def gate_accept(p, g):
    x = image(p).astype(np.float32).reshape(-1, 3).mean(0) / 255
    return bool(np.sqrt(np.sum(((x - np.array(g["median"])) / np.array(g["scale"])) ** 2)) <= g["threshold"])


def relational(lab_id):
    h, w = lab_id.shape
    props = []
    for k in range(C.K):
        mask = lab_id == k
        n = float(mask.sum())
        if n:
            ys, xs = np.where(mask)
            cx = float(xs.mean() / w)
            cy = float(ys.mean() / h)
            sx = float(xs.std() / w)
            sy = float(ys.std() / h)
        else:
            cx = cy = sx = sy = 0.0
        lbl, nn = ndimage.label(ndimage.binary_opening(mask))
        areas = np.bincount(lbl.ravel())[1:] if nn else np.zeros(0)
        ncomp = int((areas >= 0.0004 * h * w).sum())
        props.append((n / (h * w), cx, cy, sx, sy, ncomp))
    f = [x for p in props for x in p]
    # Material-to-material distances, vertical/horizontal order, co-occurrence.
    for i in range(C.K):
        for j in range(i + 1, C.K):
            a, b = props[i], props[j]
            f.extend([np.hypot(a[1] - b[1], a[2] - b[2]), a[1] - b[1], a[2] - b[2]])
    return np.asarray(f, np.float32)


def comp_extract(path, centers):
    lab = C.load_lab(path, reg=None)
    x, lab_id = C.features(lab, centers)
    return x, relational(lab_id)


def run_component(cat):
    train = manifest("train", cat)
    val = manifest("validation", cat)
    dev = manifest("dev", cat)
    rng = np.random.default_rng(47)
    labs = [C.load_lab(r["path"], reg=None) for r in train[:65]]
    px = np.concatenate(
        [
            lab.reshape(-1, 3)[
                rng.choice(lab.shape[0] * lab.shape[1], min(320, lab.shape[0] * lab.shape[1]), replace=False)
            ]
            for lab in labs
        ]
    )
    km = MiniBatchKMeans(C.K, random_state=47, n_init=3, batch_size=4096).fit(px)
    centers = km.cluster_centers_.astype(np.float32)
    del labs, px, km
    print("COMP FIT", cat, "train", len(train), "val", len(val), "dev", len(dev), flush=True)
    X = []
    R = []
    for i, r in enumerate(train):
        x, rel = comp_extract(r["path"], centers)
        X.append(x)
        R.append(rel)
        if (i + 1) % 80 == 0:
            print("COMP TRAIN", cat, i + 1, flush=True)
    X = np.stack(X)
    R = np.stack(R)
    norm = C.Normal(X)
    rnorm = C.Normal(R)
    # Relational nearest-neighbor model: nearest train-normal relation signature
    # on standardized, clipped pairwise component geometry; not trained on dev.
    rZ = np.clip(rnorm.Z, -6, 6).astype(np.float32)
    nn = NearestNeighbors(n_neighbors=min(5, len(R)), algorithm="brute").fit(rZ)

    def score(p):
        t = time.perf_counter()
        x, rel = comp_extract(p, centers)
        cs, _ = norm.scores(x)
        zr = np.clip(rnorm.z(rel), -6, 6).astype(np.float32)
        d = nn.kneighbors(zr[None], return_distance=True)[0][0]
        return {
            "component": float(cs["knn"]),
            "component_maxz": float(cs["max_z"]),
            "relational": float(np.mean(d)),
            "latency_ms": float((time.perf_counter() - t) * 1000),
        }

    payload = {
        "cat": cat,
        "centers": centers,
        "comp_med": norm.med,
        "comp_scale": norm.scale,
        "comp_Z": norm.Z,
        "rel_med": rnorm.med,
        "rel_scale": rnorm.scale,
        "rel_Z": rZ,
    }
    import io

    buf = io.BytesIO()
    np.savez_compressed(buf, **payload)
    model = M / f"component_relational_{cat}.npz"
    model.write_bytes(buf.getvalue())
    results = {
        "model": "component_relational",
        "category": cat,
        "model_sha256": sha(model),
        "validation": [],
        "development": [],
        "n_train": len(train),
    }
    for split, rows in [("validation", val), ("development", dev)]:
        for i, r in enumerate(rows):
            s = score(r["path"])
            results[split].append({"sha256": r["sha256"], "category": cat, "stratum": r["stratum"], **s})
            if (i + 1) % 30 == 0:
                print("COMP", cat, split, i + 1, "/", len(rows), flush=True)
    dest = OUT / f"p3_component_relational_{cat}_dev.json"
    save_json_atomic(dest, results)
    print("COMP DONE", cat, "sha", sha(dest), flush=True)


class Feat(torch.nn.Module):
    def __init__(self, which):
        super().__init__()
        import torchvision

        if which == "wrn":
            m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        elif which == "resnet":
            m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(which)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.dim = 1024 if which == "wrn" else 384
        self.eval()

    @torch.inference_mode()
    def forward(self, x):
        a = self.layer2(self.stem(x))
        b = self.layer3(a)
        a = F.avg_pool2d(a, 3, 1, 1)
        b = F.avg_pool2d(b, 3, 1, 1)
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([a, b], dim=1)
        N, D, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(N, H * W, D)
        if self.dim == 1024:
            f = F.adaptive_avg_pool1d(f, 1024)
        return f


def embed(net, paths, bs=4):
    out = []
    for i in range(0, len(paths), bs):
        ims = np.stack([image(p) for p in paths[i : i + bs]])
        t = torch.from_numpy(ims.astype(np.float32) / 255).permute(0, 3, 1, 2)
        out.append(net((t - MEAN) / STD).half())
        if (i + bs) % 64 < bs:
            print("EMBED", i + len(ims), "/", len(paths), flush=True)
    return torch.cat(out)


def coreset(E, seed=45, subset=20000, k=512):
    # Approximate greedy farthest-first sampling in fixed random projection.
    n, d = E.shape
    rng = np.random.default_rng(seed)
    subsetidx = np.sort(rng.choice(n, min(subset, n), replace=False))
    candidates = E[torch.from_numpy(subsetidx)].float()
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(d, 64, generator=g) / 8.0
    Z = candidates @ P
    sq = (Z * Z).sum(1)
    sel = [int(torch.randint(len(Z), (1,), generator=g))]
    dist = sq - 2 * (Z @ Z[sel[0]]) + sq[sel[0]]
    for _ in range(min(k, len(Z)) - 1):
        j = int(torch.argmax(dist))
        sel.append(j)
        dist = torch.minimum(dist, sq - 2 * (Z @ Z[j]) + sq[j])
    return candidates[sel].half()


def score_patches(net, bank, rows, cat, split):
    out = []
    b = bank.float()
    for i, r in enumerate(rows):
        t = time.perf_counter()
        f = embed(net, [r["path"]], bs=1)[0].float()
        d = torch.cdist(f, b).min(1).values
        score = float(d.max())
        out.append(
            {
                "sha256": r["sha256"],
                "category": cat,
                "stratum": r["stratum"],
                "score": score,
                "latency_ms": float((time.perf_counter() - t) * 1000),
            }
        )
        if (i + 1) % 30 == 0:
            print("SCORE", cat, split, i + 1, "/", len(rows), flush=True)
    return out


def run_patch(which, cat):
    train = manifest("train", cat)
    val = manifest("validation", cat)
    dev = manifest("dev", cat)
    print("PATCH", which, cat, "train", len(train), "val", len(val), "dev", len(dev), flush=True)
    net = Feat(which)
    t0 = time.time()
    E = embed(net, [r["path"] for r in train])
    bank = coreset(E.reshape(-1, E.shape[-1]), seed=45, k=512)
    model = M / f"{which}_{cat}.pt"
    torch.save({"bank": bank, "backbone": which, "train_count": len(train), "k": len(bank), "seed": 45}, model)
    del E
    result = {
        "model": which,
        "category": cat,
        "model_sha256": sha(model),
        "validation": [],
        "development": [],
        "n_train": len(train),
    }
    result["validation"] = score_patches(net, bank, val, cat, "validation")
    result["development"] = score_patches(net, bank, dev, cat, "development")
    dest = OUT / f"p3_{which}_{cat}_dev.json"
    save_json_atomic(dest, result)
    print("PATCH DONE", which, cat, "seconds", round(time.time() - t0, 1), "sha", sha(dest), flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--model", choices=["component", "wrn", "resnet"], required=True)
    a.add_argument("--category", choices=CATS, required=True)
    args = a.parse_args()
    ensure_dev()
    target = OUT / f"p3_{'component_relational' if args.model == 'component' else args.model}_{args.category}_dev.json"
    if target.exists():
        print("SKIP COMPLETED", target)
        sys.exit(0)
    if args.model == "component":
        run_component(args.category)
    else:
        run_patch(args.model, args.category)
