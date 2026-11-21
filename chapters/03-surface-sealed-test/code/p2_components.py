"""Phase 2 kit specialist candidate: an object-centric component verifier.

Per kit type, learned from TRAIN good kits only:
  0. registration: the kit (tray / box / bag) is segmented from the dark background (Otsu threshold on
     lightness, learned on train kits), cropped to its bounding box and resized to a fixed frame, so the
     grids below follow the kit, not the camera frame;
  1. materials: k-means (k=8) on Lab colour of pixels sampled from train kits;
  2. components: connected regions of each material (min area 0.04 % of the image);
  3. measurements, all named: per material its area fraction overall and on 2x2 / 3x3 / 5x3 grids,
     its number of components overall and per 2x2 / 5x3 cell, and its four largest component areas
     (v1).  Optional shape measurements (P2_SHAPES=1: elongated / holed counts, longest axes) were tested
     as v2 and made breakfast boxes worse on exposed images, so they are off;
  4. normal ranges: median and robust scale (max(1.4826 MAD, std, floor)) per measurement;
  5. scores: max |z| over measurements ("the most violated rule"), mean of the top-5 |z|, and the
     k-NN (k=5) distance in clipped z-space.
Calibration is OUT-OF-FOLD (5 folds over the train kits).  Every scored image carries its most violated
measurements, which is the explanation shown to a person.

Outputs: models/comp_<kit>.npz, out/comp_calibration.json, out/comp_scores.jsonl."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from sklearn.cluster import MiniBatchKMeans

HERE = Path(__file__).resolve().parent
OUT, MODELS = HERE / "out", HERE / "models"
# shape descriptors (v2) were a negative result on exposed images; off by default (v1)
SHAPES = os.environ.get("P2_SHAPES", "0") == "1"
KITS = sys.argv[1:] or ["breakfast_box", "pushpins", "screw_bag"]
W = 320
GRIDS = [(2, 2), (3, 3), (5, 3)]
COUNT_GRIDS = [(2, 2), (5, 3)]
K = 8


def rgb2lab(a):
    a = a.astype(np.float32) / 255.0
    a = np.where(a > 0.04045, ((a + 0.055) / 1.055) ** 2.4, a / 12.92)
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]], np.float32)
    xyz = a @ M.T / np.array([0.95047, 1.0, 1.08883], np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def load_lab(path, reg=None):
    """reg = (lightness threshold, frame height) -> registered crop; None -> plain resize."""
    return lab_of(Image.open(path).convert("RGB"), reg)


def lab_of(im, reg=None):
    if reg is None:
        h = int(round(im.height * W / im.width))
        return rgb2lab(np.asarray(im.resize((W, h), Image.BILINEAR)))
    thr, fh = reg
    small = rgb2lab(np.asarray(im.resize((W, int(round(im.height * W / im.width))), Image.BILINEAR)))
    fg = ndimage.binary_closing(small[..., 0] > thr, iterations=3)
    lbl, n = ndimage.label(fg)
    if n:
        big = 1 + int(np.argmax(ndimage.sum(fg, lbl, index=np.arange(1, n + 1))))
        ys, xs = np.where(lbl == big)
        sx, sy = im.width / W, im.height / small.shape[0]
        box = (int(xs.min() * sx), int(ys.min() * sy), int((xs.max() + 1) * sx), int((ys.max() + 1) * sy))
        if (box[2] - box[0]) * (box[3] - box[1]) > 0.2 * im.width * im.height:
            im = im.crop(box)
    return rgb2lab(np.asarray(im.resize((W, fh), Image.BILINEAR)))


def otsu(v):
    h, e = np.histogram(v, 128)
    c = (e[:-1] + e[1:]) / 2
    w = np.cumsum(h)
    m = np.cumsum(h * c)
    with np.errstate(divide="ignore", invalid="ignore"):
        b = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return float(c[np.nanargmax(b[:-1])])


def names_for():
    n = []
    for k in range(K):
        n.append(f"m{k}:area")
        for gx, gy in GRIDS:
            n += [f"m{k}:area@{gx}x{gy}[{i},{j}]" for j in range(gy) for i in range(gx)]
        n.append(f"m{k}:count")
        for gx, gy in COUNT_GRIDS:
            n += [f"m{k}:count@{gx}x{gy}[{i},{j}]" for j in range(gy) for i in range(gx)]
        n += [f"m{k}:blob{r}" for r in range(1, 5)]
        if SHAPES:
            n += [f"m{k}:elongated", f"m{k}:holed"] + [f"m{k}:length{r}" for r in range(1, 5)]
    return n


def shapes(lbl, keep, frame):
    """per kept component: (elongation, has_hole, major axis / frame width)"""
    out = []
    sl = ndimage.find_objects(lbl)
    for i in keep:
        s = sl[i]
        comp = lbl[s] == i + 1
        ys, xs = np.nonzero(comp)
        if len(xs) < 3:
            out.append((1.0, False, 0.0))
            continue
        cov = np.cov(np.stack([xs, ys]).astype(np.float32)) + 1e-6 * np.eye(2)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        hole = ndimage.binary_fill_holes(comp).sum() > 1.05 * comp.sum()
        out.append((float(np.sqrt(ev[0] / ev[1])), bool(hole), float(4 * np.sqrt(ev[0]) / frame)))
    return out


def features(lab, centers):
    H, Wd = lab.shape[:2]
    d = ((lab.reshape(-1, 1, 3) - centers[None]) ** 2).sum(-1)
    lab_id = d.argmin(1).reshape(H, Wd)
    min_area = 0.0004 * H * Wd
    f = []
    yy, xx = np.mgrid[0:H, 0:Wd]
    for k in range(K):
        m = lab_id == k
        f.append(m.mean())
        for gx, gy in GRIDS:
            for j in range(gy):
                for i in range(gx):
                    f.append(m[j * H // gy : (j + 1) * H // gy, i * Wd // gx : (i + 1) * Wd // gx].mean())
        lbl, n = ndimage.label(ndimage.binary_opening(m, iterations=1))
        areas = ndimage.sum(m, lbl, index=np.arange(1, n + 1)) if n else np.zeros(0)
        keep = np.where(areas >= min_area)[0]
        if len(keep):
            cy, cx = np.array(ndimage.center_of_mass(m, lbl, index=keep + 1)).T
        else:
            cy = cx = np.zeros(0)
        f.append(len(keep))
        for gx, gy in COUNT_GRIDS:
            ci = np.minimum((cx * gx / Wd).astype(int), gx - 1)
            cj = np.minimum((cy * gy / H).astype(int), gy - 1)
            for j in range(gy):
                for i in range(gx):
                    f.append(int(((ci == i) & (cj == j)).sum()))
        big = np.sort(areas[keep])[::-1] / (H * Wd) if len(keep) else np.zeros(0)
        f += [float(big[r]) if r < len(big) else 0.0 for r in range(4)]
        if not SHAPES:
            continue
        sh = shapes(lbl, keep, Wd)
        lens = sorted((x[2] for x in sh), reverse=True)
        f += [sum(x[0] > 3 for x in sh), sum(x[1] for x in sh)] + [lens[r] if r < len(lens) else 0.0 for r in range(4)]
    return np.array(f, np.float32), lab_id


class Normal:
    def __init__(self, X):
        self.med = np.median(X, 0)
        mad = 1.4826 * np.median(np.abs(X - self.med), 0)
        self.scale = np.maximum.reduce([mad, X.std(0), np.full(X.shape[1], 1e-3)])
        self.Z = np.clip(self.z(X), -10, 10)

    def z(self, x):
        return (x - self.med) / self.scale

    def scores(self, x):
        z = np.clip(self.z(x), -10, 10)
        a = np.abs(z)
        top = np.sort(a)[::-1]
        knn = np.sort(np.sqrt(((self.Z - z) ** 2).mean(1)))[:5].mean()
        return {"max_z": float(top[0]), "top5_z": float(top[:5].mean()), "knn": float(knn)}, z


def main():
    inv = json.load(open(OUT / "inventory.json"))
    calib = json.loads((OUT / "comp_calibration.json").read_text()) if (OUT / "comp_calibration.json").exists() else {}
    fout = open(OUT / "comp_scores.jsonl", "a")
    NAMES = names_for()
    for kit in KITS:
        t0 = time.time()
        train = [r["path"] for r in inv if r["category"] == kit and r["split"] == "train"]
        raw = [load_lab(p) for p in train[:60]]
        thr = otsu(np.concatenate([lab[..., 0].ravel() for lab in raw]))
        fh = int(np.median([lab.shape[0] for lab in raw]))
        reg = (thr, fh) if os.environ.get("P2_REGISTER", "1") == "1" else None
        labs = [load_lab(p, reg) for p in train]
        rng = np.random.default_rng(0)
        px = np.concatenate(
            [lab.reshape(-1, 3)[rng.choice(lab.shape[0] * lab.shape[1], 600, replace=False)] for lab in labs]
        )
        km = MiniBatchKMeans(K, random_state=0, n_init=5, batch_size=4096).fit(px)
        C = km.cluster_centers_.astype(np.float32)
        X = np.stack([features(lab, C)[0] for lab in labs])
        del labs
        folds = np.random.default_rng(1).permutation(len(X)) % 5
        oof = {k: np.zeros(len(X)) for k in ("max_z", "top5_z", "knn")}
        for f in range(5):
            nm = Normal(X[folds != f])
            for i in np.where(folds == f)[0]:
                s, _ = nm.scores(X[i])
                for k in oof:
                    oof[k][i] = s[k]
        nm = Normal(X)
        np.savez(
            MODELS / f"comp_{kit}.npz",
            centers=C,
            med=nm.med,
            scale=nm.scale,
            Z=nm.Z,
            names=np.array(NAMES),
            reg=np.array(reg if reg else (-1, -1), np.float32),
        )
        calib[kit] = {k: v.tolist() for k, v in oof.items()}
        calib[kit]["n_train"] = len(X)
        (OUT / "comp_calibration.json").write_text(json.dumps(calib))
        swatch = [[int(v) for v in c] for c in C]
        evalr = [r for r in inv if r["category"] == kit and r["split"] in ("test", "validation")]
        for r in evalr:
            w0, c0 = time.perf_counter(), time.process_time()
            x, _ = features(load_lab(r["path"], reg), C)
            s, z = nm.scores(x)
            w1, c1 = time.perf_counter(), time.process_time()
            top = np.argsort(-np.abs(z))[:5]
            fout.write(
                json.dumps(
                    {
                        "sha256": r["sha256"],
                        "category": kit,
                        "split": r["split"],
                        "type": r["type"],
                        "exposed": r["exposed"],
                        "path": r["path"],
                        **s,
                        "top": [[NAMES[i], float(x[i]), float(nm.med[i]), float(z[i])] for i in top],
                        "ms": {"wall": 1000 * (w1 - w0), "cpu": 1000 * (c1 - c0)},
                    }
                )
                + "\n"
            )
            fout.flush()
        print(
            json.dumps(
                {
                    "kit": kit,
                    "registration": reg,
                    "n_train": len(X),
                    "n_features": X.shape[1],
                    "eval_n": len(evalr),
                    "s": round(time.time() - t0, 1),
                    "materials_lab": swatch,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
