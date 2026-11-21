"""Inference specialists, wrapping the existing v3.1 and chapter 02 model artifacts unchanged.

* PatchCore (per category): image score + 256x256 anomaly map.        L1, routine path
* CLIP RN50 two-prompt contrast: "a flawless X" vs "a damaged X".     reference L1 add-on, and an
                                                                         input feature of the kit L2
* Semantic kit L2 (LOCO): ResNet18 spatial/colour/part descriptor -> normal-only
  envelope + PCA-neighbour fusion (v3.1 `kit-semantic-color-normal-v31-4`).  selective L2
* CategoryGate: refuses images that do not look like the selected category.

Preprocessing is identical to the code that produced the stored scores; the application's bench
script (not published) checks live scores against them.
The v3.1 `cascade` modules these classes import are not in this repository: VISION_V31_ROOT must
hold them (see config.py).
"""

from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager

import numpy as np

from . import config

sys.path.insert(0, str(config.VISION))  # `cascade` package of the v3.1 build

_torch = None


def torch():
    global _torch
    if _torch is None:
        import torch as t

        t.set_num_threads(config.TORCH_THREADS)
        _torch = t
    return _torch


@contextmanager
def timed(timings: dict, key: str):
    w, c = time.perf_counter(), time.process_time()
    yield
    timings[key] = {"wall_ms": 1000 * (time.perf_counter() - w), "cpu_ms": 1000 * (time.process_time() - c)}


def load_rgb(path_or_file):
    from PIL import Image, ImageOps

    with Image.open(path_or_file) as src:
        return ImageOps.exif_transpose(src).convert("RGB")


_PRE = None


def patch_tensor(rgb):
    global _PRE
    if _PRE is None:
        from torchvision import transforms

        _PRE = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    return _PRE(rgb).unsqueeze(0)


class PatchCore:
    def __init__(self):
        self.models, self.lock = {}, threading.Lock()

    def model(self, category):
        with self.lock:
            if category not in self.models:
                from cascade.v3_surface import load_patchcore

                self.models[category] = load_patchcore(config.patchcore_path(category))
            return self.models[category]

    def score(self, rgb, category, timings):
        m = self.model(category)
        with timed(timings, "patchcore"):
            x = patch_tensor(rgb)
            with torch().inference_mode():
                out = m(x)
        amap = out.anomaly_map
        amap = (amap[0, 0] if amap.dim() == 4 else amap[0]).numpy()
        return float(out.pred_score.flatten()[0]), amap


class ClipContrast:
    def __init__(self):
        import clip

        t = torch()
        self.model, self.transform = clip.load(str(config.CLIP_RN50), device="cpu", jit=False)
        self.model.eval()
        self.text = {}
        with t.inference_mode():
            for c in config.CATEGORIES:
                v = self.model.encode_text(
                    clip.tokenize([f"a flawless {c.replace('_', ' ')}", f"a damaged {c.replace('_', ' ')}"])
                )
                self.text[c] = v / v.norm(dim=-1, keepdim=True)

    def score(self, rgb, category, timings):
        t = torch()
        with timed(timings, "clip"):
            x = self.transform(rgb).unsqueeze(0)
            with t.inference_mode():
                v = self.model.encode_image(x)
                v = v / v.norm(dim=-1, keepdim=True)
                s = float((v @ self.text[category].T).softmax(dim=-1)[0, 1])
        return s


class SemanticKitL2:
    """Exact live reproduction of the scoring in setup/fit_kit_semantic_v31.py (v3.1 build, not published).
    Returns (anomaly_index, novelty_distance); the review's z_sem uses novelty_distance."""

    def __init__(self):
        from cascade.v31_kit_features import Encoder

        self.encoder = Encoder()
        torch().set_num_threads(config.TORCH_THREADS)  # Encoder() forces 2 threads
        import cv2

        cv2.setNumThreads(config.TORCH_THREADS)
        self.models, self.lock = {}, threading.Lock()

    def model(self, category):
        import joblib

        with self.lock:
            if category not in self.models:
                self.models[category] = joblib.load(str(config.SEMANTIC_MODEL).format(category=category))
            return self.models[category]

    def score(self, image_path, category, clip_score, timings):
        from cascade.v31_kit_semantic import refine_vector

        fusion = self.model(category)
        with timed(timings, "kit_encoder"):
            base, _ = self.encoder.encode(image_path, category)
        with timed(timings, "kit_refine"):
            vec, _ = refine_vector(base, image_path)
        with timed(timings, "kit_fusion"):
            x = np.concatenate([vec, [float(category == c) for c in config.KIT_CATS], [clip_score]])[None, :]
            index, novelty = fusion.evidence(x)
        # Most calibration normals have novelty exactly 0 (inside the normal region).  A live
        # re-encode can land at ~1e-9 instead of 0, which would jump it above that whole tie
        # block and change the conformal p.  Treat float noise below 1e-6 as 0.
        nov = float(novelty[0])
        return float(index[0]), (0.0 if abs(nov) < 1e-6 else nov)


class CategoryGate:
    """Global ResNet18 embedding vs per-category normal centroids (cosine distance).
    Accept the selected category only if its distance is within that category's
    calibrated radius; also report the nearest category."""

    def __init__(self, centroids: dict, radii: dict):
        from torchvision import transforms
        from torchvision.models import resnet18

        t = torch()
        m = resnet18(weights=None)
        m.load_state_dict(t.load(config.RESNET18, map_location="cpu", weights_only=True))
        self.net = t.nn.Sequential(*list(m.children())[:-1]).eval()
        self.pre = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.names = sorted(centroids)
        self.C = np.stack([centroids[n] for n in self.names])
        self.radii = radii

    def embed(self, rgb):
        t = torch()
        with t.inference_mode():
            v = self.net(self.pre(rgb).unsqueeze(0)).flatten().numpy()
        return v / (np.linalg.norm(v) + 1e-9)

    def check(self, rgb, category, timings):
        with timed(timings, "category_gate"):
            d = 1 - self.C @ self.embed(rgb)
        dist = dict(zip(self.names, map(float, d)))
        nearest = min(dist, key=dist.get)
        # accept only if the selected category is the nearest normal centroid AND the image is
        # within that category's radius (3x the farthest calibration normal: gate_margin=3.0 in
        # reference.build, as the application built the run's reference)
        ok = category in dist and nearest == category and dist[category] <= self.radii[category]
        return {
            "accepted": bool(ok),
            "distance": dist.get(category),
            "radius": self.radii.get(category),
            "nearest": nearest,
            "nearest_distance": dist[nearest],
        }
