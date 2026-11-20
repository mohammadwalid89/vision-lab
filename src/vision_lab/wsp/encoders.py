"""Frozen encoders with an in-memory token hook (local weights only, no network).

``SiglipEncoder``  SigLIP2 base, patch 16, 224 px -> 14x14 tokens, loaded from the directory in
                   ``VISION_SIGLIP_DIR``.  tokens = last_hidden_state (post-layernorm),
                   pooled = MAP-head output, mean_patch = token mean before the final layer norm
                   (the R1 cache recipe).  ``preflight`` checks alignment against the R1 cache.
``DinoV2Encoder``  timm vit_base_patch14_reg4_dinov2 from the safetensors file in
                   ``VISION_DINOV2_FILE``, 224 px -> 16x16 patch tokens (4 register tokens
                   dropped), pooled = CLS.
``extract``        runs a list of (image, meta) through named input transforms, computes the
                   slot-2 object aggregate and its permutation placebos while the tokens are
                   in memory, and keeps tokens only if asked (disk: 196x768 f16 = 301 KB/view).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .slots.patch_scoring import (
    aggregates_with_controls,
    normalise_weights,
    patch_coverage,
)


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


SIGLIP_PATH = _env_path("VISION_SIGLIP_DIR")
DINO_SNAP = _env_path("VISION_DINOV2_FILE")


def offline_env():
    for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        os.environ[k] = "1"


class SiglipEncoder:
    name = "siglip2_base"
    grid, img, mean = 14, 224, (0.5, 0.5, 0.5)

    def __init__(self, path=None, threads=3):
        path = path or SIGLIP_PATH
        if path is None:
            raise RuntimeError("set VISION_SIGLIP_DIR to the local SigLIP2 base directory")
        offline_env()
        import torch
        from transformers import SiglipImageProcessor, SiglipVisionModel
        torch.set_num_threads(threads)
        self.torch = torch
        self.proc = SiglipImageProcessor.from_pretrained(str(path))
        self.model = SiglipVisionModel.from_pretrained(str(path), dtype=torch.float32).eval()

    def __call__(self, images: Sequence[np.ndarray]) -> dict:
        from PIL import Image
        # chunked forward (memory): identical per-image results, bounded activations
        T, P, M = [], [], []
        for s0 in range(0, len(images), 32):
            x = self.proc(images=[Image.fromarray(np.asarray(i)) for i in images[s0:s0 + 32]], return_tensors="pt")
            with self.torch.no_grad():
                o = self.model(pixel_values=x["pixel_values"], output_hidden_states=True)
            T.append(o.last_hidden_state.double().numpy())
            P.append(o.pooler_output.double().numpy())
            # R1 recipe: mean_patch is the mean of the block-output tokens before the final layer norm.
            M.append(o.hidden_states[-1].double().numpy().mean(1))
            del o
        t = np.concatenate(T)
        if t.shape[1] != self.grid * self.grid:
            raise RuntimeError(f"unexpected token count {t.shape[1]}")
        return dict(tokens=t, pooled=np.concatenate(P), mean_patch=np.concatenate(M))


class DinoV2Encoder:
    name = "dinov2_vitb14_reg4"
    grid, img, mean = 16, 224, (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(self, file=None, threads=3):
        file = file or DINO_SNAP
        if file is None:
            raise RuntimeError("set VISION_DINOV2_FILE to the local DINOv2 safetensors file")
        offline_env()
        import timm
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.model = timm.create_model("vit_base_patch14_reg4_dinov2.lvd142m", pretrained=True, img_size=224,
                                       pretrained_cfg_overlay=dict(file=str(file))).eval()
        self.n_prefix = self.model.num_prefix_tokens

    def __call__(self, images: Sequence[np.ndarray]) -> dict:
        from PIL import Image
        arr = []
        for i in images:
            im = Image.fromarray(np.asarray(i)).convert("RGB").resize((self.img, self.img), Image.Resampling.BICUBIC)
            a = (np.asarray(im, dtype=np.float32) / 255.0 - np.array(self.mean, dtype=np.float32)) / np.array(self.std, dtype=np.float32)
            arr.append(a.transpose(2, 0, 1))
        x = self.torch.tensor(np.stack(arr))
        with self.torch.no_grad():   # chunked forward (memory); identical per-image results
            f = self.torch.cat([self.model.forward_features(x[s0:s0 + 32]) for s0 in range(0, len(x), 32)])
        t = f[:, self.n_prefix:].double().numpy()
        if t.shape[1] != self.grid * self.grid:
            raise RuntimeError(f"unexpected token count {t.shape[1]}")
        return dict(tokens=t, pooled=f[:, 0].double().numpy(), mean_patch=t.mean(1))


def extract(encoder, items: Sequence[tuple[np.ndarray, Mapping]], transforms: Mapping, batch: int = 8,
            keep_tokens: bool = False, n_perm: int = 3) -> dict:
    """{transform_name: dict(pooled, mean_patch, object, perm_1..n, uniform[, tokens])}.

    ``meta`` must contain ``source_id`` and ``object_mask`` (per source).  The coverage
    weights come from the ORIGINAL object mask in every transform (declared), so the
    patch-scoring features of a masked pipeline are comparable across transforms.
    """
    res = {}
    for tname, tf in transforms.items():
        acc: dict = {}
        for a in range(0, len(items), batch):
            chunk = items[a:a + batch]
            imgs = [tf(im, meta) for im, meta in chunk]
            out = encoder(imgs)
            w = normalise_weights(np.stack([patch_coverage(meta["object_mask"], encoder.grid, encoder.img)
                                            for _, meta in chunk]))
            ag = aggregates_with_controls(out["tokens"], w, [m["source_id"] for _, m in chunk], n_perm)
            part = dict(pooled=out["pooled"], mean_patch=out["mean_patch"], weights=w, **ag)
            if keep_tokens:
                part["tokens"] = out["tokens"].astype(np.float16)
            for k, v in part.items():
                acc.setdefault(k, []).append(v)
        res[tname] = {k: np.concatenate(v, 0) for k, v in acc.items()}
    return res
