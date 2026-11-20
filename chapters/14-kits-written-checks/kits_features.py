"""SigLIP2-base patch-token cache for the kits track (train and dev rows only).

Per image: `patches` (196, 768) final encoder block output before the post-layer-norm,
`pooled` (768,) MAP-head output, `layer_mean` (12, 768) mean patch token per block.
Weights are read from a local directory (--model-dir or SIGLIP2_DIR); nothing is downloaded.
"""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

ALLOWED_SPLITS = ("train", "dev")
ACTIVE_ROLES = ("fit_normal", "calib_normal", "eval_good", "eval_logical")


class SiglipPatchEncoder:
    def __init__(self, model_dir: Path, threads: int = 2):
        import torch
        from safetensors import safe_open
        from transformers import SiglipVisionConfig, SiglipVisionModel

        torch.set_num_threads(threads)
        model_dir = Path(model_dir)
        cfg_raw = json.loads((model_dir / "config.json").read_text())
        cfg = SiglipVisionConfig(**cfg_raw["vision_config"])
        cfg._attn_implementation = "eager"
        self.model = SiglipVisionModel(cfg)
        with safe_open(str(model_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            state = {k[len("vision_model."):]: f.get_tensor(k).float()
                     for k in f.keys() if k.startswith("vision_model.")}
        self.model.load_state_dict(state, strict=True)
        del state
        self.model.eval().requires_grad_(False)
        self.size = cfg.image_size
        self.grid = cfg.image_size // cfg.patch_size
        self.tokens = self.grid ** 2
        self.dim = cfg.hidden_size
        self.layers = cfg.num_hidden_layers
        self.mean = np.array([.5, .5, .5], np.float32)
        self.std = np.array([.5, .5, .5], np.float32)
        self._per_layer = []
        self._last = None
        for layer in self.model.encoder.layers:
            layer.register_forward_hook(self._hook)
        self.weights_sha256 = _sha(model_dir / "model.safetensors")

    def _hook(self, module, args, out):
        h = out[0] if isinstance(out, (tuple, list)) else out
        self._last = h.detach()
        self._per_layer.append(h.detach().mean(dim=1)[0].cpu().numpy())

    def preprocess(self, image):
        import torch
        from PIL import Image
        a = np.asarray(image.resize((self.size, self.size), Image.Resampling.BILINEAR),
                       dtype=np.float32) / 255.0
        a = (a - self.mean) / self.std
        return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)

    def encode(self, image) -> dict:
        import torch
        self._per_layer = []
        self._last = None
        with torch.inference_mode():
            out = self.model(self.preprocess(image))
        if self._last is None or len(self._per_layer) != self.layers:
            raise RuntimeError("hook capture incomplete")
        return {"patches": self._last[0].cpu().numpy().astype(np.float16),
                "pooled": out.pooler_output[0].float().cpu().numpy().astype(np.float16),
                "layer_mean": np.stack(self._per_layer).astype(np.float16)}

    def describe(self):
        return {"encoder": "siglip2_base", "weights_sha256": self.weights_sha256,
                "input_size": self.size, "patch_grid": self.grid,
                "patch_tokens": self.tokens, "width": self.dim, "layers": self.layers,
                "precision": "float32 inference, float16 stored",
                "normalisation": "mean=std=0.5"}


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


class FeatureCache:
    """Read-only access to the cached arrays, restricted to train and dev rows."""

    def __init__(self, root):
        self.root = Path(root)
        idx = json.loads((self.root / "index.json").read_text())
        if idx.get("status") != "READY":
            raise RuntimeError(f"feature cache not ready: {idx.get('status')}")
        self.index = idx
        self.rows = idx["rows"]
        self._patches = np.load(self.root / "patches.npy", mmap_mode="r")
        self._pooled = np.load(self.root / "pooled.npy", mmap_mode="r")
        self._layer = np.load(self.root / "layer_mean.npy", mmap_mode="r")
        bad = [r["id"] for r in self.rows if r["split"] not in ALLOWED_SPLITS]
        if bad:
            raise PermissionError(f"cache contains holdout rows: {bad[:3]}")

    def select(self, split, category=None, role=None):
        if split not in ALLOWED_SPLITS:
            raise PermissionError("holdout is sealed")
        out = [i for i, r in enumerate(self.rows)
               if r["split"] == split
               and (category is None or r["category"] == category)
               and (role is None or r["role"] == role)]
        if not out:
            raise RuntimeError(f"no rows for {split}/{category}/{role}")
        return out

    def get(self, i):
        if self.rows[i]["split"] not in ALLOWED_SPLITS:
            raise PermissionError("holdout is sealed")
        return {"patches": np.asarray(self._patches[i], np.float32),
                "pooled": np.asarray(self._pooled[i], np.float32),
                "layer_mean": np.asarray(self._layer[i], np.float32)}


def extract(manifest_path, dataset_root, model_dir, out_dir, limit=None):
    from kits_split import open_image

    manifest = json.loads(Path(manifest_path).read_text())
    rows = [r for r in manifest["images"]
            if r["split"] in ALLOWED_SPLITS and r["role"] in ACTIVE_ROLES]
    if limit:
        rows = rows[:limit]
    n = len(rows)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    encoder = SiglipPatchEncoder(model_dir)
    shapes = {"patches": (n, encoder.tokens, encoder.dim), "pooled": (n, encoder.dim),
              "layer_mean": (n, encoder.layers, encoder.dim)}
    arrays = {k: np.lib.format.open_memmap(out_dir / f"{k}.npy", mode="w+",
                                           dtype=np.float16, shape=s)
              for k, s in shapes.items()}
    t0 = time.perf_counter()
    for i, row in enumerate(rows):
        feats = encoder.encode(open_image(row, dataset_root))
        for k, a in arrays.items():
            if not np.isfinite(feats[k]).all():
                raise ValueError(f"non-finite {k} for {row['id']}")
            a[i] = feats[k]
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"{i + 1}/{n} {(time.perf_counter() - t0) / (i + 1):.2f}s/img", flush=True)
    for a in arrays.values():
        a.flush()
    index = {"status": "READY", "description": encoder.describe(),
             "shapes": {k: list(s) for k, s in shapes.items()}, "rows": rows,
             "splits": sorted({r["split"] for r in rows}),
             "roles": sorted({r["role"] for r in rows}), "n": n}
    (out_dir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True, help="split manifest from kits_split.py")
    ap.add_argument("--dataset-root", default=os.environ.get("LOCO_ROOT"),
                    help="MVTec LOCO root (default: $LOCO_ROOT)")
    ap.add_argument("--model-dir", default=os.environ.get("SIGLIP2_DIR"),
                    help="local SigLIP2-base directory (default: $SIGLIP2_DIR)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    if not a.dataset_root or not a.model_dir:
        ap.error("--dataset-root/LOCO_ROOT and --model-dir/SIGLIP2_DIR are required")
    extract(a.manifest, a.dataset_root, a.model_dir, a.out, a.limit)
