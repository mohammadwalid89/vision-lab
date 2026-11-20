"""Score nine MVTec AD categories that the v3.1 scorer never saw, and cache embeddings.

The earlier work used six AD and three LOCO categories. Scoring nine more with the
same PatchCore + CLIP recipe gives a held-out-category control for category-prior
and threshold overfitting without new data. Embeddings (CLIP RN50 and pooled
ResNet-18) are cached for every scored AD image, old and new.

The script checks each raw file against the manifest SHA before inference and
skips a category whose rows are already in the output file.

Environment:
  VISION_V31_ROOT    tree holding results/v31/... score tables and data/public/mvtec_ad/manifest.json
  VISION_OUT         output directory (default ./out)
  VISION_MODEL_CACHE directory with clip/RN50.pt and torch/hub/checkpoints/resnet18-f37072fd.pth
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from common import RES, ROOT, load_local_scores  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from torchvision import transforms  # noqa: E402

from vision_lab.cascade.patchcore import load_patchcore  # noqa: E402

torch.set_num_threads(2)

NEW = ["capsule", "carpet", "grid", "leather", "metal_nut",
       "screw", "toothbrush", "wood", "zipper"]
DATASET = "mvtec_ad"
MODEL_CACHE = Path(os.environ.get("VISION_MODEL_CACHE", "model_cache"))
IMAGENET_NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_npz(name: str, mapping: dict) -> None:
    RES.mkdir(parents=True, exist_ok=True)
    tmp = RES / (name + ".tmp.npz")
    np.savez_compressed(tmp, **mapping)
    tmp.replace(RES / (name + ".npz"))


def load_npz_if_any(name: str) -> dict:
    path = RES / (name + ".npz")
    if not path.exists():
        return {}
    z = np.load(path, allow_pickle=False)
    return {k: z[k].astype(np.float32) for k in z.files}


def pooled_features(spatial) -> torch.Tensor:
    return torch.cat([
        torch.nn.functional.adaptive_avg_pool2d(spatial, 1).flatten(),
        torch.nn.functional.adaptive_avg_pool2d(spatial, 2).flatten(),
    ])


class Encoders:
    def __init__(self):
        import clip
        from torchvision.models import resnet18

        self.clip_lib = clip
        # CLIP RN50 is the model the existing local scorer used
        self.clip_model, self.clip_transform = clip.load(
            str(MODEL_CACHE / "clip/RN50.pt"), device="cpu", jit=False)
        self.clip_model.eval()
        resnet = resnet18(weights=None)
        state = torch.load(MODEL_CACHE / "torch/hub/checkpoints/resnet18-f37072fd.pth",
                           map_location="cpu", weights_only=True)
        resnet.load_state_dict(state)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2]).eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(*IMAGENET_NORM),
        ])

    def image_vector(self, rgb):
        with torch.inference_mode():
            v = self.clip_model.encode_image(self.clip_transform(rgb).unsqueeze(0))
            v = v / v.norm(dim=-1, keepdim=True)
            pooled = pooled_features(self.resnet(self.preprocess(rgb).unsqueeze(0)))
        return v[0].cpu().numpy().astype(np.float32), pooled.cpu().numpy().astype(np.float32)


def manifest_rows() -> list[dict]:
    manifest = json.loads((ROOT / "data/public/mvtec_ad/manifest.json").read_text())
    rows = []
    for r in manifest["rows"]:
        if r["category"] not in NEW:
            continue
        parts = r["path"].split("/")
        split, truth = r["split"], r["truth"]
        if split == "train" and truth == "OK":
            role = "fit"
        elif split == "calibration":
            role = "calibration"
        elif split == "train" and truth == "EXCEPTION":
            role = "support_anomaly"
        elif split == "test" and truth == "OK":
            role = "test_good"
        elif split == "test" and truth == "EXCEPTION":
            role = "test_anomaly"
        else:
            continue
        rows.append({
            "dataset": DATASET,
            "category": r["category"],
            "source": r["path"],
            "source_path": str(ROOT / "data/public" / DATASET / r["path"]),
            "sha256": r["sha256"],
            "size": r.get("size"),
            "stratum": parts[-2],
            "truth": "ok" if truth == "OK" else "not_ok",
            "source_split": split,
            "role": role,
            "phash": r.get("phash"),
            "previously_exposed": False,
            "previously_exposed_v31": False,
            "v31_pipeline_unseen": True,
            "historical_surface_category_exposed": True,
            "historical_surface_test_overlap": role in ("test_good", "test_anomaly"),
        })
    return rows


def existing_rows() -> list[dict]:
    return [r for r in load_local_scores() if r["dataset"] == DATASET]


def build_patchcore(category: str, rows: list[dict]):
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    art = RES / "patchcore" / f"{category}_patchcore.pt"
    art.parent.mkdir(parents=True, exist_ok=True)
    if art.exists():
        return load_patchcore(art)
    model = PatchcoreModel(backbone="resnet18", layers=("layer2", "layer3"),
                           pre_trained=True, num_neighbors=1)
    model.train()
    torch.manual_seed(31)
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)), transforms.ToTensor(),
        transforms.Normalize(*IMAGENET_NORM),
    ])
    fit = sorted((r for r in rows if r["role"] == "fit"), key=lambda r: r["source"])
    began = time.perf_counter()
    began_cpu = time.process_time()
    for r in fit:
        with Image.open(r["source_path"]) as source:
            tensor = preprocess(ImageOps.exif_transpose(source).convert("RGB")).unsqueeze(0)
        with torch.inference_mode():
            model(tensor)
    model.subsample_embedding(sampling_ratio=0.01)
    model.eval()
    torch.save({"state_dict": model.state_dict(), "backbone": "resnet18",
                "layers": ("layer2", "layer3"), "image_size": (256, 256)}, art)
    (RES / "patchcore" / f"{category}_fit.json").write_text(json.dumps({
        "category": category, "fit_n": len(fit),
        "fit_sha256": [r["sha256"] for r in fit],
        "fit_cpu_s": time.process_time() - began_cpu,
        "fit_wall_s": time.perf_counter() - began,
    }, indent=2) + "\n")
    return load_patchcore(art)


def main(max_images=None, only_category=None, limit_per_category=None) -> None:
    started = time.time()
    enc = Encoders()
    clip_emb = load_npz_if_any("clip_embeddings")
    res_emb = load_npz_if_any("resnet_embeddings")
    print(json.dumps({"stage": "encoders_loaded", "clip_cached": len(clip_emb),
                      "resnet_cached": len(res_emb)}), flush=True)

    todo = [r for r in existing_rows() if r["sha256"] not in clip_emb]
    if max_images:
        todo = todo[:max_images]
    for i, r in enumerate(todo, 1):
        with Image.open(r["source_path"]) as source:
            rgb = ImageOps.exif_transpose(source).convert("RGB")
            c, v = enc.image_vector(rgb)
        clip_emb[r["sha256"]] = c
        res_emb[r["sha256"]] = v
        if i % 100 == 0:
            save_npz("clip_embeddings", clip_emb)
            save_npz("resnet_embeddings", res_emb)
            print(json.dumps({"stage": "existing_embeddings", "done": i, "total": len(todo)}), flush=True)
    save_npz("clip_embeddings", clip_emb)
    save_npz("resnet_embeddings", res_emb)

    all_new = manifest_rows()
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / "heldout_ad_scores.jsonl"
    previous = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            item = json.loads(line)
            previous[item["sha256"]] = item
    for category in NEW:
        if only_category and category != only_category:
            continue
        rows = [r for r in all_new if r["category"] == category]
        if limit_per_category:
            rows = sorted(rows, key=lambda z: z["source"])[:limit_per_category]
        if not rows:
            continue
        if any(r["sha256"] in previous for r in rows):
            print(json.dumps({"stage": "skip_existing", "category": category}), flush=True)
            continue
        patch = build_patchcore(category, rows)
        with torch.inference_mode():
            text = enc.clip_model.encode_text(enc.clip_lib.tokenize(
                [f"a flawless {category.replace('_', ' ')}",
                 f"a damaged {category.replace('_', ' ')}"]))
            text = text / text.norm(dim=-1, keepdim=True)
        began = time.perf_counter()
        with out_path.open("a") as stream:
            for j, r in enumerate(sorted(rows, key=lambda z: z["source"]), 1):
                if sha(r["source_path"]) != r["sha256"]:
                    raise RuntimeError("raw SHA mismatch " + r["source"])
                with Image.open(r["source_path"]) as source:
                    rgb = ImageOps.exif_transpose(source).convert("RGB")
                    p = enc.preprocess(rgb).unsqueeze(0)
                    c = enc.clip_transform(rgb).unsqueeze(0)
                began_cpu, began_wall = time.process_time(), time.perf_counter()
                with torch.inference_mode():
                    patch_score = float(patch(p).pred_score.flatten()[0])
                    cv = enc.clip_model.encode_image(c)
                    cv = cv / cv.norm(dim=-1, keepdim=True)
                    clip_score = float((cv @ text.T).softmax(dim=-1)[0, 1])
                    pooled = pooled_features(enc.resnet(p))
                cost = {"cpu": time.process_time() - began_cpu,
                        "wall": time.perf_counter() - began_wall}
                clip_emb[r["sha256"]] = cv[0].cpu().numpy().astype(np.float32)
                res_emb[r["sha256"]] = pooled.cpu().numpy().astype(np.float32)
                item = {**r, "patch_score": patch_score, "clip_score": clip_score,
                        "score_cost": cost,
                        "patch_call_id": "independent-heldout-patchcore",
                        "clip_call_id": "independent-heldout-clip-rn50",
                        "answer": "unknown",
                        "reason": "independent research score; dataset labels only"}
                stream.write(json.dumps(item) + "\n")
                stream.flush()
                previous[r["sha256"]] = item
                if j % 20 == 0:
                    save_npz("clip_embeddings", clip_emb)
                    save_npz("resnet_embeddings", res_emb)
                    print(json.dumps({"stage": "heldout_scoring", "category": category, "done": j,
                                      "total": len(rows), "elapsed_s": time.perf_counter() - began}),
                          flush=True)
        save_npz("clip_embeddings", clip_emb)
        save_npz("resnet_embeddings", res_emb)
        print(json.dumps({"stage": "category_done", "category": category}), flush=True)
    print(json.dumps({"stage": "complete", "elapsed_s": time.time() - started,
                      "heldout_rows": len(previous), "clip_embeddings": len(clip_emb),
                      "resnet_embeddings": len(res_emb)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-images", type=int, default=None, help="limit existing-image embeddings (smoke tests)")
    parser.add_argument("--only-category", default=None)
    parser.add_argument("--limit-per-category", type=int, default=None, help="deterministic prefix (smoke tests)")
    args = parser.parse_args()
    main(args.max_images, args.only_category, args.limit_per_category)
