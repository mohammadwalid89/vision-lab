"""Download six public image datasets and cache DINOv2-small features for each.

Per image: the CLS token and the mean patch token, concatenated (768 numbers). Up to N
images per dataset, drawn with a fixed seed. Datasets are fetched by torchvision into
<out>/raw; the model is `facebook/dinov2-small` from the Hugging Face hub unless --model
points to a local copy.
"""
import argparse
import time
from pathlib import Path

import numpy as np

SEED = 20261119


def datasets(raw):
    import torchvision as tv
    yield "cifar10", tv.datasets.CIFAR10(raw, train=True, download=True)
    yield "eurosat", tv.datasets.EuroSAT(raw, download=True)
    yield "dtd", tv.datasets.DTD(raw, split="train", download=True)
    yield "gtsrb", tv.datasets.GTSRB(raw, split="train", download=True)
    yield "flowers102", tv.datasets.Flowers102(raw, split="test", download=True)
    yield "pets", tv.datasets.OxfordIIITPet(raw, split="trainval", download=True)


def labels_of(ds):
    for a in ("targets", "_labels", "_samples"):
        if hasattr(ds, a):
            v = getattr(ds, a)
            return np.array([x[1] for x in v] if a == "_samples" else v)
    return np.array([ds[i][1] for i in range(len(ds))])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", help="output directory")
    ap.add_argument("n_per_dataset", type=int)
    ap.add_argument("--model", default="facebook/dinov2-small")
    ap.add_argument("--threads", type=int, default=3)
    a = ap.parse_args()

    import torch
    from transformers import AutoImageProcessor, AutoModel

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    proc = AutoImageProcessor.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).eval()

    def log(m):
        line = f"{time.strftime('%H:%M:%S')} {m}"
        print(line, flush=True)
        with open(out / "extract.log", "a") as f:
            f.write(line + "\n")

    for name, ds in datasets(out / "raw"):
        if (out / f"{name}.npz").exists():
            log(f"{name}: exists, skip")
            continue
        y_all = labels_of(ds)
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.permutation(len(ds))[:a.n_per_dataset])
        feats, ys, t0 = [], [], time.time()
        for k in range(0, len(idx), 32):
            batch = [ds[int(i)][0].convert("RGB") for i in idx[k:k + 32]]
            with torch.no_grad():
                h = model(**proc(images=batch, return_tensors="pt")).last_hidden_state
            feats.append(torch.cat([h[:, 0], h[:, 1:].mean(1)], 1).numpy().astype(np.float32))
            ys.append(y_all[idx[k:k + 32]])
            if k % 640 == 0:
                log(f"{name}: {k + len(batch)}/{len(idx)} "
                    f"({(k + len(batch)) / (time.time() - t0):.1f} img/s)")
        np.savez_compressed(out / f"{name}.npz", X=np.concatenate(feats),
                            y=np.concatenate(ys), idx=idx)
        log(f"{name}: done {len(idx)} images, {len(set(y_all.tolist()))} classes")
    log("ALL DONE")


if __name__ == "__main__":
    main()
