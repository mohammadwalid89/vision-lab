"""Phase 2 surface specialist candidate: PatchCore at the published configuration.

WideResNet-50-2 (ImageNet) layer2 + layer3, 3x3 local aggregation, layer3 upsampled to layer2's grid,
1536 -> 1024 channels (adaptive average pooling, as in the reference implementation), 1 % greedy k-center
coreset on a 128-d random projection, image score = max over patches of the nearest-neighbour distance.
Images are resized to 224x224 without cropping (kit contents and border defects must stay in frame).

Calibration is OUT-OF-FOLD: 5 memory banks are each built on 4/5 of the train good images and score the
remaining 1/5, so every calibration score comes from an image its bank never saw.  The deployed bank uses
all train images.  The baseline's calibration reused the fitting images.

Outputs: models/wrn_<category>.pt (bank, fp16), out/wrn_calibration.json, out/wrn_scores.jsonl (every test /
validation image of the category, with a 28x28 anomaly map) and per-image timings."""

from __future__ import annotations

import base64
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
OUT, MODELS = HERE / "out", HERE / "models"
OUT.mkdir(exist_ok=True)
MODELS.mkdir(exist_ok=True)
CATS = sys.argv[1:] or [
    "bottle",
    "cable",
    "hazelnut",
    "pill",
    "tile",
    "transistor",
    "breakfast_box",
    "pushpins",
    "screw_bag",
]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class WRNFeatures(torch.nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision

        m = torchvision.models.wide_resnet50_2(weights=torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1)
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.layer2, self.layer3 = m.layer2, m.layer3
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        f2 = self.layer2(self.stem(x))
        f3 = self.layer3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.avg_pool2d(f3, 3, 1, 1)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        f = torch.cat([f2, f3], 1)  # B x 1536 x 28 x 28
        B, C, H, W = f.shape
        f = f.permute(0, 2, 3, 1).reshape(B * H * W, 1, C)
        f = F.adaptive_avg_pool1d(f, 1024).reshape(B, H * W, 1024)  # B x 784 x 1024
        return f


def load(path):
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(im, np.float32) / 255.0).permute(2, 0, 1)[None]
    return (x - MEAN) / STD


def embed(net, paths, bs=8):
    out = []
    for i in range(0, len(paths), bs):
        out.append(net(torch.cat([load(p) for p in paths[i : i + bs]])).half())
    return torch.cat(out)  # N x 784 x 1024 (fp16)


def coreset(E, ratio=0.01, min_n=1000, seed=0):
    """Greedy k-center on a 128-d Gaussian random projection (approximate, as in PatchCore)."""
    g = torch.Generator().manual_seed(seed)
    n = E.shape[0]
    k = min(n, max(min_n, int(ratio * n)))
    P = torch.randn(E.shape[1], 128, generator=g) / 128**0.5
    Z = torch.cat([E[i : i + 65536].float() @ P for i in range(0, n, 65536)])
    sq = (Z * Z).sum(1)
    sel = [int(torch.randint(n, (1,), generator=g))]
    d = sq - 2 * (Z @ Z[sel[-1]]) + sq[sel[-1]]  # squared distances (norm expansion)
    for _ in range(k - 1):
        j = int(torch.argmax(d))
        sel.append(j)
        d = torch.minimum(d, sq - 2 * (Z @ Z[j]) + sq[j])
    return E[torch.tensor(sel)].clone()


def nn_dist(F_img, bank):
    """F_img: P x 1024 (fp16), bank: K x 1024 (fp16) -> P nearest-neighbour L2 distances."""
    b = bank.float()
    f = F_img.float()
    out = []
    for i in range(0, f.shape[0], 1024):
        out.append(torch.cdist(f[i : i + 1024], b).min(1).values)
    return torch.cat(out)


def main():
    torch.set_num_threads(int(os.environ.get("P2_THREADS", "3")))
    inv = json.load(open(OUT / "inventory.json"))
    net = WRNFeatures()
    calib = json.loads((OUT / "wrn_calibration.json").read_text()) if (OUT / "wrn_calibration.json").exists() else {}
    fout = open(OUT / "wrn_scores.jsonl", "a")
    for cat in CATS:
        t0 = time.time()
        train = [r["path"] for r in inv if r["category"] == cat and r["split"] == "train"]
        evalr = [r for r in inv if r["category"] == cat and r["split"] in ("test", "validation")]
        E = embed(net, train)  # N x 784 x 1024
        N = E.shape[0]
        rng = np.random.default_rng(0)
        folds = rng.permutation(N) % 5
        oof = np.zeros(N)
        for k in range(5):
            tr, te = np.where(folds != k)[0], np.where(folds == k)[0]
            bank = coreset(E[torch.from_numpy(tr)].reshape(-1, 1024), seed=k)
            for i in te:
                oof[i] = float(nn_dist(E[i], bank).max())
        bank = coreset(E.reshape(-1, 1024), seed=99)
        torch.save(
            {"bank": bank, "backbone": "wide_resnet50_2", "layers": ["layer2", "layer3"], "size": 224, "dim": 1024},
            MODELS / f"wrn_{cat}.pt",
        )
        del E
        calib[cat] = {"oof_scores": oof.tolist(), "n_train": N, "bank_size": int(bank.shape[0])}
        (OUT / "wrn_calibration.json").write_text(json.dumps(calib))
        t_fit = time.time() - t0
        for r in evalr:
            w0, c0 = time.perf_counter(), time.process_time()
            f = net(load(r["path"]))[0].half()
            w1, c1 = time.perf_counter(), time.process_time()
            d = nn_dist(f, bank)
            w2, c2 = time.perf_counter(), time.process_time()
            m = d.reshape(28, 28).numpy()
            q = np.clip(m / max(1e-6, float(np.quantile(oof, 0.95))) / 3 * 255, 0, 255).astype(np.uint8)
            fout.write(
                json.dumps(
                    {
                        "sha256": r["sha256"],
                        "category": cat,
                        "split": r["split"],
                        "type": r["type"],
                        "path": r["path"],
                        "exposed": r["exposed"],
                        "score": float(d.max()),
                        "map_q": base64.b64encode(q.tobytes()).decode(),
                        "ms": {
                            "embed_wall": 1000 * (w1 - w0),
                            "embed_cpu": 1000 * (c1 - c0),
                            "knn_wall": 1000 * (w2 - w1),
                            "knn_cpu": 1000 * (c2 - c1),
                        },
                    }
                )
                + "\n"
            )
            fout.flush()
        print(
            json.dumps(
                {
                    "category": cat,
                    "n_train": N,
                    "bank": int(bank.shape[0]),
                    "fit_s": round(t_fit, 1),
                    "eval_n": len(evalr),
                    "total_s": round(time.time() - t0, 1),
                    "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
