"""Shared helpers for the KTH video experiments: clip listing, sampling, R3D-18, metrics."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch

LABELS = {"walking": 0, "jogging": 1, "running": 2}
NFRAMES = 16
SIZE = 112


def list_clips(root: Path) -> list[dict]:
    """KTH clips named personNN_<action>_dN_*.avi under root."""
    rows = []
    for p in sorted(Path(root).rglob("*.avi")) + sorted(Path(root).rglob("*.AVI")):
        m = re.match(r"person(\d+)_([a-z]+)_d(\d+)", p.stem.lower())
        if not m or m.group(2) not in LABELS:
            continue
        rows.append({"path": str(p), "name": p.name, "subject": int(m.group(1)),
                     "action": m.group(2), "label": LABELS[m.group(2)], "scenario": int(m.group(3))})
    return rows


def sample_frames(path: str, frac: float = 1.0, n: int = NFRAMES, size: int = SIZE) -> np.ndarray:
    """n frames, uniform over a span of the clip (fraction `frac`, centred), centre-cropped."""
    import cv2

    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise RuntimeError(f"unreadable clip: {Path(path).name}")
    span = min(max(n, int(round(total * frac))), total)
    start = max(0, (total - span) // 2)
    idx = start + np.linspace(0, span - 1, n).round().astype(int)
    frames = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
            ok, frame = cap.read()
        frame = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
        y0 = (128 - size) // 2
        frames.append(cv2.cvtColor(frame[y0:y0 + size, y0:y0 + size], cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames).astype(np.uint8)


def build_model(weights_path: Path | None = None):
    """R3D-18 (Kinetics-400). With weights_path, load a local torchvision state dict."""
    from torchvision.models.video import R3D_18_Weights, r3d_18

    if weights_path is None:
        return r3d_18(weights=R3D_18_Weights.KINETICS400_V1).eval()
    model = r3d_18(weights=None)
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return model.eval()


def batch_tensor(clips: np.ndarray) -> torch.Tensor:
    x = torch.from_numpy(clips).float().permute(0, 4, 1, 2, 3) / 255.0  # N, 3, T, H, W
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(1, 3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(1, 3, 1, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def pooled_features(model, clips: np.ndarray, batch_size: int = 4) -> np.ndarray:
    """512-d average-pooled features, captured before the classification layer."""
    captured = {}
    handle = model.avgpool.register_forward_hook(
        lambda _m, _i, out: captured.__setitem__("f", out.detach().flatten(1)))
    feats = []
    try:
        for i in range(0, len(clips), batch_size):
            model(batch_tensor(clips[i:i + batch_size]))
            feats.append(captured["f"].cpu())
    finally:
        handle.remove()
    return torch.cat(feats).numpy()


def subject_folds(subjects: np.ndarray, k: int = 5) -> list[np.ndarray]:
    return [np.array(f) for f in np.array_split(np.array(sorted(set(subjects))), k)]


def metrics(y, pred, subjects) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    per_subject = {int(s): float(accuracy_score(y[subjects == s], pred[subjects == s]))
                   for s in sorted(set(subjects))}
    return {"accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro")),
            "worst_subject_acc": min(per_subject.values()),
            "per_subject": per_subject,
            "confusion": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist()}


def bootstrap_subject_delta(subjects, correct_a, correct_b, reps: int = 4000, seed: int = 0) -> dict:
    """Paired subject-level bootstrap of the accuracy difference a - b."""
    rng = np.random.default_rng(seed)
    subs = np.array(sorted(set(subjects)))
    deltas = []
    for _ in range(reps):
        pick = rng.choice(subs, len(subs), replace=True)
        idx = np.concatenate([np.where(subjects == s)[0] for s in pick])
        deltas.append(correct_a[idx].mean() - correct_b[idx].mean())
    deltas = np.array(deltas)
    return {"delta": float(correct_a.mean() - correct_b.mean()),
            "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
            "p_le_0": float((deltas <= 0).mean())}
