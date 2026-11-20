"""Reload a saved PatchCore memory bank."""
from __future__ import annotations

from pathlib import Path


def load_patchcore(artifact: Path):
    """Rebuild the fitted PatchCore model from its saved state without fetching pretrained weights."""
    import torch
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel

    saved = torch.load(artifact, map_location="cpu", weights_only=True)
    model = PatchcoreModel(backbone=saved["backbone"], layers=tuple(saved["layers"]),
                           pre_trained=False, num_neighbors=1)
    model.load_state_dict(saved["state_dict"])
    return model.eval()
