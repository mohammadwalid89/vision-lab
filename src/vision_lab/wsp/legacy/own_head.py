"""Small programmed presence/count/position head plus learned feature probes.

These are counts/positions of learned token responses, NOT annotated counts of
physical objects. No marker flag, filename, image ID or category is an input.
"""

import numpy as np
import torch
from torch import nn

from .experiments import array_hash


class ProgrammedHead(nn.Module):
    def __init__(self, width, channels=4):
        super().__init__()
        self.width = width
        self.channels = channels
        # Learned local probes shared across the 4x4 grid, plus a learned output layer.
        self.probe = nn.Linear(width, channels)
        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(-1, 1, 4), torch.linspace(-1, 1, 4), indexing='ij'
            ),
            -1,
        ).reshape(16, 2)
        # The fixed logic, stored as buffers: count, x moment, y moment.
        self.register_buffer(
            'logic_weights',
            torch.cat([torch.ones(16, 1), coords[:, [1]], coords[:, [0]]], 1),
        )
        self.register_buffer('presence_count_threshold', torch.tensor(0.5))
        self.register_buffer('probe_threshold', torch.tensor(2.0))
        self.register_buffer('probe_sharpness', torch.tensor(4.0))
        self.output = nn.Linear(2 * width + channels * 4, 1)
        if sum(p.numel() * p.element_size() for p in self.parameters()) >= 100_000_000:
            raise ValueError('Head exceeds 100 MB')

    def forward(self, x):
        base = x[:, : 2 * self.width]
        grid = x[:, 2 * self.width :].reshape(-1, 16, self.width)
        response = torch.sigmoid(
            self.probe_sharpness * (self.probe(grid) - self.probe_threshold)
        )
        logic = self.logic_features(response)
        return self.output(torch.cat([base, logic], 1)).squeeze(-1)

    def logic_features(self, response):
        """Presence, count and x/y response moments of the probe responses."""
        moments = torch.einsum('ngk,gj->nkj', response, self.logic_weights)
        count = moments[:, :, 0]
        presence = torch.sigmoid(8 * (count - self.presence_count_threshold))
        return torch.cat([presence[:, :, None], moments / 16], -1).flatten(1)


def make_head(width, seed):
    torch.manual_seed(seed)
    return ProgrammedHead(width)


def flat_weights(model):
    return np.concatenate(
        [p.detach().numpy().reshape(-1) for p in model.parameters()]
    ).astype('float64')


def set_weights(model, theta):
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(
                torch.tensor(theta[offset : offset + n].reshape(p.shape), dtype=p.dtype)
            )
            offset += n
    if offset != len(theta):
        raise ValueError('Parameter vector mismatch')


def program_hash(model):
    return array_hash(np.concatenate([b.numpy().reshape(-1) for b in model.buffers()]))


def train_program(x, y, weights, width, seed, steps=200, lr=0.03, l2=0.001):
    model = make_head(width, seed)
    x = torch.as_tensor(x, dtype=torch.float32)
    y = torch.as_tensor(y, dtype=torch.float32)
    w = torch.as_tensor(weights, dtype=torch.float32)
    w = w / w.sum()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=l2)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (
            torch.nn.functional.binary_cross_entropy_with_logits(
                model(x), y, reduction='none'
            )
            * w
        ).sum()
        loss.backward()
        optimizer.step()
    return model
