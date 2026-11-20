"""Shared code for the deletion probe and the threshold re-anchor probe.

Reads the cached Round 1 features (SigLIP2 pooled and mean-patch, last layer)
and the Round 1 iteration-1 linear heads; learns deletion masks and signed
sparse deltas on a linear head or a small MLP; retrains at k sources.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from vision_lab.wsp.legacy.r2_methods import ordered_sources

SEEDS = (20261112, 20261113, 20261114)
K = 32
# Fractions of maskable head weights that may be set to zero / may change.
BUDGETS = (0.01, 0.05, 0.10, 0.25, 0.50)
PRIMARY_BUDGET = 0.10
RANDOM_DRAWS = 3

# Mask-arm optimisation, fixed in the preregistration after a TRAIN-only pilot.
EPOCHS = 300
LR_SCORE = 0.05
LR_DELTA = 0.01
DELTA_WEIGHT_DECAY = 1e-4
TAU0, TAU1 = 1.0, 0.2
MLP_HIDDEN = 32
MLP_STEPS = 400
MLP_LR = 1e-3
MLP_L2 = 1e-4
RETRAIN_LINEAR_STEPS = 2000
RETRAIN_LINEAR_LR = 0.03
RETRAIN_LINEAR_L2 = 0.001


def sha256_of(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


class Data:
    """Read-only access to the Round 1 cache, scaler, linear heads and thresholds.

    `root` is the Round 1 experiment tree: `r1/cache/siglip2_base/`,
    `r1/fits/i1/linear/` and `r1/INTAKE.json` under it.
    """

    def __init__(self, root):
        self.root = Path(root)
        cache = self.root / 'r1/cache/siglip2_base'
        fits = self.root / 'r1/fits/i1/linear'
        self.index = json.loads((cache / 'index.json').read_text())
        if self.index['status'] != 'READY':
            raise RuntimeError('Cache not READY')
        self.rows = self.index['rows']
        if any(r['split'] not in ('train', 'dev') for r in self.rows):
            raise PermissionError('Active splits only')
        self.pooled = np.load(cache / 'pooled.npy', mmap_mode='r')
        self.mean_patch = np.load(cache / 'mean_patch.npy', mmap_mode='r')
        self.st = np.load(fits / 'standardizer.npz')
        self.thresholds = {}
        self.A = {}
        for s in SEEDS:
            self.A[s] = np.load(fits / f'{s}_fit.npz')['A'].astype(np.float64)
            self.thresholds[s] = json.loads((fits / f'{s}_threshold.json').read_text())
        # TRAIN / DEV row indices, in cache order.
        self.ti = np.array(
            [i for i, r in enumerate(self.rows) if r['split'] == 'train']
        )
        self.di = np.array([i for i, r in enumerate(self.rows) if r['split'] == 'dev'])
        self.tr = [self.rows[i] for i in self.ti]
        self.dr = [self.rows[i] for i in self.di]
        if any(r['split'] != 'dev' for r in self.dr):
            raise PermissionError('DEV reader only')
        self._train_pairs = {}
        for i, r in enumerate(self.rows):
            if r['split'] == 'train':
                self._train_pairs.setdefault(r['group_id'], {})[r['variant']] = i
        if any(set(v) != {'clean', 'marker'} for v in self._train_pairs.values()):
            raise ValueError('TRAIN pair alignment')
        # The k=32 source lists are re-derived from the Round 1 intake.
        intake = json.loads((self.root / 'r1/INTAKE.json').read_text())['rows']
        train = [r for r in intake if r['split'] == 'train']
        self.subsets = {}
        for s in SEEDS:
            self.subsets[s] = [
                dict(
                    group_id=r['group_id'],
                    source_id=r['source_id'],
                    label=r['label'],
                    category=r['category'],
                )
                for r in ordered_sources(train, s)[:K]
            ]

    def base(self, ids):
        ids = np.asarray(ids)
        x = np.concatenate(
            [
                self.pooled[ids, -1].astype('float64'),
                self.mean_patch[ids, -1].astype('float64'),
            ],
            1,
        )
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite base features')
        return x

    def scaled(self, ids):
        x = self.base(ids)
        return (
            ((x - self.st['mean']) / self.st['scale'])
            .astype('float32')
            .astype('float64')
        )

    def k_pairs(self, seed):
        items = self.subsets[seed]
        ci = np.array([self._train_pairs[r['group_id']]['clean'] for r in items])
        mi = np.array([self._train_pairs[r['group_id']]['marker'] for r in items])
        y = np.array([r['label'] for r in items], dtype=np.float64)
        return ci, mi, y

    def all_train_pair_ids(self, exclude_groups=()):
        ex = set(exclude_groups)
        out = []
        for g, v in self._train_pairs.items():
            if g not in ex:
                out.append((v['clean'], v['marker'], self.rows[v['clean']]['label']))
        out.sort()
        return out


def sigmoid_np(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def _hard_keep(score, w_abs, r):
    """Delete the r entries with lowest (score, |w|, index); exact count."""
    d = len(score)
    order = np.lexsort((np.arange(d), w_abs, score))
    keep = np.ones(d, dtype=bool)
    if r:
        keep[order[:r]] = False
    return keep


def _ste(keep_hard, score, tau):
    import torch

    soft = torch.sigmoid(score / tau)
    return torch.as_tensor(keep_hard, dtype=soft.dtype) + soft - soft.detach()


class LinearMask:
    """Learned keep/zero mask (DEL) or signed sparse delta (SIGNED) on a linear head."""

    def __init__(self, w, b, kind, r, seed):
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.w = torch.as_tensor(w, dtype=torch.float64)
        self.b = torch.as_tensor(b, dtype=torch.float64)
        self.w_abs = np.abs(w)
        self.kind = kind
        self.r = r
        self.score = torch.zeros(len(w), dtype=torch.float64, requires_grad=True)
        if kind == 'SIGNED':
            self.delta = torch.zeros(len(w), dtype=torch.float64, requires_grad=True)
        else:
            self.delta = None

    def weights(self, tau):
        kh = _hard_keep(self.score.detach().numpy(), self.w_abs, self.r)
        keep = _ste(kh, self.score, tau)
        if self.kind == 'DEL':
            return self.w * keep
        return self.w + self.delta * keep

    def fit(
        self,
        x,
        y,
        epochs=EPOCHS,
        lr_score=LR_SCORE,
        lr_delta=LR_DELTA,
        tau0=TAU0,
        tau1=TAU1,
    ):
        t = self.torch
        x = t.as_tensor(x, dtype=t.float64)
        y = t.as_tensor(y, dtype=t.float64)
        groups = [dict(params=[self.score], lr=lr_score)]
        if self.kind == 'SIGNED':
            groups.append(
                dict(params=[self.delta], lr=lr_delta, weight_decay=DELTA_WEIGHT_DECAY)
            )
        opt = t.optim.Adam(groups)
        lossf = t.nn.BCEWithLogitsLoss()
        budget = float(t.linalg.norm(self.w))
        for e in range(epochs):
            # Temperature anneals geometrically from tau0 to tau1.
            tau = tau0 * (tau1 / tau0) ** (e / max(1, epochs - 1))
            opt.zero_grad()
            w = self.weights(tau)
            loss = lossf(x @ w + self.b, y)
            loss.backward()
            opt.step()
            if self.delta is not None:
                # Project the signed delta onto the norm ball of the original weights.
                with t.no_grad():
                    n = float(t.linalg.norm(self.delta))
                    if n > budget:
                        self.delta.mul_(budget / n)
        if not np.isfinite(self.score.detach().numpy()).all():
            raise ArithmeticError('Nonfinite mask score')

    def final_weights(self):
        return self.weights(TAU1).detach().numpy().copy()

    def final_keep(self):
        return _hard_keep(self.score.detach().numpy(), self.w_abs, self.r)


class MLPMask:
    """Learned mask / signed sparse delta on the hidden layer of a 1536-32-1 MLP."""

    def __init__(self, W1, b1, W2, b2, kind, r, seed):
        import torch

        torch.manual_seed(seed)
        self.torch = torch
        self.W1 = torch.as_tensor(W1, dtype=torch.float64)
        self.b1 = torch.as_tensor(b1, dtype=torch.float64)
        self.W2 = torch.as_tensor(W2, dtype=torch.float64)
        self.b2 = torch.as_tensor(b2, dtype=torch.float64)
        self.w_abs = np.abs(W1.reshape(-1))
        self.kind = kind
        self.r = r
        self.score = torch.zeros(
            self.w_abs.size, dtype=torch.float64, requires_grad=True
        )
        if kind == 'SIGNED':
            self.delta = torch.zeros(
                self.W1.shape, dtype=torch.float64, requires_grad=True
            )
        else:
            self.delta = None

    def forward(self, x, tau):
        t = self.torch
        kh = _hard_keep(self.score.detach().numpy(), self.w_abs, self.r)
        keep = _ste(kh, self.score, tau).reshape(self.W1.shape)
        W1 = self.W1 * keep if self.kind == 'DEL' else self.W1 + self.delta * keep
        h = t.tanh(x @ W1 + self.b1)
        return h @ self.W2 + self.b2

    def fit(
        self,
        x,
        y,
        epochs=EPOCHS,
        lr_score=LR_SCORE,
        lr_delta=LR_DELTA,
        tau0=TAU0,
        tau1=TAU1,
    ):
        t = self.torch
        x = t.as_tensor(x, dtype=t.float64)
        y = t.as_tensor(y, dtype=t.float64)
        groups = [dict(params=[self.score], lr=lr_score)]
        if self.kind == 'SIGNED':
            groups.append(
                dict(params=[self.delta], lr=lr_delta, weight_decay=DELTA_WEIGHT_DECAY)
            )
        opt = t.optim.Adam(groups)
        lossf = t.nn.BCEWithLogitsLoss()
        budget = float(t.linalg.norm(self.W1))
        for e in range(epochs):
            tau = tau0 * (tau1 / tau0) ** (e / max(1, epochs - 1))
            opt.zero_grad()
            loss = lossf(self.forward(x, tau), y)
            loss.backward()
            opt.step()
            if self.delta is not None:
                with t.no_grad():
                    n = float(t.linalg.norm(self.delta))
                    if n > budget:
                        self.delta.mul_(budget / n)
        if not np.isfinite(self.score.detach().numpy()).all():
            raise ArithmeticError('Nonfinite mask score')

    def final_keep(self):
        return _hard_keep(self.score.detach().numpy(), self.w_abs, self.r)


def mlp_forward_np(theta, x):
    """theta: flattened [W1(1536,H), b1(H), W2(H), b2(1)]."""
    D = 1536
    H = (len(theta) - 1) // (D + 2)
    W1 = theta[: D * H].reshape(D, H)
    b1 = theta[D * H : D * H + H]
    W2 = theta[D * H + H : D * H + 2 * H]
    b2 = theta[-1]
    h = np.tanh(x @ W1 + b1)
    return h @ W2 + b2


def mlp_init(seed):
    import torch

    torch.manual_seed(seed)
    D, H = 1536, MLP_HIDDEN
    W1 = torch.empty(D, H, dtype=torch.float64)
    torch.nn.init.kaiming_uniform_(W1, a=5**0.5)
    b1 = (torch.rand(H, dtype=torch.float64) - 0.5) * 2 / (H**0.5)
    W2 = torch.empty(H, dtype=torch.float64)
    bound = 1 / (H**0.5)
    torch.nn.init.uniform_(W2, -bound, bound)
    b2 = torch.zeros(1, dtype=torch.float64)
    return W1.numpy().copy(), b1.numpy().copy(), W2.numpy().copy(), b2.numpy().copy()


def train_mlp(x, y, seed, steps=MLP_STEPS, lr=MLP_LR, l2=MLP_L2, init=None, warm=None):
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(x, dtype=torch.float64)
    y = torch.as_tensor(y, dtype=torch.float64)
    if warm is None:
        W1, b1, W2, b2 = mlp_init(seed) if init is None else init
        params = [
            torch.tensor(v, dtype=torch.float64, requires_grad=True)
            for v in (W1, b1, W2, b2)
        ]
    else:
        params = [
            torch.tensor(
                np.asarray(v, dtype=np.float64).copy(),
                dtype=torch.float64,
                requires_grad=True,
            )
            for v in warm
        ]
    opt = torch.optim.Adam(params, lr=lr, weight_decay=l2)
    lossf = torch.nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad()
        h = torch.tanh(x @ params[0] + params[1])
        loss = lossf(h @ params[2] + params[3], y)
        loss.backward()
        opt.step()
    out = [p.detach().numpy().copy() for p in params]
    if not all(np.isfinite(v).all() for v in out):
        raise ArithmeticError('Nonfinite MLP')
    return out


def mlp_flat(params):
    W1, b1, W2, b2 = params
    return np.concatenate([W1.reshape(-1), b1, W2, b2])


def retrain_linear_head(
    x,
    y,
    a,
    seed,
    steps=RETRAIN_LINEAR_STEPS,
    lr=RETRAIN_LINEAR_LR,
    l2=RETRAIN_LINEAR_L2,
):
    """Full-batch gradient descent warm-started from A (Round 2 retrain recipe)."""
    theta = np.array(a, dtype=np.float64, copy=True)
    n = len(y)
    for _ in range(steps):
        p = sigmoid_np(x @ theta[:-1] + theta[-1])
        err = (p - y) / n
        theta[:-1] -= lr * (x.T @ err + l2 * theta[:-1])
        theta[-1] -= lr * err.sum()
    if not np.isfinite(theta).all():
        raise ArithmeticError('Nonfinite retrain')
    return theta


def logits_np(x, theta):
    return np.asarray(x, dtype=np.float64) @ theta[:-1] + theta[-1]
