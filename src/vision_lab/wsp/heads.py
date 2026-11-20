"""Heads and matched-k retrainers.

Linear (numpy, exact copies of the R1/R2 recipes used by every probe):
  * ``Standardizer``          R1 experiments.Standardizer (float32 transform)
  * ``train_linear_A``        R1 A recipe: 200 full-batch steps, lr .03, l2 .001,
                              init default_rng(seed).normal(0, .001, d+1), optional sample weight
  * ``retrain_linear_warm``   R2 matched-k warm retrain: 2000 steps from A, lr .03, l2 .001
  * ``retrain_linear_fresh``  same schedule from the A-style random init

MLP (torch training, numpy parameters ``{"W1": (d,h), "b1": (h,), "W2": (h,), "b2": ()}``):
  * ``train_mlp_A``           1536-32-1 tanh, Adam 1e-3, weight decay 1e-4, 400 full-batch
                              steps (deletion-probe A_mlp recipe)
  * ``retrain_mlp``           warm (from A) or fresh, Adam 1e-3, wd 1e-4, 400 steps on the
                              k counterfactual views; optional clean-score preservation term
                              (the access-matched comparator for slot 3)
Every fit returns a ``ComputeRecord`` (steps, wall and process-CPU seconds, approx FLOPs)
so equal-allowance and equal-consumed-compute comparisons can be reported.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class ComputeRecord:
    name: str
    steps: int
    n_examples: int
    n_params: int
    wall_s: float
    cpu_s: float
    approx_flops: float

    def as_dict(self):
        return asdict(self)


class _Timer:
    def __enter__(self):
        self.w, self.c = time.perf_counter(), time.process_time()
        return self

    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.w
        self.cpu = time.process_time() - self.c


class Standardizer:
    def fit(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.mean = x.mean(0)
        self.scale = np.maximum(x.std(0), 1e-5)
        return self

    def transform(self, x):
        return ((np.asarray(x, dtype=np.float64) - self.mean) / self.scale).astype(np.float32)


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -np.asarray(z, dtype=np.float64)))


def linear_logits(x, theta):
    return np.asarray(x, dtype=np.float64) @ theta[:-1] + theta[-1]


def _gd_linear(x, y, theta, steps, lr, l2, w):
    for _ in range(steps):
        err = (sigmoid(x @ theta[:-1] + theta[-1]) - y) * w
        theta[:-1] -= lr * (x.T @ err + l2 * theta[:-1])
        theta[-1] -= lr * err.sum()
    if not np.isfinite(theta).all():
        raise ArithmeticError("non-finite linear head")
    return theta


def train_linear_A(x, y, seed, steps=200, lr=0.03, l2=0.001, sample_weight=None):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
    w = w / w.sum()
    theta = np.random.default_rng(seed).normal(0, 0.001, x.shape[1] + 1)
    with _Timer() as t:
        theta = _gd_linear(x, y, theta, steps, lr, l2, w)
    return theta, ComputeRecord("linear_A", steps, len(y), theta.size, t.wall, t.cpu, 4.0 * steps * x.size)


def retrain_linear_warm(x, y, theta_A, steps=2000, lr=0.03, l2=0.001):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.array(theta_A, dtype=np.float64, copy=True)
    with _Timer() as t:
        theta = _gd_linear(x, y, theta, steps, lr, l2, np.full(len(y), 1.0 / len(y)))
    return theta, ComputeRecord("linear_warm@k", steps, len(y), theta.size, t.wall, t.cpu, 4.0 * steps * x.size)


def retrain_linear_fresh(x, y, seed, steps=2000, lr=0.03, l2=0.001):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    theta = np.random.default_rng(seed).normal(0, 0.001, x.shape[1] + 1)
    with _Timer() as t:
        theta = _gd_linear(x, y, theta, steps, lr, l2, np.full(len(y), 1.0 / len(y)))
    return theta, ComputeRecord("linear_fresh@k", steps, len(y), theta.size, t.wall, t.cpu, 4.0 * steps * x.size)


def mlp_forward(params, x, return_hidden=False):
    x = np.asarray(x, dtype=np.float64)
    h = np.tanh(x @ params["W1"] + params["b1"])
    z = h @ params["W2"] + params["b2"]
    return (z, h) if return_hidden else z


def mlp_copy(params):
    return {k: np.array(v, dtype=np.float64, copy=True) for k, v in params.items()}


def _torch():
    import torch
    return torch


def _to_torch(params, requires_grad=True):
    torch = _torch()
    return {k: torch.tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64,
                            requires_grad=requires_grad) for k, v in params.items()}


def _from_torch(tp):
    return {k: v.detach().cpu().numpy().astype(np.float64) for k, v in tp.items()}


def torch_mlp(tp, x, W1=None):
    torch = _torch()
    W1 = tp["W1"] if W1 is None else W1
    return torch.tanh(x @ W1 + tp["b1"]) @ tp["W2"] + tp["b2"]


def init_mlp(d, h, seed):
    """PyTorch-default-like uniform init, deterministic from ``seed`` (numpy RNG)."""
    rng = np.random.default_rng(seed)
    b1 = 1 / np.sqrt(d)
    b2 = 1 / np.sqrt(h)
    return dict(W1=rng.uniform(-b1, b1, (d, h)), b1=rng.uniform(-b1, b1, h),
                W2=rng.uniform(-b2, b2, h), b2=np.array(rng.uniform(-b2, b2)))


def _adam_fit(tp, loss_fn, steps, lr, wd):
    torch = _torch()
    opt = torch.optim.Adam(list(tp.values()), lr=lr, weight_decay=wd)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(tp)
        loss.backward()
        opt.step()
    out = _from_torch(tp)
    if not all(np.isfinite(v).all() for v in out.values()):
        raise ArithmeticError("non-finite MLP")
    return out


def train_mlp_A(x, y, seed, hidden=32, steps=400, lr=1e-3, wd=1e-4, sample_weight=None):
    torch = _torch()
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(x, dtype=np.float64))
    Y = torch.tensor(np.asarray(y, dtype=np.float64))
    W = None if sample_weight is None else torch.tensor(np.asarray(sample_weight, dtype=np.float64))
    tp = _to_torch(init_mlp(X.shape[1], hidden, seed))
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    def loss_fn(p):
        z = torch_mlp(p, X)
        if W is None:
            return bce(z, Y)
        return (bce(z, Y, reduction="none") * W).sum() / W.sum()

    with _Timer() as t:
        params = _adam_fit(tp, loss_fn, steps, lr, wd)
    n_par = sum(v.size for v in params.values())
    return params, ComputeRecord("mlp_A", steps, len(Y), n_par, t.wall, t.cpu, 6.0 * steps * len(Y) * n_par)


def retrain_mlp(x_k, y_k, params_A=None, seed=0, mode="warm", steps=400, lr=1e-3, wd=1e-4,
                preserve_x=None, lam_pres=0.0, hidden=32):
    """Matched-k MLP retrainer.  ``mode`` in {'warm','fresh'}.

    With ``preserve_x`` and ``lam_pres>0`` it minimises the SAME objective as slot 3
    (mean BCE on the k views + lam_pres * mean squared logit change vs A on the
    preservation views) with all weights free: the access-matched comparator.
    """
    torch = _torch()
    torch.manual_seed(seed)
    X = torch.tensor(np.asarray(x_k, dtype=np.float64))
    Y = torch.tensor(np.asarray(y_k, dtype=np.float64))
    if mode == "warm":
        if params_A is None:
            raise ValueError("warm retrain needs A")
        start = mlp_copy(params_A)
    elif mode == "fresh":
        start = init_mlp(X.shape[1], hidden if params_A is None else params_A["W1"].shape[1], seed)
    else:
        raise ValueError(mode)
    tp = _to_torch(start)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    P = zA = None
    if preserve_x is not None and lam_pres > 0:
        if params_A is None:
            raise ValueError("preservation needs A")
        P = torch.tensor(np.asarray(preserve_x, dtype=np.float64))
        zA = torch.tensor(mlp_forward(params_A, preserve_x))

    def loss_fn(p):
        loss = bce(torch_mlp(p, X), Y)
        if P is not None:
            loss = loss + lam_pres * ((torch_mlp(p, P) - zA) ** 2).mean()
        return loss

    with _Timer() as t:
        params = _adam_fit(tp, loss_fn, steps, lr, wd)
    n_par = sum(v.size for v in params.values())
    n_ex = len(Y) + (0 if P is None else len(P))
    name = f"mlp_{mode}@k" + ("+preserve" if P is not None else "")
    return params, ComputeRecord(name, steps, n_ex, n_par, t.wall, t.cpu, 6.0 * steps * n_ex * n_par)
