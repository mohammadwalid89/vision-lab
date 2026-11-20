"""Slot 3 — MLP hidden-layer deletion at fixed 5 % with a clean-score-preserving objective.

Fully specified in advance (loss and coefficient, unit axis, selection, optimizer schedule):

  head        1536-32-1 tanh MLP A_mlp, rebuilt on the fitting pool (no calibration sources)
  unit axis   individual entries of W1 (1536 x 32 = 49,152); b1, W2, b2 frozen
  budget      r = round(0.05 * 49,152) = 2,458 deletions, exactly, deletion only
  selection   learned score s (init 0); hard support = keep the top D - r entries by
              (s desc, |W1| desc, flat index asc) — at init the r smallest-|W1| go
  gate        straight-through: keep = hard + sigmoid(s/tau) - stopgrad(sigmoid(s/tau))
  schedule    tau_e = 1.0 * (0.2/1.0)^(e/(E-1)), E = 300 full-batch epochs, Adam lr 0.05 on s
  loss        mean BCE over the 2k purchased views (k clean + k marker, true labels)
              + lam_pres * mean_{p in P} (f_edit(x_p) - f_A(x_p))^2,     lam_pres = 1.0
  P           2,048 CLEAN views of fitting-pool sources (never calibration), chosen by
              sha256('r3|slot3|preserve|' + source_id) order; unlabeled — only A's own
              logits are used as targets.

Everything except the preservation term is the deletion probe's DEL-MASK recipe, so the
slot changes ONE factor relative to the exploratory 5 % near-miss (disclosed selection).
lam_pres = 1.0 puts the logit-MSE term on the same order as BCE (~0.3-0.7) when the edit
moves clean logits by ~0.5-1; it is a declared value, never tuned on DEV.

Access parity: P is extra (unlabeled) information.  The matched comparator
``warm_preserve@k`` (heads.retrain_mlp with the same P and lam_pres) gets identical access,
otherwise a pass could be an access effect rather than a deletion effect.

Controls: RANDOM-DEL x3 (same r, seeds seed + 100000*draw), six no-ops (edits.noop_suite),
warm@k, fresh@k, warm_preserve@k.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np

from ..edits import delete, n_deletions, noop_suite, random_deletion_keep
from ..heads import (
    ComputeRecord,
    _Timer,
    _to_torch,
    mlp_copy,
    mlp_forward,
    retrain_mlp,
    torch_mlp,
)


@dataclass(frozen=True)
class DeletionRecipe:
    fraction: float = 0.05
    epochs: int = 300
    tau_start: float = 1.0
    tau_end: float = 0.2
    lr: float = 0.05
    lam_pres: float = 1.0
    preserve_n: int = 2048
    preserve_namespace: str = "r3|slot3|preserve|"
    random_draws: int = 3

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def select_preserve_sources(candidate_source_ids, n: int, namespace: str, calibration=frozenset()) -> list:
    ids = [s for s in candidate_source_ids if s not in calibration]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate source ids")
    ranked = sorted(ids, key=lambda s: (hashlib.sha256((namespace + s).encode()).hexdigest(), s))
    return ranked[:n]


def _hard_keep(s_flat, absw_flat, r):
    D = len(s_flat)
    order = np.lexsort((np.arange(D), -absw_flat, -s_flat))
    keep = np.zeros(D, dtype=bool)
    keep[order[: D - r]] = True
    return keep


def learn_deletion_mask(params_A, x_k, y_k, preserve_x, recipe: DeletionRecipe = DeletionRecipe(),
                        seed: int = 0, lam_pres: float | None = None):
    """Learn the slot-3 keep mask on W1.  Returns (keep (d,h) bool, info dict)."""
    import torch
    torch.manual_seed(seed)
    lam = recipe.lam_pres if lam_pres is None else lam_pres
    W1 = np.asarray(params_A["W1"], dtype=np.float64)
    D = W1.size
    r = n_deletions(recipe.fraction, D)
    absw = np.abs(W1).reshape(-1)
    tp = _to_torch(params_A, requires_grad=False)
    X = torch.tensor(np.asarray(x_k, dtype=np.float64))
    Y = torch.tensor(np.asarray(y_k, dtype=np.float64))
    P = torch.tensor(np.asarray(preserve_x, dtype=np.float64)) if lam > 0 else None
    zA = torch.tensor(mlp_forward(params_A, preserve_x)) if lam > 0 else None
    s = torch.zeros(D, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([s], lr=recipe.lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    E = recipe.epochs
    hist = []
    with _Timer() as t:
        for e in range(E):
            tau = recipe.tau_start * (recipe.tau_end / recipe.tau_start) ** (e / max(1, E - 1))
            hard = torch.tensor(_hard_keep(s.detach().numpy(), absw, r), dtype=torch.float64)
            soft = torch.sigmoid(s / tau)
            keep = hard + soft - soft.detach()
            W1e = tp["W1"] * keep.view(W1.shape)
            l_bce = bce(torch_mlp(tp, X, W1e), Y)
            l_pres = ((torch_mlp(tp, P, W1e) - zA) ** 2).mean() if P is not None else torch.zeros((), dtype=torch.float64)
            loss = l_bce + lam * l_pres
            opt.zero_grad()
            loss.backward()
            opt.step()
            if e in (0, E - 1) or e % 50 == 0:
                hist.append(dict(epoch=e, tau=tau, bce=float(l_bce.detach()), pres=float(l_pres.detach()),
                                 loss=float(loss.detach())))
    keep = _hard_keep(s.detach().numpy(), absw, r).reshape(W1.shape)
    if int((~keep).sum()) != r:
        raise AssertionError("deletion count drifted")
    n_ex = len(Y) + (0 if P is None else len(P))
    rec = ComputeRecord("slot3_delmask", E, n_ex, D, t.wall, t.cpu, 6.0 * E * n_ex * D)
    return keep, dict(r=r, D=D, lam_pres=lam, recipe_sha256=recipe.digest(), history=hist,
                      compute=rec.as_dict())


def apply_keep(params, keep):
    q = mlp_copy(params)
    q["W1"] = delete(q["W1"], keep)
    return q


def build_slot3_arms(params_A, x_k, y_k, preserve_x, seed: int, recipe: DeletionRecipe = DeletionRecipe()) -> dict:
    """All slot-3 arms for one seed and one k: candidate, placebos, no-ops, comparators.

    Returns {arm_name: dict(params=..., role=..., compute=..., extra=...)}.  No-op entries
    carry their own reference params (dtype_roundtrip compares against a snapped A).
    """
    arms = {}
    keep, info = learn_deletion_mask(params_A, x_k, y_k, preserve_x, recipe, seed)
    arms["slot3_delmask"] = dict(params=apply_keep(params_A, keep), role="candidate",
                                 compute=info["compute"], extra=dict(r=info["r"], history=info["history"]))
    for d in range(1, recipe.random_draws + 1):
        rk = random_deletion_keep(params_A["W1"].shape, info["r"], seed + 100000 * d)
        arms[f"random_del_{d}"] = dict(params=apply_keep(params_A, rk), role="placebo", compute=None, extra={})
    for name, (ref, q) in noop_suite(params_A, support=~keep).items():
        arms[f"noop_{name}"] = dict(params=q, reference=ref, role="noop", compute=None, extra={})
    for mode in ("warm", "fresh"):
        p, c = retrain_mlp(x_k, y_k, params_A, seed=seed, mode=mode)
        arms[f"{mode}@k"] = dict(params=p, role="retrainer", compute=c.as_dict(), extra={})
    p, c = retrain_mlp(x_k, y_k, params_A, seed=seed, mode="warm", preserve_x=preserve_x, lam_pres=recipe.lam_pres)
    arms["warm_preserve@k"] = dict(params=p, role="retrainer", compute=c.as_dict(), extra={})
    return arms
