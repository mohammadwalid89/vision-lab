"""Actuator checks — "does the edit move what it claims to move?"

An earlier SigLIP2 edit in this lab had zero effect because the instantiated
path was wrong; that was an invalid actuator, not a negative result.  Before any efficacy
claim each edit must show, on a fixed fixture:
  (a) the intended weights changed            (n_changed > 0)
  (b) the activation right after the edited layer moved
  (c) the output moved
and each no-op must show (c) == 0.  Verdicts:
  NO_EDIT      no weight changed
  WRONG_PATH   weights changed but the edited layer's activation did not move
  INERT        activation moved but output did not (or function-preserving edit)
  ACTIVE       all three moved
"""
from __future__ import annotations

import copy
from typing import Callable

import numpy as np

from .edits import weight_diff_stats
from .heads import mlp_forward

TOL = 1e-10


def _verdict(n_changed, d_act, d_out, tol=TOL):
    if n_changed == 0:
        return "NO_EDIT"
    if d_act is not None and d_act <= tol:
        return "WRONG_PATH"
    if d_out <= tol:
        return "INERT"
    return "ACTIVE"


def head_actuator(params_before, params_after, x) -> dict:
    """For the numpy MLP heads (W1 edits act on the hidden pre-activations)."""
    z0, h0 = mlp_forward(params_before, x, return_hidden=True)
    z1, h1 = mlp_forward(params_after, x, return_hidden=True)
    w = weight_diff_stats(params_before, params_after)
    d_act = float(np.max(np.abs(np.sort(h1, axis=1) - np.sort(h0, axis=1))))  # permutation-invariant
    d_act_raw = float(np.max(np.abs(h1 - h0)))
    d_out = float(np.max(np.abs(z1 - z0)))
    return dict(weights=w, max_abs_hidden_change=d_act_raw, max_abs_hidden_change_sorted=d_act,
                max_abs_output_change=d_out,
                verdict=_verdict(w["n_changed"], d_act_raw, d_out))


def module_actuator(model, layer: str, inputs, edit_fn: Callable, output_fn: Callable | None = None) -> dict:
    """Torch version: deep-copies ``model``, applies ``edit_fn(copy)`` in place, and compares
    the output of submodule ``layer`` and the model output on ``inputs``."""
    import torch

    def run(m):
        cap = {}
        sub = dict(m.named_modules())[layer]
        hook = sub.register_forward_hook(lambda mod, i, o: cap.__setitem__("a", o[0] if isinstance(o, tuple) else o))
        try:
            with torch.no_grad():
                out = m(**inputs) if isinstance(inputs, dict) else m(inputs)
        finally:
            hook.remove()
        out = output_fn(out) if output_fn else out
        return cap["a"].detach().double(), out.detach().double()

    edited = copy.deepcopy(model)
    edit_fn(edited)
    n_changed = 0
    for (n0, p0), (n1, p1) in zip(model.state_dict().items(), edited.state_dict().items()):
        n_changed += int((p0 != p1).sum())
    a0, o0 = run(model)
    a1, o1 = run(edited)
    d_act = float((a1 - a0).abs().max())
    d_out = float((o1 - o0).abs().max())
    return dict(n_changed=n_changed, max_abs_layer_change=d_act, max_abs_output_change=d_out,
                verdict=_verdict(n_changed, d_act, d_out))


def input_actuator(encode: Callable, image_a, image_b) -> dict:
    """Do two inputs (e.g. clean vs marked, after a transform) give different features?"""
    fa = {k: np.asarray(v, dtype=np.float64) for k, v in encode(image_a).items()}
    fb = {k: np.asarray(v, dtype=np.float64) for k, v in encode(image_b).items()}
    pix = int(np.sum(np.asarray(image_a) != np.asarray(image_b)))
    return dict(pixels_different=pix,
                max_abs_feature_change={k: float(np.max(np.abs(fa[k] - fb[k]))) for k in fa})
