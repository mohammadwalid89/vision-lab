"""Assemble the declared pipelines of one slot x one seed into harness.PipelineScores.

Shared by scripts/smoke.py and scripts/run_r3.py so the smoke exercises the exact code the
full run will use.  Declared recipe per slot (no variants, nothing chosen on DEV):

  A (every slot)   linear A recipe on [pooled | mean_patch] of the ORIGINAL view, fitted on
                   the fitting pool's confounded views; calibration sources excluded.
  slot1            candidate  = A recipe refit on the sam_meanfill representation (train-
                                and test-time masking: the deployable pipeline)
                   placebos   = A recipe on random_area_{1,2,3}
                   references = padding_only pipeline; test-time-only masking with A's head
                   no-op      = 'keep_all' transform (re-encoded, must equal A bit-for-bit)
  slot2            candidate  = A recipe on the object-only 14x14 aggregate (768-d)
                   placebos   = A recipe on perm_{1,2,3} aggregates
                   references = uniform-weight aggregate; [pooled | object] diagnostic
                   no-op      = A rescored through an independent arithmetic path
  slots 1, 2       retrainers = warm@k / fresh@k on the candidate representation AND on the
                                original representation (gate 5 is against each)
  slot3            A_mlp + hidden_deletion.build_slot3_arms (candidate, 3 random deletions,
                   six no-ops, warm@k, fresh@k, warm_preserve@k)
Feature dict layout: feats[transform][key] with transforms identity, sam_meanfill,
padding_only, random_area_d, keep_all and keys pooled, mean_patch, object, perm_d, uniform.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .calibration import AccessLedger
from .harness import PipelineScores
from .heads import (
    Standardizer,
    linear_logits,
    mlp_forward,
    retrain_linear_fresh,
    retrain_linear_warm,
    train_linear_A,
    train_mlp_A,
)
from .slots.hidden_deletion import (
    DeletionRecipe,
    build_slot3_arms,
    select_preserve_sources,
)


class SlotData:
    """Rows, labels and index sets shared by all slots of one run."""

    def __init__(self, rows: Sequence[dict], calibration_sources: frozenset, k_lists: Mapping, k: int):
        self.rows = list(rows)
        self.calibration_sources = frozenset(calibration_sources)
        cal = set(calibration_sources)
        self.idx_cal = [i for i, r in enumerate(rows) if r["split"] == "train" and r["source_id"] in cal]
        self.idx_fit = [i for i, r in enumerate(rows) if r["split"] == "train" and r["source_id"] not in cal]
        self.idx_dev = [i for i, r in enumerate(rows) if r["split"] == "dev"]
        if any(r["split"] not in ("train", "dev") for r in rows):
            raise PermissionError("only TRAIN and DEV rows may enter a pipeline")
        self.y = np.array([r["label"] for r in rows], dtype=float)
        self.w = np.array([r["confounded_train"] for r in rows], dtype=float)
        self.k_lists, self.k = k_lists, k
        self.fit_sources = sorted({rows[i]["source_id"] for i in self.idx_fit})
        self.cal_rows = [rows[i] for i in self.idx_cal]
        self.dev_rows = [rows[i] for i in self.idx_dev]

    def kidx(self, seed):
        ks = set(self.k_lists[seed][self.k])
        idx = [i for i in self.idx_fit if self.rows[i]["source_id"] in ks]
        if {self.rows[i]["source_id"] for i in idx} != ks:
            raise ValueError("a purchased source is missing from the fitting pool")
        return idx

    def P(self, role, s, **kw):
        s = np.asarray(s, dtype=float)
        return PipelineScores(role, s[self.idx_cal], s[self.idx_dev], **kw)

    def lin(self, X, seed):
        st = Standardizer().fit(np.asarray(X)[self.idx_fit])
        Z = st.transform(X).astype(np.float64)
        th, comp = train_linear_A(Z[self.idx_fit], self.y[self.idx_fit], seed, sample_weight=self.w[self.idx_fit])
        return Z, th, comp


def base(feats, t):
    return np.concatenate([feats[t]["pooled"], feats[t]["mean_patch"]], 1)


def slot_pipelines(slot: str, seed: int, feats: Mapping, D: SlotData, acc: AccessLedger | None = None,
                   recipe3: DeletionRecipe = DeletionRecipe(), n_draws: int = 3,
                   slot1_placebo: str = "random_area_") -> tuple[dict, dict]:
    """Return ({name: PipelineScores}, {name: compute record}) for one slot and seed."""
    rows, y = D.rows, D.y
    ki = D.kidx(seed)
    k_src = {rows[i]["source_id"] for i in ki}
    if acc is not None:
        acc.charge("adapt_pairs", k_src, arm=slot, seed=seed, k=D.k)
        acc.charge("fit_A", D.fit_sources, arm=f"{slot}/A", seed=seed)
        acc.charge("fit_scaler", D.fit_sources, arm=f"{slot}/scalers", seed=seed)
    compute = {}
    Zb, thA, cA = D.lin(base(feats, "identity"), seed)
    compute["A"] = cA.as_dict()
    pipes = {"A": D.P("A", linear_logits(Zb, thA))}
    if slot == "slot1":
        Zc, thc, cc = D.lin(base(feats, "sam_meanfill"), seed)
        pipes["slot1_sam_meanfill"] = D.P("candidate", linear_logits(Zc, thc))
        for d in range(1, n_draws + 1):
            Zp, thp, _ = D.lin(base(feats, f"{slot1_placebo}{d}"), seed)
            pipes[f"{slot1_placebo}{d}"] = D.P("placebo", linear_logits(Zp, thp))
        Zr, thr, _ = D.lin(base(feats, "padding_only"), seed)
        pipes["padding_only"] = D.P("reference", linear_logits(Zr, thr))
        st = Standardizer().fit(base(feats, "identity")[D.idx_fit])
        pipes["testtime_mask_A_head"] = D.P("reference", linear_logits(
            st.transform(base(feats, "sam_meanfill")).astype(np.float64), thA))
        Xk = base(feats, "keep_all")
        pipes["noop_keep_all_transform"] = D.P(
            "noop", linear_logits(Standardizer().fit(Xk[D.idx_fit]).transform(Xk).astype(np.float64), thA),
            noop_ref="A")
    elif slot == "slot2":
        Zc, thc, cc = D.lin(feats["identity"]["object"], seed)
        pipes["slot2_object_only"] = D.P("candidate", linear_logits(Zc, thc))
        for d in range(1, n_draws + 1):
            Zp, thp, _ = D.lin(feats["identity"][f"perm_{d}"], seed)
            pipes[f"perm_{d}"] = D.P("placebo", linear_logits(Zp, thp))
        Zu, thu, _ = D.lin(feats["identity"]["uniform"], seed)
        pipes["uniform_weights"] = D.P("reference", linear_logits(Zu, thu))
        Zq, thq, _ = D.lin(np.concatenate([feats["identity"]["pooled"], feats["identity"]["object"]], 1), seed)
        pipes["pooled_concat_diag"] = D.P("reference", linear_logits(Zq, thq))
        pipes["noop_rescore_einsum"] = D.P("noop", np.einsum("nd,d->n", Zb, thA[:-1]) + thA[-1], noop_ref="A")
    elif slot == "slot3":
        Amlp, cm = train_mlp_A(Zb[D.idx_fit], y[D.idx_fit], seed, sample_weight=D.w[D.idx_fit])
        compute["A_mlp"] = cm.as_dict()
        pres_src = set(select_preserve_sources(D.fit_sources, recipe3.preserve_n, recipe3.preserve_namespace,
                                               calibration=D.calibration_sources))
        pres = [i for i in D.idx_fit if rows[i]["variant"] == "clean" and rows[i]["source_id"] in pres_src]
        if acc is not None:
            acc.charge("preserve_set", {rows[i]["source_id"] for i in pres}, arm="slot3", seed=seed)
            acc.charge("retrain", k_src, arm="slot3/retrainers", seed=seed)
        arms = build_slot3_arms(Amlp, Zb[ki], y[ki], Zb[pres], seed, recipe3)
        pipes = {"A": D.P("A", mlp_forward(Amlp, Zb))}
        pipes["A_snapped"] = D.P("reference", mlp_forward(arms["noop_dtype_roundtrip"]["reference"], Zb))
        for nm, a in arms.items():
            if a["role"] == "noop":
                ref = "A_snapped" if nm == "noop_dtype_roundtrip" else "A"
                pipes[nm] = D.P("noop", mlp_forward(a["params"], Zb), noop_ref=ref)
            else:
                pipes[nm] = D.P(a["role"], mlp_forward(a["params"], Zb))
            if a.get("compute"):
                compute[nm] = a["compute"]
        return pipes, compute
    else:
        raise ValueError(slot)
    if acc is not None:
        acc.charge("retrain", k_src, arm=f"{slot}/retrainers", seed=seed)
    compute["candidate_fit"] = cc.as_dict()
    for nm, Z_, th0 in (("cand_rep", Zc, thc), ("orig_rep", Zb, thA)):
        thw, cw = retrain_linear_warm(Z_[ki], y[ki], th0)
        thf, cf = retrain_linear_fresh(Z_[ki], y[ki], seed)
        pipes[f"warm@k|{nm}"] = D.P("retrainer", linear_logits(Z_, thw))
        pipes[f"fresh@k|{nm}"] = D.P("retrainer", linear_logits(Z_, thf))
        compute[f"warm@k|{nm}"], compute[f"fresh@k|{nm}"] = cw.as_dict(), cf.as_dict()
    return pipes, compute
