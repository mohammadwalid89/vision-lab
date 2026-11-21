"""Versioned decision model: calibration references + LOCO evidence-fusion weights + policy.

A model *state* is a small JSON document:
  {"version": "v003", "parent": "v002", "weights": {"loco": [w_patch, w_sem]},
   "extra_normals": {category: [[patch, sem_or_null], ...]}, "policy": {...}, "created": ..., "note": ...}

Decision flow (per category, conformal p-values against that category's normal references):
  AD   : p1 = p(PatchCore)          release if p1 > a_r; hold if p1 < a_h; else HUMAN
  LOCO : p1 < a_h  -> HOLD on L1 alone (L2 skipped)
         p1 > a_r  -> L2 is mandatory: p2 = p(w . slog[z_patch, z_sem]); release if p2 > a_r, else HUMAN
         otherwise -> HUMAN (L2 not needed: a person decides)
The initial LOCO weights [0, 1] make p2 the semantic-L2 p-value exactly as in the research.
"""

from __future__ import annotations

import copy

import numpy as np

from . import config
from .reference import Reference, conformal_p, slog

INITIAL_STATE = {
    "version": "v001",
    "parent": None,
    "weights": {"loco": [0.0, 1.0]},
    "extra_normals": {},
    "policy": copy.deepcopy(config.POLICY),
    "created": None,
    "note": "Initial production model: research calibration references, semantic-only L2 score.",
}


class DecisionModel:
    def __init__(self, ref: Reference, state: dict):
        self.ref, self.state = ref, state
        self._cache = {}

    def _p1_ref(self, c):
        key = ("p1", c)
        if key not in self._cache:
            extra = [x[0] for x in self.state["extra_normals"].get(c, [])]
            self._cache[key] = np.sort(np.concatenate([self.ref.cats[c]["cal_patch"], extra]))
        return self._cache[key]

    def stage2(self, z_patch, z_sem):
        w = np.asarray(self.state["weights"]["loco"], float)
        return float(w @ slog([z_patch, z_sem]))

    def _p2_ref(self, c):
        key = ("p2", c)
        if key not in self._cache:
            zp, zs = self.ref.calibration_pairs(c, self.state["extra_normals"].get(c, []))
            w = np.asarray(self.state["weights"]["loco"], float)
            self._cache[key] = np.sort(w[0] * slog(zp) + w[1] * slog(zs))
        return self._cache[key]

    def p1(self, c, patch):
        return conformal_p(patch, self._p1_ref(c))

    def p2(self, c, patch, sem):
        return conformal_p(self.stage2(self.ref.z_patch(c, patch), self.ref.z_sem(c, sem)), self._p2_ref(c))

    def needs_l2(self, c, p1):
        pol = self.state["policy"][config.family(c)]
        return pol["mode"] == "gate" and p1 > pol["alpha_release"]

    def decide(self, c, patch, sem=None):
        fam = config.family(c)
        pol = self.state["policy"][fam]
        ar, ah = pol["alpha_release"], pol["alpha_hold"]
        p1 = self.p1(c, patch)
        out = {
            "category": c,
            "family": fam,
            "p1": p1,
            "p2": None,
            "alpha_release": ar,
            "alpha_hold": ah,
            "l2_run": False,
            "model_version": self.state["version"],
        }
        if fam == "ad" or pol["mode"] == "l1":
            if p1 > ar:
                d, why = (
                    "release",
                    f"PatchCore p = {p1:.3f} is above the release threshold {ar:.2f}: this part looks like the category's normal references.",
                )
            elif p1 < ah:
                d, why = (
                    "hold",
                    f"PatchCore p = {p1:.3f} is below the hold threshold {ah:.3f}: fewer than {100 * p1:.1f}% of normal references look this anomalous.",
                )
            else:
                d, why = (
                    "human",
                    f"PatchCore p = {p1:.3f} is between hold ({ah:.3f}) and release ({ar:.2f}) thresholds: not confident either way, so a person decides.",
                )
            out.update(decision=d, reason=why, path="L1 PatchCore")
            return out
        if p1 < ah:
            out.update(
                decision="hold",
                path="L1 PatchCore (L2 skipped)",
                reason=f"PatchCore p = {p1:.3f} < {ah:.3f}: a local defect is already evident, so the kit is held without running the semantic L2.",
            )
            return out
        if p1 <= ar:
            out.update(
                decision="human",
                path="L1 PatchCore",
                reason=f"PatchCore p = {p1:.3f} is ambiguous ({ah:.3f}–{ar:.2f}). Only release requires L2, so the case goes to a person.",
            )
            return out
        out["l2_run"] = True
        if sem is None:
            out.update(
                decision="human",
                path="L1 → L2 unavailable",
                reason="L1 would release, but the semantic L2 result is missing; a release without L2 is never allowed.",
            )
            return out
        p2 = self.p2(c, patch, sem)
        out["p2"] = p2
        if p2 > ar:
            out.update(
                decision="release",
                path="L1 → L2 semantic",
                reason=f"PatchCore p = {p1:.3f} and the L2 kit rank r = {p2:.3f} are both above {ar:.2f}: no local or arrangement anomaly found.",
            )
        else:
            out.update(
                decision="human",
                path="L1 → L2 semantic",
                reason=f"PatchCore would release (p = {p1:.3f}) but the L2 kit rank r = {p2:.3f} ≤ {ar:.2f} flags the kit's content or arrangement; L2 can block a release but cannot hold on its own, so a person decides. (r ranks the kit against the verifier's own training kits; it is not a calibrated probability.)",
            )
        return out
