"""Phase 2 policy selection on EXPOSED images.  Fresh images are only loaded by p2_final.py.

Every system reduces to two per-image scores behind the same category gate (a refusal goes to a person):
  release if gate and s_rel > ar;  hold if gate and s_hold < ah;  otherwise a person decides.
  v001        deployed scores and policy (DecisionModel, unchanged)
  v001_tuned  deployed scores, thresholds re-selected by the pre-registered rule (threshold-only control):
              surface s_rel = s_hold = p1; kits s_rel = min(p1, p2), s_hold = p1 (L2 can only block)
  wrn         surface: WRN-50 PatchCore p (out-of-fold calibrated)
  wrn_only / comp_only / wrn_comp   kits: WRN p, component-verifier p, or both (s = min)
Kit variants: verifier score (max_z / top5_z / knn) x calibration source (oof = out-of-fold train;
confirmed = exposed LOCO validation good kits, removed from the selection population).
Selection rule (PREREGISTRATION.md): 300 stratified half-splits; on half A minimise expected cost at a 2 %
defect rate subject to zero escapes on A; reported on half B.  Global and per-category release thresholds.
Writes out/p2_dev_results.json."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
# SYSTEM_ONE_APP: folder holding the v001 application package `system_one` (default: this folder)
sys.path.insert(0, os.environ.get("SYSTEM_ONE_APP", str(HERE)))
sys.dont_write_bytecode = True
from system_one.model import INITIAL_STATE, DecisionModel  # noqa: E402
from system_one.reference import Reference  # noqa: E402

COST = {"pi": 0.02, "ch": 0.25, "cf": 2.0, "ce": 50.0}
AR = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
AH = np.array([0.0, 0.002, 0.005, 0.01, 0.02])
LOCO = ["breakfast_box", "pushpins", "screw_bag"]


def conf_p(s, ref_sorted):
    return (len(ref_sorted) - np.searchsorted(ref_sorted, s, side="left") + 1) / (len(ref_sorted) + 1)


def dedup(path):
    return {r["sha256"]: r for r in map(json.loads, open(path))}


class Data:
    """All per-image quantities for one population (exposed or fresh)."""

    def __init__(self, fresh=False, comp_key="knn", kit_calib="oof", n_confirmed=None, seed=0):
        base, wrn, comp = (
            dedup(OUT / "base_scores.jsonl"),
            dedup(OUT / "wrn_scores.jsonl"),
            dedup(OUT / "comp_scores.jsonl"),
        )
        wcal = {c: np.sort(v["oof_scores"]) for c, v in json.loads((OUT / "wrn_calibration.json").read_text()).items()}
        ccal = {c: np.sort(v[comp_key]) for c, v in json.loads((OUT / "comp_calibration.json").read_text()).items()}
        # The confirmed-goods pool (exposed LOCO validation goods) never enters any selection population,
        # whichever calibration a variant uses, so every kit variant is compared on the same images (Amendment 2).
        pool = {
            c: sorted(
                h
                for h, b in base.items()
                if b["category"] == c
                and b["split"] == "validation"
                and b["type"] == "good"
                and b["exposed"] is not None
            )
            for c in LOCO
        }
        confirmed = set().union(*pool.values())
        if kit_calib == "confirmed":
            rng = np.random.default_rng(seed)
            for c in LOCO:
                hs = pool[c]
                if n_confirmed is not None:
                    hs = list(rng.choice(hs, min(n_confirmed, len(hs)), replace=False))
                wcal[c] = np.sort([wrn[h]["score"] for h in hs])
                ccal[c] = np.sort([comp[h][comp_key] for h in hs])
        dm = DecisionModel(Reference.load(), INITIAL_STATE)
        R = []
        for h, b in base.items():
            if (b["exposed"] is None) != fresh or h not in wrn or h in confirmed:
                continue
            c = b["category"]
            fam = "loco" if c in LOCO else "ad"
            if fam == "loco" and h not in comp:
                continue
            d0 = dm.decide(c, b["patch"], b.get("sem"))
            R.append(
                {
                    "sha": h,
                    "c": c,
                    "fam": fam,
                    "y": int(b["type"] != "good"),
                    "stratum": b["type"],
                    "gate": bool(b["gate_accepted"]),
                    "v001": d0["decision"],
                    "p1": d0["p1"],
                    "p2": d0["p2"] if d0["p2"] is not None else 1.0,
                    "pw": float(conf_p(wrn[h]["score"], wcal[c])),
                    "pc": float(conf_p(comp[h][comp_key], ccal[c])) if fam == "loco" else None,
                    "comp_top": comp[h]["top"][:3] if fam == "loco" else None,
                }
            )
        self.rows = R
        self.meta = {
            "comp_key": comp_key,
            "kit_calib": kit_calib,
            "n_confirmed": n_confirmed,
            "calibration_refs_per_kit": {c: len(ccal[c]) for c in LOCO},
        }

    def arrays(self, fam, system):
        R = [r for r in self.rows if r["fam"] == fam]
        A = {
            "c": np.array([r["c"] for r in R]),
            "y": np.array([r["y"] for r in R]),
            "gate": np.array([r["gate"] for r in R]),
            "stratum": np.array([r["stratum"] for r in R]),
            "v001": np.array([r["v001"] for r in R]),
        }
        p1, p2, pw = (np.array([r[k] for r in R]) for k in ("p1", "p2", "pw"))
        pc = np.array([r["pc"] if r["pc"] is not None else 1.0 for r in R])
        if system == "v001_tuned":
            A["s_rel"] = p1 if fam == "ad" else np.minimum(p1, p2)
            A["s_hold"] = p1
        elif system in ("wrn", "wrn_only"):
            A["s_rel"] = A["s_hold"] = pw
        elif system == "comp_only":
            A["s_rel"] = A["s_hold"] = pc
        elif system == "wrn_comp":
            A["s_rel"] = A["s_hold"] = np.minimum(pw, pc)
        return A


def decisions(A, ar, ah):
    """ar: scalar or per-row array."""
    if "s_rel" not in A:  # v001 as deployed
        d = A["v001"].copy()
        d[~A["gate"]] = "human"
        return d
    d = np.full(len(A["y"]), "human", dtype=object)
    d[A["gate"] & (A["s_hold"] < ah)] = "hold"
    d[A["gate"] & (A["s_rel"] > ar)] = "release"
    return d


def outcome(A, d, idx=None):
    y, s = (A["y"], A["stratum"]) if idx is None else (A["y"][idx], A["stratum"][idx])
    d = d if idx is None else d[idx]
    g, df = (y == 0), (y == 1)
    ng, nd = int(g.sum()), int(df.sum())
    rel_g = int((g & (d == "release")).sum())
    esc = int((df & (d == "release")).sum())
    fh = int((g & (d == "hold")).sum())
    hg = int((g & (d == "human")).sum())
    ha = int((df & (d == "human")).sum())
    held = int((df & (d == "hold")).sum())
    C = COST
    er, fr, hgr, har = esc / max(1, nd), fh / max(1, ng), hg / max(1, ng), ha / max(1, nd)
    return {
        "n_good": ng,
        "n_defect": nd,
        "good_released": rel_g,
        "good_release_rate": rel_g / max(1, ng),
        "good_release_ci95": [
            float(beta.ppf(0.025, rel_g, ng - rel_g + 1)) if rel_g else 0.0,
            float(beta.ppf(0.975, rel_g + 1, ng - rel_g)) if rel_g < ng else 1.0,
        ],
        "escapes": esc,
        "escape_rate": er,
        "escape_ucb95": float(beta.ppf(0.95, esc + 1, nd - esc)) if esc < nd else 1.0,
        "escapes_by_stratum": dict(Counter(s[df & (d == "release")].tolist())),
        "false_holds": fh,
        "false_hold_rate": fr,
        "good_to_person_rate": hgr,
        "defects_held_auto_rate": held / max(1, nd),
        "defects_to_person_rate": har,
        "usd_per_1000": 1000
        * (C["pi"] * (er * C["ce"] + har * C["ch"]) + (1 - C["pi"]) * (fr * C["cf"] + hgr * C["ch"])),
        "person_reviews_per_1000": 1000 * (C["pi"] * har + (1 - C["pi"]) * hgr),
        "good_released_per_1000_parts": 1000 * (1 - C["pi"]) * rel_g / max(1, ng),
    }


def select(A, idx, per_category):
    """Pre-registered rule on rows idx: minimise cost s.t. zero escapes; returns (ar per row-category map, ah)."""
    cats = np.unique(A["c"][idx])
    best = None
    for ah in AH:
        if per_category:
            arm = {}
            for c in cats:
                ic = idx[A["c"][idx] == c]
                cand = []
                for ar in AR:
                    d = decisions(A, ar, ah)
                    o = outcome(A, d, ic)
                    if o["escapes"] == 0:
                        cand.append((o["usd_per_1000"], -ar, ar))
                arm[str(c)] = float(min(cand)[2]) if cand else 1.0
            arr = np.array([arm.get(str(c), 1.0) for c in A["c"]])
            o = outcome(A, decisions(A, arr, ah), idx)
            key = (o["escapes"] > 0, o["usd_per_1000"])
            pol = {"ar": arm, "ah": float(ah)}
            if best is None or key < best[0]:
                best = (key, pol)
        else:
            for ar in AR:
                o = outcome(A, decisions(A, ar, ah), idx)
                key = (o["escapes"] > 0, o["usd_per_1000"])
                if best is None or key < best[0]:
                    best = (key, {"ar": float(ar), "ah": float(ah)})
    return best[1]


def apply(A, pol):
    ar = np.array([pol["ar"].get(str(c), 1.0) for c in A["c"]]) if isinstance(pol["ar"], dict) else pol["ar"]
    return decisions(A, ar, pol["ah"])


def splits(A, n, seed=7):
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, (c, y) in enumerate(zip(A["c"], A["y"])):
        groups[(c, y)].append(i)
    for _ in range(n):
        a = []
        for v in groups.values():
            v = list(v)
            rng.shuffle(v)
            a += v[: len(v) // 2]
        a = np.array(sorted(a))
        b = np.setdiff1d(np.arange(len(A["y"])), a)
        yield a, b


def evaluate(A, system, per_cat, n_splits):
    allidx = np.arange(len(A["y"]))
    if system == "v001":
        return {"in_sample_dev": outcome(A, decisions(A, 0, 0))}
    oos, picks = [], Counter()
    for a, b in splits(A, n_splits):
        pol = select(A, a, per_cat)
        picks[json.dumps(pol, sort_keys=True)] += 1
        oos.append(outcome(A, apply(A, pol), b))
    pol = select(A, allidx, per_cat)
    agg = {
        k: float(np.mean([o[k] for o in oos]))
        for k in (
            "good_release_rate",
            "escape_rate",
            "false_hold_rate",
            "good_to_person_rate",
            "usd_per_1000",
            "person_reviews_per_1000",
            "good_released_per_1000_parts",
        )
    }
    agg["p_any_escape"] = float(np.mean([o["escapes"] > 0 for o in oos]))
    agg["good_release_rate_q05_q95"] = [
        float(np.quantile([o["good_release_rate"] for o in oos], q)) for q in (0.05, 0.95)
    ]
    return {
        "oos_half_splits": agg,
        "policy_on_full_dev": pol,
        "in_sample_dev": outcome(A, apply(A, pol)),
        "n_splits": n_splits,
        "most_selected_on_half_A": json.loads(picks.most_common(1)[0][0]),
    }


def freeze(res):
    """Amendment 2: per family, the variant with the lowest mean out-of-sample cost among those whose mean
    out-of-sample escape rate is <= 1 %; its deployed point is the policy most often selected on half A, unless
    that point releases a defect on the full selection set, in which case the full-set selection is used."""
    frozen = {"rule": freeze.__doc__.strip(), "systems": {}}
    for fam in ("ad", "loco"):
        V = {k: v for k, v in res["variants"].items() if k.startswith(fam + "|") and "oos_half_splits" in v}
        ok = {k: v for k, v in V.items() if v["oos_half_splits"]["escape_rate"] <= 0.01}
        best = min(ok, key=lambda k: ok[k]["oos_half_splits"]["usd_per_1000"])
        _, system, calib, comp, scope = best.split("|")
        A = Data(
            fresh=False, comp_key=comp.split("=")[1] if comp != "comp=-" else "knn", kit_calib=calib.split("=")[1]
        ).arrays(fam, system)
        pol = V[best]["most_selected_on_half_A"]
        source = "most_selected_on_half_A"
        if outcome(A, apply(A, pol))["escapes"] > 0:
            pol = V[best]["policy_on_full_dev"]
            source = "policy_on_full_dev"
        spec = {
            "family": fam,
            "system": system,
            "policy": pol,
            "policy_source": source,
            "variant": best,
            "comp_key": comp.split("=")[1] if comp != "comp=-" else "knn",
            "kit_calib": calib.split("=")[1],
        }
        frozen["systems"][f"{fam}|phase2"] = spec
        frozen["systems"][f"{fam}|v001"] = {"family": fam, "system": "v001", "variant": f"{fam}|v001 as deployed"}
        tuned = f"{fam}|v001_tuned|calib=oof|comp=-|global"
        frozen["systems"][f"{fam}|v001_tuned"] = {
            "family": fam,
            "system": "v001_tuned",
            "policy": V[tuned]["most_selected_on_half_A"],
            "variant": tuned,
            "kit_calib": "oof",
            "comp_key": "knn",
        }
    return frozen


def learning_curve(res, n_splits=100, seeds=5):
    """Kits: out-of-sample release vs the number of confirmed good kits per kit type (exposed only)."""
    curve = {}
    for system, key in (("comp_only", "knn"), ("wrn_comp", "knn"), ("wrn_comp", "max_z"), ("wrn_only", "max_z")):
        for n in (10, 25, 50, None):
            rows = []
            for sd in range(seeds if n is not None else 1):
                A = Data(fresh=False, comp_key=key, kit_calib="confirmed", n_confirmed=n, seed=sd).arrays(
                    "loco", system
                )
                rows.append(evaluate(A, system, False, n_splits)["oos_half_splits"])
            curve[f"{system}|comp={key}|n={n or 'all'}"] = {
                k: float(np.mean([r[k] for r in rows]))
                for k in ("good_release_rate", "escape_rate", "usd_per_1000", "p_any_escape")
            }
            print(
                "curve",
                system,
                key,
                n,
                json.dumps({k: round(v, 4) for k, v in curve[f"{system}|comp={key}|n={n or 'all'}"].items()}),
                flush=True,
            )
    return curve


def main(n_splits=300):
    res = {"cost": COST, "variants": {}}
    plan = [("ad", "oof", "knn", ["v001", "v001_tuned", "wrn"])]
    for calib in ("oof", "confirmed"):
        for key in ("max_z", "top5_z", "knn"):
            systems = (
                ["comp_only", "wrn_comp"]
                + (["wrn_only"] if key == "max_z" else [])
                + (["v001", "v001_tuned"] if (calib, key) == ("oof", "max_z") else [])
            )
            plan.append(("loco", calib, key, systems))
    for fam, calib, key, systems in plan:
        D = Data(fresh=False, comp_key=key, kit_calib=calib)
        for system in systems:
            A = D.arrays(fam, system)
            for per_cat in [False] if system == "v001" else [False, True]:
                name = f"{fam}|{system}|calib={calib}|comp={key if 'comp' in system else '-'}|{'per_category' if per_cat else 'global'}"
                r = evaluate(A, system, per_cat, n_splits)
                r["population"] = {"good": int((A["y"] == 0).sum()), "defect": int((A["y"] == 1).sum())}
                res["variants"][name] = r
                o = r.get("oos_half_splits") or r["in_sample_dev"]
                print(
                    name,
                    json.dumps(
                        {
                            k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in o.items()
                            if k
                            in (
                                "good_release_rate",
                                "p_any_escape",
                                "escapes",
                                "escape_rate",
                                "false_hold_rate",
                                "usd_per_1000",
                                "good_release_rate_q05_q95",
                            )
                        }
                    ),
                    r["population"],
                    flush=True,
                )
    (OUT / "p2_dev_results.json").write_text(json.dumps(res, indent=1))
    frozen = freeze(res)
    (OUT / "p2_frozen_policy.json").write_text(json.dumps(frozen, indent=1))
    print("FROZEN", json.dumps(frozen), flush=True)
    res["confirmed_goods_curve"] = learning_curve(res)
    (OUT / "p2_dev_results.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
