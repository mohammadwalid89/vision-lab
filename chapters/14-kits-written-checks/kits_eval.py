"""Single-shot development evaluation of the written-check head for one category.

Fits on train-split rows only, chooses operating points on calibration normals
(never on dev), then scores the dev split once. Refuses to overwrite an existing result.
"""
import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import kits_head as H
import numpy as np
from kits_features import FeatureCache

CATS = ["juice_bottle", "splicing_connectors"]
K_ANCHORS = 32
POOLED_DIM = 16
FPR_GRID = (0.005, 0.01, 0.02)
ALPHAS = (1.0, 0.5, 0.25, 0.0)
PRIMARY = {"alpha": 0.5, "target_fpr": 0.01}
CANDIDATES = ([f"programmed_a{a}" for a in ALPHAS]
              + ["learned_only_mlp", "learned_part_only", "linear_head", "patch_knn",
                 "written_shuffled", "written_uniform"])


def log(msg):
    print(f'{time.strftime("%H:%M:%S")} {msg}', flush=True)


@dataclass
class Frozen:
    category: str
    bank: H.CheckBank
    learned: H.LearnedPart
    projector: H.PooledProjector
    heads: dict
    mlp: H.MLPHead
    linear: H.LinearHead
    knn: H.PatchKNN
    thresholds: dict
    std_shuffled: H.Standardizer = None
    std_uniform: H.Standardizer = None
    bank_shuffled: H.CheckBank = None
    bank_uniform: H.CheckBank = None


def tokens_factory(cache, idx):
    def gen():
        for i in idx:
            yield cache.get(i)["patches"]
    return gen


def fit_category(cache, cat):
    fit_idx = cache.select("train", cat, "fit_normal")
    calib_idx = cache.select("train", cat, "calib_normal")
    train_good = cache.select("train", cat, "eval_good")
    train_log = cache.select("train", cat, "eval_logical")
    log(f"{cat}: fit {len(fit_idx)} calib {len(calib_idx)} "
        f"train-good {len(train_good)} train-logical {len(train_log)}")

    sample = np.concatenate([H.unit(cache.get(i)["patches"])[::4] for i in fit_idx])
    dictionary = H.PartDictionary.fit(sample, k=K_ANCHORS)
    del sample
    bank = H.CheckBank.fit(dictionary, tokens_factory(cache, fit_idx))

    projector = H.PooledProjector(dim=POOLED_DIM).fit(
        np.stack([cache.get(i)["pooled"] for i in fit_idx]))

    labeled = ([(i, 0) for i in fit_idx] + [(i, 0) for i in train_good]
               + [(i, 1) for i in train_log])
    F = np.stack([bank.compact(cache.get(i)["patches"], cache.get(i)["pooled"], projector)
                  for i, _ in labeled])
    y = np.array([lab for _, lab in labeled])
    learned = H.LearnedPart().fit(F, y)

    calib = [(cache.get(i)["patches"], cache.get(i)["pooled"]) for i in calib_idx]
    heads = {a: H.ProgrammedHead(alpha=a).freeze(bank, learned, calib, projector)
             for a in ALPHAS}

    linear = H.LinearHead().fit(
        np.stack([np.concatenate([cache.get(i)["pooled"], cache.get(i)["layer_mean"][-1]])
                  for i, _ in labeled]), y)
    mlp = H.MLPHead().fit(F, y)
    knn = H.PatchKNN(k=512).fit([cache.get(i)["patches"] for i in fit_idx])
    frozen = Frozen(cat, bank, learned, projector, heads, mlp, linear, knn, {},
                    bank_shuffled=bank.permuted(), bank_uniform=bank.with_weights(
                        {c: 1.0 for c in H.CHECK_ORDER}))
    frozen.std_shuffled = H.Standardizer.fit(
        [frozen.bank_shuffled.written_score(cache.get(i)["patches"]) for i in calib_idx])
    frozen.std_uniform = H.Standardizer.fit(
        [frozen.bank_uniform.written_score(cache.get(i)["patches"]) for i in calib_idx])

    calib_scores = [raw_scores(frozen, cache.get(i)) for i in calib_idx]
    for name in CANDIDATES:
        v = np.sort(np.array([s[name] for s in calib_scores]))
        frozen.thresholds[name] = {f"fpr_{f}": float(np.quantile(v, 1 - f))
                                   for f in FPR_GRID}
    return frozen


def raw_scores(frozen, d):
    """Every candidate score for one image."""
    X, pooled = d["patches"], d["pooled"]
    bank = frozen.bank
    viol = bank.violations(bank.image_stats(X))
    compact = bank.compact(X, pooled, frozen.projector)
    # All alpha heads share the same calibration normals, so their standardisers
    # are identical; the 0.5 head is the reference.
    zw = frozen.heads[0.5].written_std(bank.written_score(X))
    zl = frozen.heads[0.5].learned_std(frozen.learned.decision(compact))
    out = {f"programmed_a{a}": float(a * zw + (1 - a) * zl) for a in ALPHAS}
    out["learned_only_mlp"] = frozen.mlp.score(compact)
    out["learned_part_only"] = float(zl)
    out["linear_head"] = frozen.linear.score(
        np.concatenate([pooled, d["layer_mean"][-1]]))
    out["patch_knn"] = frozen.knn.score(X)
    out["written_shuffled"] = float(frozen.std_shuffled(
        frozen.bank_shuffled.written_score(X)))
    out["written_uniform"] = float(frozen.std_uniform(
        frozen.bank_uniform.written_score(X)))
    out.update({f"check_{c}": float(viol[c]) for c in H.CHECK_ORDER})
    return out


def score_split(cache, frozen, split="dev"):
    idx = cache.select(split, frozen.category)
    rows = []
    for k, i in enumerate(idx):
        row = {"id": cache.rows[i]["id"], "role": cache.rows[i]["role"],
               "y": int(cache.rows[i]["role"] == "eval_logical")}
        row.update(raw_scores(frozen, cache.get(i)))
        rows.append(row)
        if (k + 1) % 25 == 0:
            log(f"{frozen.category} {split} {k + 1}/{len(idx)}")
    return rows


def run_meta(frozen):
    bank = frozen.bank
    return {"category": frozen.category, "grid": bank.grid,
            "menu": {k: v.tolist() for k, v in bank.menu.items()},
            "thresholds": frozen.thresholds, "check_weights": H.CHECK_WEIGHTS,
            "control_weights_uniform": {c: 1.0 for c in H.CHECK_ORDER},
            "control_shuffle_seed": H.SEED + 1,
            "alphas": list(ALPHAS), "fpr_grid": list(FPR_GRID), "primary": PRIMARY,
            "written_std": [(h.written_std.median, h.written_std.scale)
                            for h in frozen.heads.values()],
            "learned_std": [(h.learned_std.median, h.learned_std.scale)
                            for h in frozen.heads.values()]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="feature cache from kits_features.py")
    ap.add_argument("--category", required=True, choices=CATS)
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    result = out / f"DEV_RESULTS_{args.category}.json"
    if result.exists():
        raise SystemExit(f"refusing: {result} exists (dev is scored once)")
    out.mkdir(parents=True, exist_ok=True)

    cache = FeatureCache(args.features)
    frozen = fit_category(cache, args.category)
    rows = score_split(cache, frozen, "dev")
    result.write_text(json.dumps({"rows": rows, "meta": run_meta(frozen)},
                                 indent=1, default=float) + "\n")
    log(f"wrote {result.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
