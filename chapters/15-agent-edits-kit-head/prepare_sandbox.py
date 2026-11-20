"""Build the agent's sandbox folder for one category.

The sandbox holds everything the agent can see: edit-fit and edit-val features, the
starting head, the scoring tool (score_tool.py, 50-call budget) and the brief
(brief_template.md). Development and holdout images are never copied into it.

Features per image: pooled SigLIP2 embedding (768) + last-layer mean patch token (768).
Uses the feature cache and LinearHead from chapter 14.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "14-kits-written-checks"))
from kits_features import FeatureCache  # noqa: E402
from kits_head import LinearHead  # noqa: E402

SPLIT_SEED = 20261119
FIT_FRACTION = 0.6


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True, help="feature cache from chapter 14")
    ap.add_argument("--category", default="juice_bottle")
    ap.add_argument("--out", required=True, help="parent directory for sandbox_<category>")
    a = ap.parse_args()

    cache = FeatureCache(a.features)
    cat = a.category
    sb = Path(a.out) / f"sandbox_{cat}"
    sb.mkdir(parents=True, exist_ok=True)

    def feat(i):
        d = cache.get(i)
        return np.concatenate([d["pooled"], d["layer_mean"][-1]]).astype(np.float32)

    fit = cache.select("train", cat, "fit_normal")
    calib = cache.select("train", cat, "calib_normal")
    good = cache.select("train", cat, "eval_good")
    logi = cache.select("train", cat, "eval_logical")
    rng = np.random.default_rng(SPLIT_SEED)
    good, logi = list(rng.permutation(good)), list(rng.permutation(logi))
    ng, nl = int(round(FIT_FRACTION * len(good))), int(round(FIT_FRACTION * len(logi)))
    fit_rows = [(i, 0) for i in fit] + [(i, 0) for i in good[:ng]] + [(i, 1) for i in logi[:nl]]
    val_rows = [(i, 0) for i in calib] + [(i, 0) for i in good[ng:]] + [(i, 1) for i in logi[nl:]]

    def pack(rows):
        return (np.stack([feat(i) for i, _ in rows]), np.array([y for _, y in rows]),
                np.array([cache.rows[i]["id"] for i, _ in rows]),
                np.array([cache.rows[i]["role"] for i, _ in rows]))

    Xf, yf, idf, rf = pack(fit_rows)
    Xv, yv, idv, rv = pack(val_rows)
    head = LinearHead().fit(Xf, yf)
    coef = head.clf.coef_.ravel().astype(np.float64)
    intercept = np.float64(head.clf.intercept_[0])
    np.savez_compressed(sb / "edit_fit.npz", X=Xf, y=yf, ids=idf, roles=rf)
    np.savez_compressed(sb / "edit_val.npz", X=Xv, y=yv, ids=idv, roles=rv)
    np.savez_compressed(sb / "start_head.npz", coef=coef, intercept=intercept)
    np.savez_compressed(sb / "candidate.npz", coef=coef, intercept=intercept)

    shutil.copy(HERE / "score_tool.py", sb / "score.py")
    brief = (HERE / "brief_template.md").read_text().replace("{CATEGORY}", cat)
    (sb / "BRIEF.md").write_text(brief)
    print(json.dumps({"fit": [int(len(yf)), int(yf.sum())], "val": [int(len(yv)), int(yv.sum())]}))


if __name__ == "__main__":
    main()
