"""Score finished heads once on the development split (images the agent never saw).

Run by the experimenter after the agent finishes; the agent has no access to the
development features. Each head file holds `coef` and `intercept`.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "14-kits-written-checks"))
from kits_features import FeatureCache  # noqa: E402


def auc(s, y):
    p, n = s[y == 1], s[y == 0]
    return float(((p[:, None] > n[None, :]).sum() + .5 * (p[:, None] == n[None, :]).sum())
                 / (len(p) * len(n)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--category", default="juice_bottle")
    ap.add_argument("--out", required=True, help="result JSON to write")
    ap.add_argument("heads", nargs="+", help="head .npz files")
    a = ap.parse_args()

    cache = FeatureCache(a.features)
    idx = cache.select("dev", a.category)
    X = np.stack([np.concatenate([cache.get(i)["pooled"], cache.get(i)["layer_mean"][-1]])
                  .astype(np.float64) for i in idx])
    y = np.array([int(cache.rows[i]["role"] == "eval_logical") for i in idx])
    res = {}
    for path in a.heads:
        c = np.load(path)
        res[Path(path).name] = round(auc(X @ c["coef"] + float(c["intercept"]), y), 4)
    res["n_dev"] = [int(y.sum()), int((1 - y).sum())]
    print(json.dumps(res))
    Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
