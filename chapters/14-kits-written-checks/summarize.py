"""Per-candidate AUC from a DEV_RESULTS_<category>.json written by kits_eval.py."""
import argparse
import json
from collections import Counter
from pathlib import Path

from sklearn.metrics import roc_auc_score


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results")
    ap.add_argument("--out", help="write the summary JSON here")
    a = ap.parse_args()

    d = json.loads(Path(a.results).read_text())
    rows = d["rows"]
    y = [r["y"] for r in rows]
    names = [k for k in rows[0] if k not in ("id", "role", "y")]
    summary = {
        "category": d["meta"]["category"],
        "n_images": len(rows),
        "roles": {role: n for (role, _), n in Counter((r["role"], r["y"]) for r in rows).items()},
        "primary": d["meta"]["primary"],
        "auc": {k: round(roc_auc_score(y, [r[k] for r in rows]), 4) for k in names},
    }
    text = json.dumps(summary, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
