"""Final evaluation on the sealed FRESH images, with the policies frozen in out/p2_frozen_policy.json.

Run once.  Refuses to run if the result file already exists (delete it deliberately to re-run, and say so).
Reports, per family and per category, for v001 as deployed and for each frozen Phase 2 system: good parts
released (with 95 % Clopper-Pearson interval), escapes (structural / logical) with the 95 % upper bound,
false holds, person rate, cost per 1,000 parts at a 2 % defect rate."""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from p2_policy import Data, apply, decisions, outcome  # noqa: E402

OUT = HERE / "out"


def main():
    dst = OUT / "p2_final_results.json"
    if dst.exists():
        sys.exit("p2_final_results.json exists: the fresh set has already been opened once")
    frozen = json.loads((OUT / "p2_frozen_policy.json").read_text())
    res = {"frozen": frozen, "systems": {}}
    for name, spec in frozen["systems"].items():
        D = Data(fresh=True, comp_key=spec.get("comp_key", "knn"), kit_calib=spec.get("kit_calib", "oof"))
        A = D.arrays(spec["family"], spec["system"])
        d = decisions(A, 0, 0) if spec["system"] == "v001" else apply(A, spec["policy"])
        r = {"overall": outcome(A, d), "per_category": {}}
        for c in sorted(set(A["c"].tolist())):
            r["per_category"][c] = outcome(A, d, np.where(A["c"] == c)[0])
        res["systems"][name] = r
        o = r["overall"]
        print(
            name,
            json.dumps(
                {
                    k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in o.items()
                    if k
                    in (
                        "n_good",
                        "n_defect",
                        "good_release_rate",
                        "good_release_ci95",
                        "escapes",
                        "escape_ucb95",
                        "escapes_by_stratum",
                        "false_hold_rate",
                        "good_to_person_rate",
                        "usd_per_1000",
                    )
                }
            ),
        )
    dst.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
