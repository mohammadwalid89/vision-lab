#!/usr/bin/env python3
"""Final-scoring implementation preflight on validation normals ONLY; never reads locked final."""

import json

from p3_final import score_components, score_patch
from p3_models import CATS, OUT, manifest


def main():
    assert (OUT / "P3_DEV_MODELS_COMPLETE").exists()
    assert not (OUT / "P3_FINAL_OPENED").exists()
    checks = []
    for cat in CATS:
        val = manifest("validation", cat)[:2]
        for model in ["component_relational", "resnet", "wrn"]:
            d = json.loads((OUT / f"p3_{model}_{cat}_dev.json").read_text())
            expected = {r["sha256"]: r for r in d["validation"]}
            scored = score_components(cat, val) if model == "component_relational" else score_patch(model, cat, val)
            for row in scored:
                exp = expected[row["sha256"]]
                for k in ["component", "relational"] if model == "component_relational" else ["score"]:
                    delta = abs(float(row[k]) - float(exp[k]))
                    assert delta < 1e-3, (cat, model, k, delta)
                    checks.append((cat, model, k, delta))
    print(
        "PASS FINAL INFERENCE PREFLIGHT",
        len(checks),
        "comparisons; maximum absolute score delta",
        max(x[3] for x in checks),
        flush=True,
    )


if __name__ == "__main__":
    main()
