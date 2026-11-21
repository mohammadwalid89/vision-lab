"""Baseline (deployed v001) raw scores on every test / validation image of the nine products.

Uses the application's own specialists and category gate, imported read-only from system_one/ in this folder:
ResNet-18 PatchCore score, CLIP contrast score, semantic kit novelty (kits), gate distance / accept.
The semantic L2 is computed for every kit image (not only when L1 would release) so any policy can be
replayed.  Output: out/base_scores.jsonl."""

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# SYSTEM_ONE_APP: folder holding the v001 application package `system_one` (default: this folder)
sys.path.insert(0, os.environ.get("SYSTEM_ONE_APP", str(HERE)))
sys.dont_write_bytecode = True
import numpy as np  # noqa: E402
from system_one import config  # noqa: E402
from system_one import specialists as S  # noqa: E402
from system_one.reference import Reference  # noqa: E402

CATS = ["bottle", "cable", "hazelnut", "pill", "tile", "transistor", "breakfast_box", "pushpins", "screw_bag"]


def main():
    inv = json.load(open(HERE / "out/inventory.json"))
    ref = Reference.load()
    gate = ref.data["gate"]
    G = S.CategoryGate(
        {c: np.asarray(v["centroid"]) for c, v in gate.items()}, {c: v["radius"] for c, v in gate.items()}
    )
    pc, clip, sem = S.PatchCore(), S.ClipContrast(), S.SemanticKitL2()
    done = set()
    outp = HERE / "out/base_scores.jsonl"
    if outp.exists():
        done = {json.loads(line)["sha256"] for line in open(outp)}
    fout = open(outp, "a")
    for r in inv:
        if r["category"] not in CATS or r["split"] not in ("test", "validation") or r["sha256"] in done:
            continue
        c = r["category"]
        tm = {}
        rgb = S.load_rgb(r["path"])
        g = G.check(rgb, c, tm)
        patch, _ = pc.score(rgb, c, tm)
        rec = {
            "sha256": r["sha256"],
            "category": c,
            "split": r["split"],
            "type": r["type"],
            "exposed": r["exposed"],
            "gate_accepted": bool(g["accepted"]),
            "gate_distance": float(g["distance"]),
            "patch": float(patch),
        }
        if c in config.LOCO:
            cs = clip.score(rgb, c, tm)
            _, novelty = sem.score(r["path"], c, cs, tm)
            rec.update(clip=float(cs), sem=float(novelty))
        rec["ms"] = {k: round(v["wall_ms"], 2) for k, v in tm.items()}
        fout.write(json.dumps(rec) + "\n")
        fout.flush()


if __name__ == "__main__":
    main()
