"""Feasibility probe (EXPOSED images only): can zero-shot OWLv2 count screw-bag parts on CPU?

For 15 good and 15 logical-anomaly screw bags from the exposed set, detect "a screw", "a hex nut",
"a washer" (score >= 0.2, NMS IoU 0.3 within class) and record per-class counts and CPU time.  Good bags
in MVTec LOCO contain 2 screws, 2 nuts and 2 washers; logical anomalies change counts or screw length."""

import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import Owlv2ForObjectDetection, Owlv2Processor

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent


def main():
    M = os.environ["OWLV2_DIR"]  # local copy of the OWLv2 base checkpoint
    proc = Owlv2Processor.from_pretrained(M)
    model = Owlv2ForObjectDetection.from_pretrained(M).eval()
    Q = ["a photo of a screw", "a photo of a hex nut", "a photo of a washer"]
    inv = json.load(open(HERE / "out/inventory.json"))
    rows = [r for r in inv if r["category"] == "screw_bag" and r["split"] == "test" and r["exposed"] is not None]
    sel = [r for r in rows if r["type"] == "good"][:15] + [r for r in rows if r["type"] == "logical_anomalies"][:15]
    out = []
    for r in sel:
        im = Image.open(r["path"]).convert("RGB")
        t0 = time.process_time()
        w0 = time.perf_counter()
        with torch.no_grad():
            inp = proc(text=[Q], images=im, return_tensors="pt")
            o = model(**inp)
        W = max(im.size)
        res = proc.post_process_grounded_object_detection(o, threshold=0.2, target_sizes=torch.tensor([[W, W]]))[0]
        counts = []
        for q in range(3):
            m = res["labels"] == q
            b, s = res["boxes"][m], res["scores"][m]
            keep = __import__("torchvision").ops.nms(b, s, 0.3) if len(b) else []
            counts.append(int(len(keep)))
        out.append(
            {"type": r["type"], "counts": counts, "cpu_s": time.process_time() - t0, "wall_s": time.perf_counter() - w0}
        )
        print(r["type"][:4], r["path"].split("/")[-1], counts, round(out[-1]["wall_s"], 1), flush=True)
    g = [o["counts"] for o in out if o["type"] == "good"]
    lg = [o["counts"] for o in out if o["type"] != "good"]
    print(
        "good exactly (2,2,2):",
        sum(c == [2, 2, 2] for c in g),
        "/",
        len(g),
        " logical != (2,2,2):",
        sum(c != [2, 2, 2] for c in lg),
        "/",
        len(lg),
        " median wall s:",
        round(float(np.median([o["wall_s"] for o in out])), 2),
    )
    json.dump(out, open(HERE / "out/owlv2_probe.json", "w"))


if __name__ == "__main__":
    main()
