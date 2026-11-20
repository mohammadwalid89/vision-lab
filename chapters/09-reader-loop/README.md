# 09. Reader loop (Mirage)

**Question.** Can a reader model plus an adapter, revised once from measured errors, beat a sham adapter?

**Setup.** Mirage 4.0.1 (Apache-2.0), a 22 MB 2-bit local perception model that returns a structured record per image (object names, boxes in 0-1000 coordinates, scores, relations, text). Planned arms on the same frozen images: the Mirage record alone, a text-only language-model reader over that record, a learned adapter over the record trained on 12 images and applied to 24, a sham adapter trained on permuted labels, and one revision of the adapter chosen from measured errors. Only the preflight below was run.
The preflight used three VisDrone training images and the six detector classes (person, bicycle, car, truck, bus, motorcycle), with a frozen table that maps Mirage's object names to those classes.

**Result.** No arm ran, so there is no comparison. Preflight, as box detection at IoU 0.5 and score 0.25 (`results/summary.json`, from `score_preflight.py`; the same totals are in the run's status record):

| Images | Ground-truth objects | Mirage target boxes | True positives | False positives | Misses |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 476 | 18 | 1 | 17 | 475 |

One CPU call took about 160 to 216 seconds per image (run status record). The misses are objects Mirage never emitted at aerial scale, not output-format failures, so a reader of its record or an adapter over it cannot recover them. The task was therefore changed before any arm ran, and the change was recorded in the pre-registration, to image-level presence of each of the six classes on the same VisDrone images: 36 frozen images (24 dev, 12 train, scene groups disjoint, the three probed groups excluded), 92 pool images unused.

**What it means.** This is a feasibility stop, not a result about readers or adapters. Because Mirage's weights are 2-bit and the adapter would sit on its output record, the study could only have tested a learned post-processor, not an edit of Mirage's weights. The preflight covers three images, so it shows the failure mode, not its rate. The study was paused by the owner for a higher-priority experiment before any arm ran and has not been resumed.

**How to run.** Requires the `cactus` package (Mirage) and VisDrone2019-DET train images (128-image pool, folder with `images/` and `annotations/`; see the dataset notice in the repository README).
```
python make_manifest.py --pool-dir <train> --out results/manifest.json
python preflight_probe.py <image ids> --pool-dir <train> --out results/preflight_probe.json
python score_preflight.py --probe results/preflight_probe.json --pool-dir <train>
python run_mirage.py --manifest results/manifest.json --pool-dir <train>
```
`run_mirage.py` caches Mirage records for the frozen ids; no arm consumed them, and it is not needed for the preflight.

**Status.** Paused, preflight only. Code is cleaned from the run scripts; the shared constants, split rule and name map are unchanged. `score_preflight.py` is not a copy of an original script: the original run recorded the preflight totals but the matching step was not saved, so it was rewritten here from the saved helpers (frozen name map, coordinate conversion) with a greedy highest-score-first matcher. On the stored probe output of the three images it gives 476, 18, 1, 17, 475, the same totals as the run record. Verified here: `py_compile` and `ruff` pass; `make_manifest.py` was run on the real 128-image pool (24 dev, 12 train, 92 unused); `run_mirage.py` and `preflight_probe.py` were run only against a stub `cactus` module that returns a fixed record, which checks control flow and file handling, not Mirage. The adapter, reader and sham code do not exist because no arm was built.
