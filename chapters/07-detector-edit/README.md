# 07. Programming weights: a detector edit

**Question.** Can a sparse edit to a detector's output weights repair its low recall?

**Setup.** YOLOv10s (Ultralytics, COCO weights), six target classes (person, bicycle, car, truck, bus, motorcycle), 32 VisDrone validation images with 1,878 target objects, already exposed in earlier work. Matching is greedy, class-matched, IoU 0.5, confidence 0.25. Baseline: 333 true positives, 128 false positives, 1,545 misses.
Three experiments on the same images:

1. `class_gain_edit.py`: multiply the output rows of the three most under-recalled classes (car, motorcycle, person) by exp(lambda), with lambda set from the median confidence of their near-miss boxes. A random-direction placebo is run alongside.
2. `directional_edit.py`: a second edit written as a bounded step along the fit-image anchor direction (class anchor mean minus background mean), gated on a preflight, with a same-support placebo, a logistic-regression fit written into the same rows, and a threshold-only control. Every arm is also scored at an equal alert budget.
3. `head_finetune.py`: ordinary head-only fine-tuning (128 fit images, backbone frozen), as a check that the cohort has headroom at all.

**Result.** From `results/summary.json`.

Class-gain edit (1,152 of 8,170,675 weights, 0.0141%):

| Arm | Precision | Recall | True positives | False positives |
| --- | ---: | ---: | ---: | ---: |
| Original | 0.7223 | 0.1773 | 333 | 128 |
| Edited | 0.1262 | 0.4526 | 850 | 5,885 |
| Placebo | 0.1078 | 0.4260 | 800 | 6,619 |

Recall rose by 0.275, with 5,757 new false positives. In the run report's error map (not published) 3,609 of the 5,885 false positives were pure background and 1,421 were duplicates, and the missed objects had no same-class candidate at IoU 0.1 to 0.5, so the edit pushed existing weak responses over the threshold instead of creating localisations.

Directional edit (2,304 weights, 0.0283%; preflight median object-vs-background specificity AUC 0.875, gate 0.55). At fixed confidence 0.25, dose 0.10 gave 576 true positives and 704 false positives; a threshold change alone on the original scores gave 563 and 718. At an equal budget of 461 alerts (the original's alert count):

| Arm | True positives | False positives |
| --- | ---: | ---: |
| Original | 333 | 129 |
| Directional, dose 0.05 | 336 | 126 |
| Directional, dose 0.10 | 341 | 121 |
| Directional, dose 0.20 | 345 | 117 |
| Same-support placebo, dose 0.10 | 338 | 124 |
| Logistic fit in the same rows | 147 | 315 |

At dose 0.10, 29 of 32 images got worse and 3 better.

Head-only fine-tuning at the same alert budget: 338 true positives (1 epoch, lr 1e-3) and 339 (3 epochs, lr 3e-4), against 333 for the original and a pre-registered bar of +10; the backbone was unchanged.

**What it means.** A recall gain bought with thousands of false positives is a regression, and for the directional edit a threshold change alone reproduced almost all of the fixed-confidence effect. Once alert counts are equalised, the best edit finds 8 more objects than the original (341 against 333), the placebo finds 5 more, and ordinary head training 5 to 6 more, so this cohort has little headroom at the head. This is a result for these edits on 32 exposed images, not a general claim about editing detectors; a real test would need backbone training and a fresh cohort.

**How to run.** Requires `ultralytics` and a local YOLOv10s checkpoint (`YOLOV10S_CKPT` or `--checkpoint`), and VisDrone2019-DET train and val folders (each with `images/` and `annotations/`; see the dataset notice in the repository README). From the repository root with `PYTHONPATH=src`:
```
python chapters/07-detector-edit/class_gain_edit.py --eval-dir <val> --eval-ids <ids.txt> --work-dir work
python chapters/07-detector-edit/directional_edit.py --fit-dir <train> --eval-dir <val> --eval-ids <ids.txt> --old-edit-ckpt work/yolov10s_sparse_class_repair.pt
python chapters/07-detector-edit/head_finetune.py --fit-dir <train> --eval-dir <val> --eval-ids <ids.txt> --epochs 1 --lr0 0.001
python chapters/07-detector-edit/head_finetune.py --fit-dir <train> --eval-dir <val> --eval-ids <ids.txt> --epochs 3 --lr0 0.0003
```
The evaluation ids are the 32 frozen images (not published; any list of ids works but the numbers will differ). The run used a 128-image fit folder of VisDrone2019-DET train images; the same folder layout is expected by `--fit-dir`.

**Status.** Negative, controlled. Code is cleaned from the run scripts; the checkpoint-manifest and budget framework of the original suite was replaced by plain arguments, so these scripts read images and annotations directly. The two head-only runs share one script, which applies the class-name restoration that the original run needed as a separate corrective pass. Known caveat in the class-gain run: its placebo adds random noise to the already edited weights (the code reads the edited state as its base), so the placebo row above is the edit plus noise, not an independent random edit; the directional run's placebo is built from the original weights. This was found by reading the code, not by re-running it.
Verified here: `py_compile` and `ruff` pass. Smoke runs on synthetic noise images with the locally cached YOLOv10s weights: `directional_edit.py` and `head_finetune.py` ran end to end (all metrics are zero on noise, so only control flow is checked); `class_gain_edit.py` stops at its no-target guard on noise, so its edit, save, reload and placebo path was exercised with a forced target class (source checkpoint unchanged, reload exact, 3 tensors edited). No numeric result above was recomputed here: they come from the original run records (`results/summary.json` is extracted from them), because the 32 frozen images and the VisDrone data are not available in this repository.
