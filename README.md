# Vision lab

**Question.** When can a vision model act alone, and when should a person check? The system under study escalates only on doubt: a cheap specialist decides when it is sure, a second opinion runs before any release, and a person has the final say on anything unsure. Every experiment here runs on public benchmarks, on CPU.

**Finding so far.** On a sealed public test (MVTec AD), a calibrated per-category gate over a frozen WRN-50 PatchCore score released 84 of 100 good surface parts without a person, up from 4, and released 0 of 200 defects (a 95% upper bound of 1.49% on the escape rate, not zero). Kits, where faults are logical (a missing or wrong part), did not qualify: none of ten candidates met the rule written in advance, so every kit goes to a person. Writing weights directly, by formula or by an agent, works but mostly trails ordinary training; the exception is an agent that keeps notes on its own failures, which matched a tuned trained head on 11 public image tasks. Removing a planted shortcut from a model's weights has not yet passed every gate.

## Chapters

Each chapter is one experiment from the Lab page, in order. Stages group them.

| # | Stage | Question | Finding | Status |
|---|---|---|---|---|
| [01](chapters/01-cascade-feasibility) | 1 Surface parts | Can a low-cost cascade cover surface defects, kits, damage, identity and text, and abstain when evidence is weak? | 30–36 wrong auto-accepts per seed on surface parts, so release was rejected | done |
| [02](chapters/02-two-layer-audit) | 1 Surface parts | Can per-category calibration and a selective second layer buy safe autonomy cheaply? | The second layer caught 63 kit defects the first missed; the semantic layer matched the full one's AUC (.732 vs .733) at .751 CPU seconds against 7.2 | done |
| [03](chapters/03-surface-sealed-test) | 1 Surface parts | Does a frozen WRN-50 score with a per-category rule release more good parts without releasing defects? | 4 → 84 of 100 good released; 0 of 200 defects | done |
| [04](chapters/04-kits-logical-faults) | 2 Kits | Does the surface approach carry over to kits? | 0 of 10 candidates met the 1% rule; every kit goes to a person | done |
| [05](chapters/05-embedding-pilot) | 2 Kits | Can a lightweight multimodal embedding help on hard kits? | AUC .48 on 32 exposed images: too small to be a negative | done |
| [06](chapters/06-programmed-anomaly-head) | 3 Writing weights | Can closed-form formulas write useful weights without gradient descent? | AUC .816 against .516 for a placebo; below a trained head (.858) | done |
| [07](chapters/07-detector-edit) | 3 Writing weights | Can a sparse output edit repair a detector's low recall? | 341 objects against 338 for a same-size placebo at an equal alert budget | negative |
| [08](chapters/08-video-heads) | 3 Writing weights | Can video heads be written directly, and can a kernel edit improve them? | Programmed head +2.0 points; low-pass edit +3.0, short of the +5 bar | exploratory |
| [09](chapters/09-reader-loop) | 3 Writing weights | Can a reader model plus an adapter beat a sham adapter? | Paused after preflight (160–216 s per image); no arm ran | paused |
| [10](chapters/10-vlm-frozen-protocol) | 2 Kits | Do small vision-language models cut unsure referrals without releasing more defects? | 1,000 test images frozen; too slow on one CPU core, so 0 predictions | planned; code not in this repo |
| [11](chapters/11-shortcut-rounds-1-2) | 4 Shortcut removal | Can a visual shortcut be removed from a model's weights without raising false alarms? | 0 of 21 registered attempts passed all six gates | negative |
| [12](chapters/12-deletion-threshold-probes) | 4 Shortcut removal | Why did no arm pass: the edit, or the threshold? | Deleting 5% of one layer removed 64.3% of the shortcut; 5 of 6 gates at a development-fitted threshold | exploratory |
| [13](chapters/13-round3-mask-the-object) | 4 Shortcut removal | Does removing the background from the input remove the shortcut? | Closest: object masking on DINOv2, 7 of 8 gates on every seed; did not beat retraining | negative |
| [14](chapters/14-kits-written-checks) | 5 Written checks and agents | Can a kit's specification, written as part checks, decide what anomaly scores could not? | Juice bottles: AUC .82 against .89 for a trained head and .45 for a placebo | in progress |
| [15](chapters/15-agent-edits-kit-head) | 5 Written checks and agents | Can a frontier model with weight access fix the kit head? | Unseen kits: agent .791, random edits .849, start .841 | negative |
| [16](chapters/16-agent-writes-weights) | 5 Written checks and agents | Can an agent write a model's weights by hand and improve from its own failures? | +8.4 points over start, +0.6 over a tuned trained head (9 of 11 tasks) | done |

Chapter numbers follow the order the experiments were added to the Lab page; chapter 10 belongs to stage 2. Not here: the client calibration that started the question (private), and two designed but unrun studies (the improve loop and mechanistic grounding).

## Results

| Measure | Value | Chapter |
|---|---|---|
| Good surface parts released without a person (sealed, of 100) | 4 → 84 | 03 |
| Defects released (sealed, of 200) | 0; 95% upper bound 1.49% | 03 |
| Modeled person reviews per 1,000 parts (2% defect rate assumed) | 956.8 → 162.9 | 03 |
| Kit candidates meeting the 1% rule set in advance | 0 of 10 | 04 |
| Programmed anomaly head AUC (placebo / trained) | .816 (.516 / .858) | 06 |
| Detector edit vs placebo, objects found at 461 alerts | 341 vs 338 | 07 |
| Shortcut removal, registered attempts passing all gates | 0 of 21 (Rounds 1–2); 0 of 18 (Round 3) | 11, 13 |
| Kit written-check head AUC (trained / placebo) | .82 (.89 / .45) | 14 |
| Agent-edited kit head AUC on unseen kits (random edits / start) | .791 (.849 / .841) | 15 |
| Agent-written heads vs tuned trained head, 11 tasks | +0.60 points, ahead on 9 | 16 |

Each chapter README names the source of every number it reports, usually a file in its `results/` folder. Chapter 10's figures come from its protocol status files, which are not published here (its code was not in the restored archive; see CHANGELOG). Chapter 03's paired test and scoring times and chapter 04's OWLv2 timing come from the Phase 2 run report.

## Layout

```
src/vision_lab/cascade/   PatchCore scoring, metrics and hashing for the cascade chapters
src/vision_lab/weights/   accounting and detection metrics for the weight-writing chapters
src/vision_lab/wsp/       the shortcut-removal harness: encoders, edits, gates, calibration, ledger
chapters/NN-slug/         one experiment each: README, entry scripts (chapters 03-05: the run's code in code/), results/
scripts/fetch_data.py     where to get each dataset, and its licence
```

## How to run

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install -e .                       # or: export PYTHONPATH=src
python scripts/fetch_data.py           # lists datasets, licences and target folders
```

Each chapter README gives its own commands. Data and model locations come from arguments or environment variables (for example `LOCO_ROOT`, `VISION_COHORT_DIR`, `KTH_CLIPS_DIR`, `SIGLIP2_DIR`); nothing points at a fixed path. Most chapters include a smoke run on synthetic input that needs no dataset. Chapters 15 and 16 start a coding agent through the command in `WSP_AGENT_CMD`; credentials come from your environment and none are stored here.

`python scripts/check_results.py` re-checks the sha256 of every published result file, pre-registration and plan file against `results.sha256`. CI (`.github/workflows/ci.yml`) runs ruff, that check, and the tests that need no data: `tests/`, and the tests of chapters 11, 12, 13 and 16.

## Licence and dataset credits

Code: MIT, Copyright (c) 2026 Mohammad Walid (see `LICENSE`).

The MIT licence covers the code. Files under `chapters/*/results/` are measurements on the datasets below and stay subject to those datasets' terms (MVTec AD and LOCO: non-commercial).

No dataset, image, feature file or model weight is in this repository.

The licences below were checked against each official page on 2026-11-21. `LICENSES/DATA_LICENSES.md` gives the page, what it says and what was recorded before the check. Where a page states no licence, the table says so.

| Dataset | Licence | Note |
|---|---|---|
| MVTec AD (Bergmann et al., CVPR 2019, IJCV 2021) | CC BY-NC-SA 4.0 | Non-commercial; never redistributed. Download from MVTec. |
| MVTec LOCO AD (Bergmann et al., IJCV 2022) | CC BY-NC-SA 4.0 | Non-commercial; never redistributed. Chapters 05 and 14 ship per-image scores keyed by hashed ids only (chapter 05: the SHA256 of each image file); no image is included. The Hugging Face mirror used for two chapter 04 categories was not checked separately. |
| VisA (Zou et al., ECCV 2022) | CC BY 4.0 | |
| Total-Text (Ch'ng, Chan, Liu) | BSD-3-Clause | |
| VisDrone2019-DET | No licence stated in the official repository | Research use only, per the challenge pages. |
| KTH Actions | Non-commercial use; citation requested | Clips must not be redistributed. |
| CIFAR-10, GTSRB | No licence stated; citation requested | |
| EuroSAT | MIT (dataset release) | Sentinel-2 imagery is Copernicus open data. |
| Oxford-IIIT Pets | CC BY-SA 4.0 | Copyright of the images stays with their owners. |
| Oxford Flowers 102 (Nilsback and Zisserman, 2008) | No licence stated | Used in chapter 16 v1 only, excluded from the counted tasks. |
| Describable Textures Dataset (Cimpoi et al., CVPR 2014) | Research use only | Features extracted in chapter 16; no task built. |

Models used as frozen feature extractors or baselines (weights not included): SigLIP2 base, DINOv2 ViT-B/14 with registers, DINOv2 small, SAM 2.1 hiera tiny, EmbeddingGemma 2 and OWLv2 base (all Apache-2.0); CLIP RN50 (MIT, code and released weights); YOLOv10s (AGPL-3.0); ResNet-18, WRN-50 and R3D-18 torchvision weights (no separate weight licence is published; torchvision code is BSD-3-Clause, and the weights may carry terms from their training data, ImageNet and Kinetics). Mirage 4.0.1 is given as Apache-2.0 in the experiments' documents; this is not verified, because no official page was identified.
