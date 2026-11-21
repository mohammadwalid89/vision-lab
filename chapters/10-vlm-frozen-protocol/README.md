# 10. Vision-language models: frozen protocol

**Question.** Do small vision-language models cut unsure referrals without releasing more defects?

**Setup.** 1,000 official public test images frozen before any inference: 500 MVTec AD surface and 500 MVTec LOCO kit images; per task 200 good and 300 defective. Two zero-shot triage models, SmolVLM2-500M-Video-Instruct and Qwen3-VL-2B-Instruct, on the same 1,000 images with a fixed GOOD / ANOMALY / UNSURE parser, would make exactly 2,000 primary predictions. The win rule was written in advance. Florence-2 is a caption baseline only, and PaddleOCR-VL is an OCR model, not a ranked anomaly detector. Engineering checks used a separate 32-image set.

**Result.** From the frozen-protocol status and runbook as quoted in the Lab research atlas of 2026-11-11 (those files are not published and not in this repo). No primary prediction was made. Engineering feasibility on one CPU core:

| Model and input | Seconds per image |
| --- | ---: |
| SmolVLM2-500M, cropped to 512 px, one engineering prompt (output UNSURE) | 29.93 |
| Florence-2 caption (after a cache workaround; four tokens) | 46.4 |
| Qwen3-VL-2B | timed out at 420 |

SmolVLM2 at full resolution also timed out at 420 s. Six model assets total 20.0626 GB with verified SHA256. 2,000 predictions exceed the 10-hour one-core budget set in advance, so none were scored. One model alone would need about 8.3 hours on one core (29.93 s times 1,000, excluding overhead).

**What it means.** The comparison is waiting, not failed: the test set was not reduced to fit the hardware, and no accuracy claim is made. Any new adapter, precision or compute choice has to be declared before the primary images are opened. A one-model amendment is the planned next step.

**How to run.** Not available. See Status.

**Status.** Protocol frozen, not run; code not in this repository. The protocol, prompts, image-ID lists, validator and tracker lived in a separate handoff folder. It was not part of the archive restore of 2026-11-21 (which brought back the code for chapters 03 to 05), and no inference or scoring script for this study remains in the sources used for this repo (the earlier v3 local-VLM module in the cascade package is a different, earlier experiment). The figures above come from the atlas and were not re-measured; nothing was run for this chapter. If the folder is restored, the chapter should hold the frozen protocol and prompt JSON, the ID lists with checksums, and the validator.
