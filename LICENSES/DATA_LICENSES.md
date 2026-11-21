# Dataset and model licences

Record of the licence of every dataset and model this repository uses, the official page it should be checked against, and whether that check was done.

**Check of 2026-11-21.** Each official page below was fetched and read on 2026-11-21 (Hugging Face API model cards for models; repository LICENSE files or dataset pages for datasets). The Status column gives what was found. Where a page states no licence, that is recorded as such.

No dataset file, image, feature file or model weight is in this repository.

## Datasets

| Dataset | Chapters | Official page | Licence recorded before the check | Status (2026-11-21) |
|---|---|---|---|---|
| MVTec AD | 01, 02, 03, 06, 10 | https://www.mvtec.com/company/research/datasets/mvtec-ad | CC BY-NC-SA 4.0 | Verified (Checked 2026-11-21, official page): CC BY-NC-SA 4.0; commercial use not allowed. |
| MVTec LOCO AD | 01, 02, 03, 04, 05, 10, 11, 13, 14, 15 | https://www.mvtec.com/company/research/datasets/mvtec-loco | CC BY-NC-SA 4.0 | Verified (Checked 2026-11-21, official page): CC BY-NC-SA 4.0; commercial use not allowed. The Hugging Face mirror used for two chapter 04 categories was not separately checked. |
| VisA (release 2022-09-22) | 01, 11, 13 | the publisher's `spot-diff` repository on GitHub, linked from the paper page https://arxiv.org/abs/2207.14315 | CC BY 4.0 | Verified (Checked 2026-11-21): LICENSE-DATASET in the publisher's spot-diff repository is CC BY 4.0. |
| Total-Text | 01 | https://github.com/cs-chan/Total-Text-Dataset | BSD-3-Clause | Verified (Checked 2026-11-21): repository LICENSE is BSD-3-Clause (Copyright 2017 Chee Seng Chan). |
| VisDrone2019-DET | 07, 09 | https://github.com/VisDrone/VisDrone-Dataset | Academic, non-commercial | Checked (Checked 2026-11-21): the official repository states no licence; research use per the challenge pages. Treat as research use only. |
| KTH Actions | 08 | https://www.csc.kth.se/cvap/actions/ | Research use; no licence text recorded; clips must not be redistributed | Verified (Checked 2026-11-21, official page): publicly available for non-commercial use, citation requested. |
| CIFAR-10 | 16 | https://www.cs.toronto.edu/~kriz/cifar.html | No formal licence; citation requested | Checked (Checked 2026-11-21, official page): no licence stated; citation requested. |
| GTSRB | 16 | https://benchmark.ini.rub.de/gtsrb_dataset.html | No formal licence; citation requested | Checked (Checked 2026-11-21, official page): no licence stated; citation requested. |
| EuroSAT | 16 | https://github.com/phelber/EuroSAT | MIT (dataset release); Sentinel-2 imagery is Copernicus open data | Verified (Checked 2026-11-21): repository LICENSE is MIT (2023, Patrick Helber). |
| Oxford-IIIT Pets | 16 | https://www.robots.ox.ac.uk/~vgg/data/pets/ | CC BY-SA 4.0 | Verified (Checked 2026-11-21, official page): CC BY-SA 4.0, commercial and research use; copyright stays with the image owners. |
| Oxford Flowers 102 | 16 (v1 only, not counted) | https://www.robots.ox.ac.uk/~vgg/data/flowers/102/ | No formal licence; research use | Checked (Checked 2026-11-21, official page): no licence stated. |
| Describable Textures Dataset (DTD) | 16 (features only) | https://www.robots.ox.ac.uk/~vgg/data/dtd/ | Research use only | Verified (Checked 2026-11-21, official page): made available for research purposes. |

## Models (weights are not included)

| Model | Chapters | Official page | Licence recorded before the check | Status (2026-11-21) |
|---|---|---|---|---|
| SigLIP2 base patch16-224 | 11, 13, 14, 15 | https://huggingface.co/google/siglip2-base-patch16-224 | Apache-2.0 | Verified (Checked 2026-11-21, Hugging Face model card): Apache-2.0. |
| DINOv2 ViT-B/14 with registers (timm `vit_base_patch14_reg4_dinov2.lvd142m`) | 13 | https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m | Apache-2.0 | Verified (Checked 2026-11-21, Hugging Face model card): Apache-2.0. |
| DINOv2 small (`facebook/dinov2-small`) | 16 | https://huggingface.co/facebook/dinov2-small | Apache-2.0 (the original 2023 research code release was CC BY-NC) | Verified (Checked 2026-11-21, Hugging Face model card): Apache-2.0. |
| SAM 2.1 hiera tiny | 11 (data preparation; masker not published) | https://huggingface.co/facebook/sam2.1-hiera-tiny | Apache-2.0 | Verified (Checked 2026-11-21, Hugging Face model card): Apache-2.0. |
| ResNet-18 and WRN-50 ImageNet weights (torchvision) | 01, 02, 03, 04 | https://pytorch.org/vision/stable/models.html | Not stated | Checked (Checked 2026-11-21): torchvision code is BSD-3-Clause; the docs say pretrained weights may carry their own terms derived from the training data (ImageNet, research use). No separate weight licence is published. |
| R3D-18 Kinetics-400 weights (torchvision) | 08 | https://pytorch.org/vision/stable/models.html | Not stated | Checked (Checked 2026-11-21): as above; weights may carry terms from the training data (Kinetics). No separate weight licence is published. |
| CLIP RN50 | 02, 03 | https://github.com/openai/CLIP | MIT (code); weights' terms not recorded | Verified (Checked 2026-11-21): repository LICENSE is MIT (code and released weights, 2021 OpenAI). |
| YOLOv10s COCO checkpoint | 07 | https://github.com/THU-MIG/yolov10 | AGPL-3.0 (Ultralytics-based code and weights) | Verified (Checked 2026-11-21): repository LICENSE is AGPL-3.0. |
| Mirage 4.0.1 (2-bit) | 09 | official page not identified in the experiments' documents | Apache-2.0 | Not verified (Checked 2026-11-21): no official page identified. |
| EmbeddingGemma 2 (`google/embeddinggemma-2`, language and vision tensors only) | 05 | https://huggingface.co/google/embeddinggemma-2 | Gemma terms of use (not recorded by the experiment) | Verified (Checked 2026-11-21, Hugging Face model card): Apache-2.0. |
| OWLv2 base | 03 (probe for chapter 04) | https://huggingface.co/docs/transformers/model_doc/owlv2 | Not recorded | Verified (Checked 2026-11-21, Hugging Face model cards for google/owlv2-base-patch16 and -ensemble): Apache-2.0. |

## Notes

No dataset file, image or weight is redistributed here. Non-commercial datasets (MVTec AD, MVTec LOCO AD, KTH, VisDrone, DTD) are used for research results only; the code in this repository is MIT and does not change those datasets' terms.
