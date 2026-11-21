#!/usr/bin/env bash
# Phase 3 in the order it ran: download two new MVTec LOCO categories, fix the 40/60 split from file paths
# alone, fit and score every candidate on development images, check final-scoring code on validation
# normals, freeze, open the locked final set once, write the report. Each step refuses to redo a finished one.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p out

[ -f out/DOWNLOAD_COMPLETE ] || "$PY" -u download_new_categories.py
[ -f out/p3_split_manifest.json ] || "$PY" -u build_split.py

if [ ! -f out/P3_DEV_MODELS_COMPLETE ]; then
  for model in component resnet wrn; do
    for cat in juice_bottle splicing_connectors; do
      echo "RUN $(date -u +%FT%TZ) $model $cat"
      "$PY" -u p3_models.py --model "$model" --category "$cat"
    done
  done
  date -u +%FT%TZ > out/P3_DEV_MODELS_COMPLETE
fi

"$PY" -u p3_preflight.py > out/p3_preflight.log 2>&1
[ -f out/p3_frozen_policy.json ] || "$PY" -u p3_select_freeze.py > out/p3_selection.log 2>&1
if [ -f out/P3_FINAL_OPENED ]; then echo 'BLOCK: final previously opened; no automatic re-run'; exit 7; fi
"$PY" -u p3_final.py > out/p3_final.log 2>&1
"$PY" -u p3_report.py > out/p3_report.log 2>&1
date -u +%FT%TZ > out/P3_COMPLETE
