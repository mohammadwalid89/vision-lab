#!/usr/bin/env bash
# Phase 2 in the order it ran: score every image, seal the score files, check the fast selector against the
# original, run the 300-split selection and freeze, then open the fresh set exactly once.
#   ./run_sealed.sh score   writes out/{wrn,comp,base}_scores.jsonl (needs out/inventory.json, VISION_V31_ROOT and state/reference.json)
#   ./run_sealed.sh seal    everything after scoring; refuses to run if a policy or final result already exists
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 SYSTEM_ONE_THREADS=1
log() { echo "$(date -u +'%Y-%m-%dT%H:%M:%SZ') $*"; }
run() { # NAME THREADS SCRIPT ARGS...
  local name=$1 threads=$2; shift 2
  set +e
  OMP_NUM_THREADS=$threads P2_THREADS=$threads "$PY" -W ignore "$@" > "out/$name.log" 2>&1
  echo "exit $?" >> "out/$name.log"
  set -e
}

case "${1:-}" in
score)
  mkdir -p out
  run wrn_run 3 p2_patchcore_wrn.py bottle cable hazelnut pill tile transistor breakfast_box pushpins screw_bag
  run comp_run4_v1 1 p2_components.py breakfast_box pushpins screw_bag
  run base_run 1 p2_baseline_scores.py
  ;;
seal)
  export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
  for name in wrn_run base_run comp_run4_v1; do
    grep -qx 'exit 0' "out/${name}.log" || { log "BLOCKED: out/${name}.log has no successful exit. Do NOT open final."; exit 11; }
  done
  "$PY" - <<'PY'
import json
from collections import Counter
from pathlib import Path
p = Path("out")
expected = {"bottle": 80, "cable": 137, "hazelnut": 100, "pill": 152, "tile": 108, "transistor": 90,
            "breakfast_box": 321, "pushpins": 354, "screw_bag": 382}
kits = ["breakfast_box", "pushpins", "screw_bag"]
for filename, want in [("base_scores.jsonl", expected), ("wrn_scores.jsonl", expected),
                       ("comp_scores.jsonl", {k: expected[k] for k in kits})]:
    rows = [json.loads(s) for s in open(p / filename) if s.strip()]
    counts = Counter(r["category"] for r in rows)
    assert counts == want, (filename, counts, want)
    assert len(rows) == len({r["sha256"] for r in rows}), filename + " duplicates"
    print("INPUT COMPLETE", filename, len(rows), flush=True)
for name in ["p2_dev_results.json", "p2_frozen_policy.json", "p2_final_results.json"]:
    assert not (p / name).exists(), name + " already exists; avoid reopening or overwriting"
PY
  sha256sum ../PREREGISTRATION.md p2_policy.py p2_policy_fast.py p2_final.py out/base_scores.jsonl out/wrn_scores.jsonl out/comp_scores.jsonl > out/p2_sealed_inputs.sha256
  log 'Inputs sealed; checking the vectorised selector against the original.'
  "$PY" -u p2_policy_fast.py --regression > out/p2_selector_regression.log 2>&1 || { log 'BLOCKED: selector regression'; exit 14; }
  log 'Regression passed. Running the 300-split selection on exposed images.'
  "$PY" -u p2_policy_fast.py 300 > out/p2_policy_run.log 2>&1 || { log 'BLOCKED: policy evaluation failed; final remains unopened'; exit 12; }
  test -s out/p2_frozen_policy.json
  sha256sum out/p2_frozen_policy.json > out/p2_frozen_policy.sha256
  log 'Policy frozen and hashed. Opening the fresh set once.'
  "$PY" -u p2_final.py > out/p2_final_run.log 2>&1 || { log 'BLOCKED: final evaluation failed'; exit 13; }
  test -s out/p2_final_results.json
  sha256sum out/p2_final_results.json > out/p2_final_results.sha256
  log 'Done.'
  ;;
*)
  echo "usage: $0 score|seal" >&2; exit 2 ;;
esac
