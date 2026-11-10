#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/scripts/env.sh"
cd "$REPO"
cmd="${1:-help}"
[[ $# == 0 ]] || shift
config=configs/main.json
case "$cmd" in
  verify) "$PYTHON" scripts/verify_package.py ;;
  test)
    "$PYTHON" -m unittest discover -s tests -v ;;
  prepare)
    "$PYTHON" scripts/prepare_data.py "${DATA_ARGS[@]}" ;;
  smoke)
    "$PYTHON" -u scripts/smoke.py --config "$config" "${DATA_ARGS[@]}" --output "${1:-runs/smoke}" ;;
  train|baseline|recency-baseline)
    output="${1:?fresh output directory required}"; seed="${2:-2026}"; device="${3:-cuda}"
    [[ ! -e "$output" ]] || { echo "Output exists: $output" >&2; exit 1; }
    [[ "$cmd" != baseline ]] || config=configs/ablation_training_only.json
    [[ "$cmd" != recency-baseline ]] || config=configs/ablation_recency_only.json
    "$PYTHON" -u -m mupsr.train --config "$config" "${DATA_ARGS[@]}" \
      --output "$output" --seed "$seed" --device "$device" ;;
  evaluate)
    "$PYTHON" -u -m mupsr.evaluate --checkpoint "${1:?checkpoint required}" \
      --output "${2:?fresh evaluation output required}" --device "${3:-cuda}" "${DATA_ARGS[@]}" ;;
  diagnose)
    "$PYTHON" -u -m mupsr.diagnostics "${DATA_ARGS[@]}" --out "${1:-runs/diagnostics}" ;;
  compare)
    "$PYTHON" scripts/accept_experiment.py --base "${1:?baseline directory required}" \
      --candidate "${2:?candidate directory required}" --diagnostics "${3:-runs/diagnostics}" \
      --output "${4:-runs/acceptance_valid.json}" ;;
  *)
    echo 'Usage: bash run.sh verify|test|prepare|smoke [out]|train out [seed] [cuda|cpu]|evaluate checkpoint out [device]'
    echo 'Optional: baseline out [seed] [device] | diagnose [out] | recency-baseline out [seed] [device] | compare baseline_dir candidate_dir [diagnostics_dir] [report]'
    [[ "$cmd" == help ]] || exit 2 ;;
esac
