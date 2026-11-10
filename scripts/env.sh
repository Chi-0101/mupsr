#!/usr/bin/env bash
# Shared environment for run.sh: repo root, interpreter, data paths and thread limits.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"
MUPSR_SOURCE="${MUPSR_SOURCE:-$REPO/data/source}"
MUPSR_DATA="${MUPSR_DATA:-$REPO/work/prepared}"
MUPSR_CACHE="${MUPSR_CACHE:-$REPO/work/events}"
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
DATA_ARGS=(--data "$MUPSR_DATA" --source "$MUPSR_SOURCE" --cache "$MUPSR_CACHE")
