#!/usr/bin/env bash
# Run inside an existing single-GPU allocation; this script never submits jobs.
set -eo pipefail
if [ "$#" -ne 3 ]; then
  echo "Usage: bash scripts/run_smoke.sh FOUNDATION.demff AUTHOR_POSCAR NEW_OUTPUT_DIR" >&2
  exit 2
fi
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
MANIFEST=$(realpath -- "$1")
POSCAR_PATH=$(realpath -- "$2")
OUTPUT=$(realpath -m -- "$3")
mkdir -- "$OUTPUT"
python "$SCRIPT_DIR/check_environment.py" > "$OUTPUT/environment.json"
python -m unittest discover -s "$SCRIPT_DIR/../tests" -v
python "$SCRIPT_DIR/validate_inference.py" "$MANIFEST" "$POSCAR_PATH" "$OUTPUT/inference"
python "$SCRIPT_DIR/validate_training.py" "$MANIFEST" "$OUTPUT/training"
