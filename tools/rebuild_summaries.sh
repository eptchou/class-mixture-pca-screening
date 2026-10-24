#!/usr/bin/env bash
set -euo pipefail
pkg_root="$(cd "$(dirname "$0")/.." && pwd)"
rebuild_root="$pkg_root/rebuilt"
mkdir -p "$rebuild_root/source_multifile"
if [ -e "$rebuild_root/results" ]; then
  echo 'rebuilt/results already exists; choose a clean destination or inspect it before repeating.' >&2
  exit 1
fi
cp -R "$pkg_root/code/results" "$rebuild_root/results"
export ADAC_RESULTS_DIR="$rebuild_root/results"
export ADAC_DATA_DIR="$pkg_root/code/data"
export ADAC_PAPER_DIR="$rebuild_root/source_multifile"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python "$pkg_root/code/aggregate.py"
python "$pkg_root/code/analysis_extra.py"
python "$pkg_root/code/revision_sensitivity.py"
python "$pkg_root/code/audit_fresh_results.py"
python "$pkg_root/tools/generate_sign_table.py"
echo "Tables, figures and tests written to $rebuild_root. No neural models were trained."
