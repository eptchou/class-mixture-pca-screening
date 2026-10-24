#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
"${PYTHON:-python3}" diagnostics.py
"${PYTHON:-python3}" benchmark_timing.py
