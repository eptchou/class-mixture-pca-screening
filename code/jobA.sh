#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
S="1 2 3 4 5 6 7 8 9 10"
"${PYTHON:-python3}" run_condition.py fmnist 0.1 fmnist_r0.1 $S
"${PYTHON:-python3}" run_condition.py fmnist 1.0 fmnist_bal $S
"${PYTHON:-python3}" run_condition.py fmnist 0.002 fmnist_r0.002 $S
"${PYTHON:-python3}" run_condition.py mfeat 0.1 mfeat_r0.1 $S
"${PYTHON:-python3}" run_condition.py mfeat 1.0 mfeat_bal $S
