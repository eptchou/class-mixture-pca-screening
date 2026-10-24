#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
S="1 2 3 4 5 6 7 8 9 10"
"${PYTHON:-python3}" run_condition.py reuters_nat 1.0 reuters_nat $S
"${PYTHON:-python3}" run_condition.py reuters_bal 1.0 reuters_bal $S
"${PYTHON:-python3}" run_condition.py fmnist 0.3 fmnist_r0.3 $S
"${PYTHON:-python3}" run_condition.py fmnist 0.02 fmnist_r0.02 $S
"${PYTHON:-python3}" run_condition.py fmnist 0.01 fmnist_r0.01 $S
