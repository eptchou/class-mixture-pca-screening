"""Compatibility entry point. Primary stability always uses stratified halves.

Usage: python stability.py <name> <ratio> <tag> [--mixture] [--timing]
Mixture results are generated automatically for fmnist_bal. --timing is
provided separately by benchmark_timing.py.
"""
import subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parent
subprocess.run([sys.executable,str(root/'diagnostics.py'),'--tags',sys.argv[3]],check=True)
if '--timing' in sys.argv:
    subprocess.run([sys.executable,str(root/'benchmark_timing.py')],check=True)
