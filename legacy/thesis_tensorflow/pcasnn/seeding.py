"""Global seeding for reproducibility.

The notebook set the same three environment variables and seeded numpy,
random and TensorFlow in many places. That logic is centralised here.
Note that full bitwise determinism on GPU also depends on the CUDA/cuDNN
build; the flags below request deterministic ops where available.
"""

from __future__ import annotations

import os
import random

import numpy as np


def _enable_deterministic_env() -> None:
    os.environ.setdefault("PYTHONHASHSEED", "42")
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")


def set_global_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and TensorFlow, and request deterministic ops."""
    _enable_deterministic_env()
    random.seed(seed)
    np.random.seed(seed)
    # Import lazily so that importing the package does not force a TF import
    # in contexts (e.g. docs) that do not need it.
    import tensorflow as tf

    tf.random.set_seed(seed)
