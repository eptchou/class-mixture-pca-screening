"""Stratified splitting and controlled class imbalance.

Splitting is always done first, on the balanced data, so that class
proportions are identical across the train/validation/test splits; the
imbalance is then imposed on each split by subsampling the designated
minority class. This ordering keeps the test distribution under the
experimenter's control and avoids leakage between splits.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.model_selection import train_test_split


def downsample_single_class(
    x: np.ndarray,
    y: np.ndarray,
    target_class: int,
    keep_ratio: float,
    seed: int = 42,
    min_keep: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample one class to ``keep_ratio`` of its size (>= ``min_keep``).

    If ``keep_ratio >= 1`` the data are returned shuffled but unchanged.
    """
    rng = np.random.default_rng(seed)
    if keep_ratio >= 1.0:
        idx = np.arange(len(y))
        rng.shuffle(idx)
        return x[idx], y[idx]

    keep_idx = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if c == target_class:
            n_keep = max(int(len(idx) * keep_ratio), min_keep)
            n_keep = min(n_keep, len(idx))
            idx = rng.choice(idx, n_keep, replace=False)
        keep_idx.extend(idx)
    keep_idx = np.array(keep_idx)
    rng.shuffle(keep_idx)
    return x[keep_idx], y[keep_idx]


def split_and_imbalance(
    x_all: np.ndarray,
    y_all: np.ndarray,
    ratio: float,
    test_size: float = 0.10,
    val_size_in_trainval: float = 0.30,
    minority_class_label: int = 0,
    min_keep: int = 5,
    seed_base: int = 42,
):
    """Split into train/val/test, then impose imbalance on each split.

    Returns
    -------
    (x_train, y_train), (x_val, y_val), (x_test_bal, y_test_bal),
    (x_test_imb, y_test_imb)

    The test set is returned in both a balanced and an imbalanced form so
    that a model can be scored under either distribution.
    """
    x_trainval, x_test_bal, y_trainval, y_test_bal = train_test_split(
        x_all, y_all, test_size=test_size, stratify=y_all, random_state=seed_base
    )
    x_train_bal, x_val_bal, y_train_bal, y_val_bal = train_test_split(
        x_trainval, y_trainval, test_size=val_size_in_trainval,
        stratify=y_trainval, random_state=seed_base,
    )

    x_train, y_train = downsample_single_class(
        x_train_bal, y_train_bal, minority_class_label, ratio,
        seed=seed_base + 1, min_keep=min_keep,
    )
    x_val, y_val = downsample_single_class(
        x_val_bal, y_val_bal, minority_class_label, ratio,
        seed=seed_base + 2, min_keep=min_keep,
    )
    x_test_imb, y_test_imb = downsample_single_class(
        x_test_bal, y_test_bal, minority_class_label, ratio,
        seed=seed_base + 3, min_keep=min_keep,
    )
    return (
        (x_train, y_train),
        (x_val, y_val),
        (x_test_bal, y_test_bal),
        (x_test_imb, y_test_imb),
    )


def print_counts(tag: str, y: np.ndarray) -> None:
    """Print per-class counts for a split (diagnostic)."""
    cls, cnt = np.unique(y, return_counts=True)
    msg = ", ".join(f"{c}:{n}" for c, n in zip(cls, cnt))
    print(f"{tag} => total={len(y)} | {msg}")
