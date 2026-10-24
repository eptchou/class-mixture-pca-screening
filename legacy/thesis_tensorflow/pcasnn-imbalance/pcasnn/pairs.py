"""Pair construction and Siamese evaluation.

The Siamese network consumes labelled pairs. Positive pairs draw two
samples from a class chosen uniformly at random (so every class
contributes evenly regardless of its size, which is what protects the
minority class); negative pairs draw one sample from each of two distinct
classes. Even indices are positive and odd indices negative, giving a
balanced pair label distribution.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def create_pairs(
    x: np.ndarray,
    y: np.ndarray,
    num_pairs: int = 10000,
    rng_seed: Optional[int] = None,
    return_pair_classes: bool = False,
):
    """Generate labelled same/different-class pairs.

    Parameters
    ----------
    x, y : arrays
        Samples and integer class labels.
    num_pairs : int
        Number of pairs to generate (half positive, half negative).
    rng_seed : int, optional
        Seed for the pair-sampling RNG.
    return_pair_classes : bool
        If True, also return the per-side class labels ``(y_left,
        y_right)``, used by the per-class analysis.

    Returns
    -------
    [left, right], labels[, y_left, y_right]
    """
    rng = np.random.default_rng(rng_seed)
    classes = np.unique(y)
    class_to_idx = {c: np.where(y == c)[0] for c in classes}
    valid_classes = [c for c in classes if len(class_to_idx[c]) > 1]
    if not valid_classes:
        raise ValueError("No class has >= 2 samples; cannot create positive pairs.")

    left = np.empty((num_pairs, *x.shape[1:]), dtype=x.dtype)
    right = np.empty((num_pairs, *x.shape[1:]), dtype=x.dtype)
    labels = np.empty((num_pairs,), dtype=int)
    y_left = np.empty((num_pairs,), dtype=int)
    y_right = np.empty((num_pairs,), dtype=int)

    for k in range(num_pairs):
        if k % 2 == 0:  # positive pair
            c = rng.choice(valid_classes)
            i1, i2 = rng.choice(class_to_idx[c], 2, replace=False)
            labels[k], y_left[k], y_right[k] = 1, c, c
        else:  # negative pair
            c1, c2 = rng.choice(classes, 2, replace=False)
            i1 = rng.choice(class_to_idx[c1])
            i2 = rng.choice(class_to_idx[c2])
            labels[k], y_left[k], y_right[k] = 0, c1, c2
        left[k], right[k] = x[i1], x[i2]

    if return_pair_classes:
        return [left, right], labels, y_left, y_right
    return [left, right], labels


def evaluate_siamese(
    s_model,
    test_x: np.ndarray,
    test_y: np.ndarray,
    num_sets: int = 10,
    pairs_per_set: int = 4000,
    batch_size: int = 128,
    seed_base: int = 2000,
) -> Tuple[float, float]:
    """Mean F1 and accuracy over ``num_sets`` independent test-pair sets.

    Returns
    -------
    (mean_f1, mean_accuracy)
    """
    f1_list: List[float] = []
    acc_list: List[float] = []
    for t in range(num_sets):
        pairs, labels = create_pairs(test_x, test_y, pairs_per_set, rng_seed=seed_base + t)
        probs = s_model.predict(pairs, batch_size=batch_size, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)
        f1_list.append(f1_score(labels, preds))
        acc_list.append(accuracy_score(labels, preds))
    return float(np.mean(f1_list)), float(np.mean(acc_list))
