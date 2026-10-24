"""Neuron-importance scores and screening thresholds.

Three ways of ranking the first hidden layer's neurons are provided:

* ``pca_importance_and_thresholds`` -- the proposed method. Each neuron is
  scored by the sum of its absolute PCA loadings weighted by the
  explained-variance ratios of the components; five data-driven cutoffs
  (elbow point and the 85/90/95/99% cumulative-variance counts) are
  returned alongside.
* ``calculate_inf_fs_scores`` -- the Infinite Feature Selection score used
  by the NISP baseline (propagated backward through the weights in the
  model module).
* random selection -- implemented inline in the experiment scripts.

``find_elbow_point_geometry`` and ``calculate_jaccard`` are small helpers
used by the above and by the stability analysis.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA


def calculate_inf_fs_scores(activations: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Infinite Feature Selection (Inf-FS) scores for a set of features.

    A fully connected affinity graph is built over features from a convex
    mix of (i) pairwise maximum standard deviation and (ii) one minus the
    absolute Spearman correlation, and the score of each feature is the
    sum over all path lengths, obtained in closed form via the matrix
    inverse ``(I - r A)^{-1} - I``. A tiny Gaussian jitter avoids
    degenerate correlations on constant activations.

    Parameters
    ----------
    activations : array of shape (n_samples, n_features)
    alpha : float
        Weight on the dispersion term versus the correlation term.

    Returns
    -------
    scores : array of shape (n_features,)
    """
    num_features = activations.shape[1]
    noise = np.random.normal(0, 1e-8, activations.shape)
    acts_noisy = activations + noise

    stds = np.std(acts_noisy, axis=0)
    max_std = np.max(stds)
    stds_norm = stds / max_std if max_std > 0 else stds
    sigma_matrix = np.maximum.outer(stds_norm, stds_norm)

    corr_matrix, _ = spearmanr(acts_noisy, axis=0)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    c_matrix = 1.0 - np.abs(corr_matrix)

    affinity = alpha * sigma_matrix + (1.0 - alpha) * c_matrix
    np.fill_diagonal(affinity, 0)

    eigenvalues = np.linalg.eigvals(affinity)
    rho = np.max(np.abs(eigenvalues))
    r = 0.9 / rho if rho > 0 else 0.0

    identity = np.eye(num_features)
    inverse_term = np.linalg.inv(identity - r * affinity)
    s_tilde = inverse_term - identity
    return np.real(np.sum(s_tilde, axis=1))


def find_elbow_point_geometry(explained_variances: np.ndarray) -> int:
    """Return the elbow index of a curve by maximum distance to the chord.

    The point on the explained-variance curve that is farthest from the
    straight line joining its first and last points is taken as the elbow.
    """
    x = np.arange(1, len(explained_variances) + 1)
    y = np.asarray(explained_variances)
    p1, p2 = np.array([x[0], y[0]]), np.array([x[-1], y[-1]])
    numerator = np.abs(
        (p2[1] - p1[1]) * x - (p2[0] - p1[0]) * y + p2[0] * p1[1] - p2[1] * p1[0]
    )
    denominator = np.sqrt((p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return int(x[np.argmax(numerator / denominator)])


def pca_importance_and_thresholds(
    model,
    data_samples: np.ndarray,
    batch_size: int = 128,
    activation_layer: str = "dense1_activation",
    variance_thresholds: Tuple[float, ...] = (0.85, 0.90, 0.95, 0.99),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """PCA-based neuron importance and data-driven retention thresholds.

    Parameters
    ----------
    model : keras Model
        A trained classifier exposing the first-layer post-activation
        output under ``activation_layer``.
    data_samples : array
        Inputs (typically the validation set) whose activations are
        decomposed.
    variance_thresholds : tuple of float
        Cumulative-variance levels for which the neuron count is reported.

    Returns
    -------
    importance : array of shape (n_neurons,)
        The per-neuron screening score ``sum_k |V_ki| * lambda_k``.
    sorted_indices : array
        Neuron indices in decreasing importance.
    thresholds : dict
        Retention counts for the elbow point and each variance level,
        keyed ``"Elbow Point"``, ``"Var 85%"``, ... .
    """
    from tensorflow.keras import models

    act_model = models.Model(
        inputs=model.input, outputs=model.get_layer(activation_layer).output
    )
    activations = act_model.predict(data_samples, batch_size=batch_size, verbose=0)

    pca = PCA(n_components=None, random_state=random_state).fit(activations)
    vars_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(vars_ratio)

    importance = np.sum(np.abs(pca.components_) * vars_ratio[:, np.newaxis], axis=0)
    sorted_indices = np.argsort(importance)[::-1]

    thresholds: Dict[str, int] = {"Elbow Point": find_elbow_point_geometry(vars_ratio)}
    for tau in variance_thresholds:
        thresholds[f"Var {int(tau * 100)}%"] = int(np.argmax(cum_var >= tau) + 1)

    return importance, sorted_indices, thresholds


def calculate_jaccard(set1: set, set2: set) -> float:
    """Jaccard index between two neuron-index sets."""
    union = len(set1.union(set2))
    return len(set1.intersection(set2)) / union if union > 0 else 0.0
