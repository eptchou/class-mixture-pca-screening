"""Experiment configuration.

The original notebook kept its settings in module-level globals that were
duplicated (and occasionally drifted) across cells. They are collected
here in one immutable dataclass so that every script is configured the
same way and any change is made in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ExperimentConfig:
    """Hyperparameters and protocol settings for all experiments."""

    # --- Architecture ---
    first_layer_neurons: int = 128
    batch_size: int = 128

    # --- Training length ---
    classifier_epochs: int = 15
    siamese_epochs: int = 10
    classifier_patience: int = 3
    siamese_patience: int = 2

    # --- Repetitions ---
    n_runs: int = 3          # PRR curves: retrainings per K
    num_test_sets: int = 10  # PRR curves: independent test-pair sets per model
    stability_runs: int = 3  # cross-domain stability replicates
    multiratio_stability_runs: int = 30  # multi-ratio stability replicates

    # --- Data splitting (train/val/test = 0.63/0.27/0.10) ---
    test_size: float = 0.10
    val_size_in_trainval: float = 0.30

    # --- Imbalance ---
    minority_class_label: int = 0
    min_keep_per_split: int = 5

    # --- Screening ---
    use_full_scan: bool = True
    inf_fs_alpha: float = 0.2
    base_seed: int = 42

    # --- Variance thresholds reported in the paper ---
    variance_thresholds: List[float] = field(
        default_factory=lambda: [0.85, 0.90, 0.95, 0.99]
    )

    def scan_points(self) -> List[int]:
        """Return the list of K values (retained-neuron counts) to sweep."""
        p = self.first_layer_neurons
        if self.use_full_scan:
            pts = list(range(1, p + 1))
        else:
            pts = list(range(1, 21)) + list(range(25, 65, 5)) + list(range(70, p + 1, 10))
        pts = sorted(set(pts))
        if p not in pts:
            pts.append(p)
        return pts

    def pair_counts(self, data_type: str) -> tuple[int, int, int]:
        """(train, val, test) pair counts, scaled to data-set size."""
        if data_type == "image":
            return 20000, 4000, 4000
        return 5000, 1000, 1000


# A ready-to-use default; scripts may override fields via dataclasses.replace.
DEFAULT_CONFIG = ExperimentConfig()
