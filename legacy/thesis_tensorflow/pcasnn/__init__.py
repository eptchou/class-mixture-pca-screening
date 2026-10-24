"""PCA-guided neuron screening for Siamese networks under class imbalance.

This package refactors the research notebook into a reusable library. The
core screening logic (PCA importance scoring, the NISP baseline, the
Siamese architecture, pair generation and evaluation) lives here and is
shared by every experiment script, so there is a single source of truth
rather than the repeated copies that appeared in the original notebook.

Public API
----------
Screening scores
    pca_importance_and_thresholds, calculate_inf_fs_scores,
    find_elbow_point_geometry
Models
    create_mlp_classifier, create_finetune_subnet, create_siamese_network
Pairs and evaluation
    create_pairs, evaluate_siamese
Data handling
    downsample_single_class, split_and_imbalance
Config
    ExperimentConfig
"""

from .config import ExperimentConfig
from .screening import (
    calculate_inf_fs_scores,
    find_elbow_point_geometry,
    pca_importance_and_thresholds,
    calculate_jaccard,
)
from .models import (
    create_mlp_classifier,
    create_finetune_subnet,
    create_siamese_network,
)
from .pairs import create_pairs, evaluate_siamese
from .data import downsample_single_class, split_and_imbalance
from .seeding import set_global_seed

__all__ = [
    "ExperimentConfig",
    "calculate_inf_fs_scores",
    "find_elbow_point_geometry",
    "pca_importance_and_thresholds",
    "calculate_jaccard",
    "create_mlp_classifier",
    "create_finetune_subnet",
    "create_siamese_network",
    "create_pairs",
    "evaluate_siamese",
    "downsample_single_class",
    "split_and_imbalance",
    "set_global_seed",
]

__version__ = "1.0.0"
