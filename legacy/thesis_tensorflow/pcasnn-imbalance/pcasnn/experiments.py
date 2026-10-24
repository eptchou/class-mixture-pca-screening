"""Experiment routines shared by all scripts.

The original notebook implemented the same stability analysis, PRR sweep,
Kruskal--Wallis testing and chi-square independence testing several times
over, with small divergences between copies. They are unified here so that
the cross-domain and multi-ratio drivers call one implementation.

Functions
---------
run_stability_analysis
    Split-half Jaccard and Spearman stability of the PCA selection.
train_baseline_and_rank
    Train the baseline classifier and rank neurons by PCA and NISP.
run_prr_sweep
    Sweep K and record F1/accuracy-based performance retention (PRR) for
    PCA, NISP and random screening.
kruskal_with_posthoc
    Kruskal--Wallis omnibus test with epsilon-squared effect size and, if
    significant, Dunn's post hoc with Holm correction.
pca_vs_nisp_independence
    Chi-square independence and overlap of PCA vs NISP selections.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal, spearmanr
from sklearn.metrics import confusion_matrix
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping

from .config import ExperimentConfig
from .models import (
    create_finetune_subnet,
    create_mlp_classifier,
    create_siamese_network,
    slice_weights,
)
from .pairs import create_pairs, evaluate_siamese
from .screening import (
    calculate_inf_fs_scores,
    calculate_jaccard,
    pca_importance_and_thresholds,
)


# ----------------------------------------------------------------------
# Stability
# ----------------------------------------------------------------------
def run_stability_analysis(
    x_train, y_train, x_val, y_val, num_classes,
    cfg: ExperimentConfig, n_runs: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[dict]]:
    """Split-half Jaccard and Spearman stability of the PCA selection.

    The validation set is split into two halves; the PCA screening is run
    independently on each and the retained sets are compared by Jaccard at
    every threshold and by Spearman correlation over the full ranking. The
    baseline is retrained ``n_runs`` times to average out initialisation.

    Returns
    -------
    summary : DataFrame with one row per threshold (mean/std Jaccard, K).
    per_run : list of raw per-run records (including spearman_rho).
    """
    import tensorflow as tf

    n_runs = n_runs if n_runs is not None else cfg.stability_runs
    half = len(x_val) // 2
    val_a, val_b = x_val[:half], x_val[half:]

    per_run: List[dict] = []
    for r in range(n_runs):
        seed = 100 + r
        tf.random.set_seed(seed)
        np.random.seed(seed)

        model = create_mlp_classifier(
            x_train.shape[1:], num_classes, cfg.first_layer_neurons
        )
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        es = EarlyStopping(monitor="val_accuracy", patience=cfg.classifier_patience,
                           restore_best_weights=True)
        model.fit(x_train, y_train, validation_data=(x_val, y_val),
                  batch_size=cfg.batch_size, epochs=cfg.classifier_epochs,
                  verbose=0, callbacks=[es])

        imp_a, idx_a, thr_a = pca_importance_and_thresholds(
            model, val_a, cfg.batch_size, variance_thresholds=tuple(cfg.variance_thresholds)
        )
        imp_b, idx_b, thr_b = pca_importance_and_thresholds(
            model, val_b, cfg.batch_size, variance_thresholds=tuple(cfg.variance_thresholds)
        )

        record = {"run": r, "spearman_rho": float(spearmanr(imp_a, imp_b)[0])}
        for key in thr_a:
            ka, kb = thr_a[key], thr_b[key]
            record[f"jaccard_{key}"] = calculate_jaccard(set(idx_a[:ka]), set(idx_b[:kb]))
            record[f"kavg_{key}"] = (ka + kb) / 2
        per_run.append(record)

    df = pd.DataFrame(per_run)
    thresholds = [k[len("jaccard_"):] for k in df.columns if k.startswith("jaccard_")]
    summary = pd.DataFrame([
        {
            "Metric Point": t,
            "Avg Neurons (K)": df[f"kavg_{t}"].mean(),
            "Avg Jaccard Index": df[f"jaccard_{t}"].mean(),
            "Std Jaccard": df[f"jaccard_{t}"].std(),
        }
        for t in thresholds
    ])
    return summary, per_run


# ----------------------------------------------------------------------
# Baseline + ranking
# ----------------------------------------------------------------------
def train_baseline_and_rank(
    x_train, y_train, x_val, y_val, num_classes, cfg: ExperimentConfig,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, Dict[str, int], Dict[str, float]]:
    """Train the baseline and rank neurons by PCA and NISP.

    Returns
    -------
    baseline_weights : ``[W1, b1, W2]``
    pca_indices, nisp_indices : ranked neuron indices (descending)
    thresholds : PCA retention counts
    timings : {"pca": seconds, "nisp": seconds}
    """
    import tensorflow as tf

    tf.random.set_seed(cfg.base_seed)
    np.random.seed(cfg.base_seed)

    model = create_mlp_classifier(x_train.shape[1:], num_classes, cfg.first_layer_neurons)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    es = EarlyStopping(monitor="val_accuracy", patience=cfg.classifier_patience,
                       restore_best_weights=True)
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              batch_size=cfg.batch_size, epochs=cfg.classifier_epochs,
              verbose=0, callbacks=[es])

    w1, b1 = model.get_layer("dense1").get_weights()
    w2, _ = model.get_layer("dense2").get_weights()
    baseline_weights = [w1, b1, w2]

    # PCA ranking (timed)
    t0 = time.time()
    _, pca_indices, thresholds = pca_importance_and_thresholds(
        model, x_val, cfg.batch_size, variance_thresholds=tuple(cfg.variance_thresholds)
    )
    pca_time = time.time() - t0

    # NISP ranking (timed): Inf-FS on dense2 activations, propagated back
    t0 = time.time()
    act2 = models.Model(inputs=model.input,
                        outputs=model.get_layer("dense2").output).predict(
        x_val, batch_size=cfg.batch_size, verbose=0)
    inf_fs = calculate_inf_fs_scores(act2, alpha=cfg.inf_fs_alpha)
    nisp_scores = np.dot(np.abs(w2), inf_fs)
    nisp_indices = np.argsort(nisp_scores)[::-1]
    nisp_time = time.time() - t0

    return baseline_weights, pca_indices, nisp_indices, thresholds, {
        "pca": pca_time, "nisp": nisp_time,
    }


# ----------------------------------------------------------------------
# PRR sweep
# ----------------------------------------------------------------------
def _train_one_siamese(x_train, y_train, x_val, y_val, input_shape, k,
                       init_weights, cfg, n_train_pairs, n_val_pairs, seed):
    val_data, val_y = create_pairs(x_val, y_val, n_val_pairs, rng_seed=seed)
    tr_data, tr_y = create_pairs(x_train, y_train, n_train_pairs, rng_seed=seed)
    s_model = create_siamese_network(
        create_finetune_subnet(input_shape, k, init_weights), input_shape
    )
    s_model.compile(loss="binary_crossentropy", optimizer="adam")
    es = EarlyStopping(monitor="val_loss", patience=cfg.siamese_patience,
                       restore_best_weights=True)
    s_model.fit(tr_data, tr_y, validation_data=(val_data, val_y),
                batch_size=cfg.batch_size, epochs=cfg.siamese_epochs,
                verbose=0, callbacks=[es])
    return s_model


def run_prr_sweep(
    x_train, y_train, x_val, y_val, test_sets: Dict[str, tuple],
    baseline_weights, pca_indices, nisp_indices,
    cfg: ExperimentConfig, data_type: str = "image",
) -> pd.DataFrame:
    """Sweep K and record performance-retention rate for each strategy.

    Parameters
    ----------
    test_sets : dict
        Maps a label (e.g. "bal", "imb") to ``(x_test, y_test)``. The full
        model and every pruned model are scored on each, so both balanced
        and imbalanced test performance can be recorded in one pass.

    Returns
    -------
    DataFrame with columns Method, K, and for each test label F1_<label>,
    ACC_<label>, PRR_F1_<label>, PRR_ACC_<label>.
    """
    input_shape = x_train.shape[1:]
    n_train_pairs, n_val_pairs, n_test_pairs = cfg.pair_counts(data_type)
    methods = ["PCA-based", "NISP", "Random"]
    labels = list(test_sets.keys())
    rows: List[dict] = []

    # Full model (K = p), averaged over n_runs
    full = {lab: {"f1": [], "acc": []} for lab in labels}
    for run in range(cfg.n_runs):
        s_model = _train_one_siamese(
            x_train, y_train, x_val, y_val, input_shape, cfg.first_layer_neurons,
            baseline_weights, cfg, n_train_pairs, n_val_pairs, seed=999 + run,
        )
        for lab, (xt, yt) in test_sets.items():
            f1, acc = evaluate_siamese(s_model, xt, yt, cfg.num_test_sets,
                                       n_test_pairs, cfg.batch_size)
            full[lab]["f1"].append(f1)
            full[lab]["acc"].append(acc)
    full_f1 = {lab: float(np.mean(full[lab]["f1"])) for lab in labels}
    full_acc = {lab: float(np.mean(full[lab]["acc"])) for lab in labels}

    base_row = {"Method": None, "K": cfg.first_layer_neurons}
    for m in methods:
        row = dict(base_row, Method=m)
        for lab in labels:
            row[f"F1_{lab}"] = full_f1[lab]
            row[f"ACC_{lab}"] = full_acc[lab]
            row[f"PRR_F1_{lab}"] = 1.0
            row[f"PRR_ACC_{lab}"] = 1.0
        rows.append(row)

    # Pruned models
    for k in cfg.scan_points():
        if k == cfg.first_layer_neurons:
            continue
        for method in methods:
            agg = {lab: {"f1": [], "acc": []} for lab in labels}
            for run in range(cfg.n_runs):
                if method == "PCA-based":
                    idx = pca_indices[:k]
                elif method == "NISP":
                    idx = nisp_indices[:k]
                else:
                    idx = np.random.choice(range(cfg.first_layer_neurons), k, replace=False)
                init_w = slice_weights(baseline_weights, idx)
                s_model = _train_one_siamese(
                    x_train, y_train, x_val, y_val, input_shape, k, init_w,
                    cfg, n_train_pairs, n_val_pairs, seed=1000 + run + k,
                )
                for lab, (xt, yt) in test_sets.items():
                    f1, acc = evaluate_siamese(s_model, xt, yt, cfg.num_test_sets,
                                               n_test_pairs, cfg.batch_size)
                    agg[lab]["f1"].append(f1)
                    agg[lab]["acc"].append(acc)

            row = {"Method": method, "K": k}
            for lab in labels:
                mf1, macc = float(np.mean(agg[lab]["f1"])), float(np.mean(agg[lab]["acc"]))
                row[f"F1_{lab}"] = mf1
                row[f"ACC_{lab}"] = macc
                row[f"PRR_F1_{lab}"] = mf1 / full_f1[lab] if full_f1[lab] > 0 else np.nan
                row[f"PRR_ACC_{lab}"] = macc / full_acc[lab] if full_acc[lab] > 0 else np.nan
            rows.append(row)

    return pd.DataFrame(rows).sort_values(["Method", "K"]).reset_index(drop=True)


# ----------------------------------------------------------------------
# Inference: Kruskal-Wallis + Dunn
# ----------------------------------------------------------------------
def epsilon_squared_kw(h_stat: float, k_groups: int, n_total: int) -> float:
    """Epsilon-squared effect size for a Kruskal--Wallis test."""
    return (h_stat - k_groups + 1) / (n_total - k_groups) if (n_total - k_groups) > 0 else np.nan


def kruskal_with_posthoc(
    groups_by_ratio: Dict[float, np.ndarray],
    do_posthoc: bool = True,
) -> Tuple[float, float, float, Optional[pd.DataFrame]]:
    """Kruskal--Wallis omnibus test with optional Dunn post hoc (Holm).

    Parameters
    ----------
    groups_by_ratio : dict
        Maps each imbalance ratio to its array of metric values.

    Returns
    -------
    (H, p_value, epsilon_squared, dunn_table_or_None)
    """
    ratios = list(groups_by_ratio.keys())
    groups = [np.asarray(groups_by_ratio[r]) for r in ratios]
    h_stat, p = kruskal(*groups)
    n_total = sum(len(g) for g in groups)
    eps2 = epsilon_squared_kw(h_stat, len(groups), n_total)

    dunn = None
    if do_posthoc and p < 0.05:
        import scikit_posthocs as sp

        long = pd.DataFrame(
            {"ratio": np.repeat(ratios, [len(g) for g in groups]),
             "value": np.concatenate(groups)}
        )
        dunn = sp.posthoc_dunn(long, val_col="value", group_col="ratio", p_adjust="holm")
    return float(h_stat), float(p), float(eps2), dunn


# ----------------------------------------------------------------------
# PCA vs NISP independence
# ----------------------------------------------------------------------
def pca_vs_nisp_independence(
    pca_indices: np.ndarray, nisp_indices: np.ndarray, k: int, n_neurons: int,
) -> Tuple[float, int]:
    """Chi-square p-value and overlap for one pair of top-k selections.

    Returns
    -------
    (chi2_p_value, overlap_count)
    """
    pca_sel = np.zeros(n_neurons, dtype=bool)
    nisp_sel = np.zeros(n_neurons, dtype=bool)
    pca_sel[pca_indices[:k]] = True
    nisp_sel[nisp_indices[:k]] = True

    table = confusion_matrix(pca_sel, nisp_sel)
    try:
        _, p_val, _, _ = chi2_contingency(table)
    except ValueError:
        p_val = np.nan
    overlap = len(set(pca_indices[:k]).intersection(set(nisp_indices[:k])))
    return float(p_val), int(overlap)
