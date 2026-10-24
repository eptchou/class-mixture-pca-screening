#!/usr/bin/env python
"""Multi-ratio robustness study on Fashion-MNIST.

For each imbalance ratio in ``--ratios`` this:

1. repeats the split-half PCA stability analysis ``--stability-runs`` times
   (30 in the paper) and saves the per-run Jaccard/Spearman values,
2. trains a baseline, ranks neurons by PCA and NISP, records their ranking
   times, and runs a PRR sweep evaluated on both balanced and imbalanced
   test sets.

After all ratios, run ``run_analysis.py`` to produce the Kruskal--Wallis /
Dunn tables and the PCA-vs-NISP independence table from the saved per-run
files.

Example
-------
    python scripts/run_multi_ratio.py --ratios 0.3 0.1 0.02 0.01 0.002 \
        --out results/multi_ratio
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

from pcasnn import ExperimentConfig, set_global_seed
from pcasnn.datasets import load_fashion_mnist
from pcasnn.data import split_and_imbalance, print_counts
from pcasnn.experiments import (
    run_stability_analysis,
    train_baseline_and_rank,
    run_prr_sweep,
)
from pcasnn.plotting import plot_prr_curve


def run_one_ratio(ratio: float, x_all, y_all, cfg: ExperimentConfig,
                  out_dir: str, make_plots: bool) -> None:
    print(f"\n{'=' * 70}\nRatio = {ratio}\n{'=' * 70}")
    num_classes = len(np.unique(y_all))

    set_global_seed(cfg.base_seed)
    (x_tr, y_tr), (x_val, y_val), (x_te_bal, y_te_bal), (x_te_imb, y_te_imb) = \
        split_and_imbalance(
            x_all, y_all, ratio, cfg.test_size, cfg.val_size_in_trainval,
            cfg.minority_class_label, cfg.min_keep_per_split, cfg.base_seed,
        )
    print_counts("Train_imb", y_tr)
    print_counts("Val_imb  ", y_val)
    print_counts("Test_imb ", y_te_imb)

    # 1. Stability (30 runs), keep per-run records for later inference
    _, per_run = run_stability_analysis(
        x_tr, y_tr, x_val, y_val, num_classes, cfg,
        n_runs=cfg.multiratio_stability_runs,
    )
    df_runs = pd.DataFrame(per_run)
    df_runs.insert(0, "ratio", ratio)
    df_runs.to_csv(os.path.join(out_dir, f"stability_ratio_{ratio}.csv"), index=False)
    print("  stability: mean Spearman rho = "
          f"{df_runs['spearman_rho'].mean():.4f}")

    # 2. Baseline + ranking + PRR sweep (bal & imb test sets)
    weights, pca_idx, nisp_idx, thresholds, timings = train_baseline_and_rank(
        x_tr, y_tr, x_val, y_val, num_classes, cfg
    )
    print(f"  ranking time: PCA={timings['pca']:.3f}s NISP={timings['nisp']:.3f}s")

    df_prr = run_prr_sweep(
        x_tr, y_tr, x_val, y_val,
        {"bal": (x_te_bal, y_te_bal), "imb": (x_te_imb, y_te_imb)},
        weights, pca_idx, nisp_idx, cfg, data_type="image",
    )
    df_prr.insert(0, "ratio", ratio)
    df_prr.to_csv(os.path.join(out_dir, f"prr_ratio_{ratio}.csv"), index=False)

    if make_plots:
        for lab in ["bal", "imb"]:
            plot_prr_curve(
                df_prr, f"PRR_F1_{lab}",
                f"PRR (F1-based) on test_{lab} | ratio={ratio}", thresholds,
                save_path=os.path.join(out_dir, f"PRR_{lab}_ratio_{ratio}.png"),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.3, 0.1, 0.02, 0.01, 0.002])
    parser.add_argument("--out", default="results/multi_ratio")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke-test settings (not for paper results).")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        from dataclasses import replace
        cfg = replace(cfg, use_full_scan=False, classifier_epochs=5,
                      siamese_epochs=3, n_runs=1, num_test_sets=2,
                      multiratio_stability_runs=5)

    os.makedirs(args.out, exist_ok=True)
    print("Loading Fashion-MNIST ...")
    x_all, y_all = load_fashion_mnist()
    for ratio in args.ratios:
        run_one_ratio(ratio, x_all, y_all, cfg, args.out, make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
