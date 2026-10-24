#!/usr/bin/env python
"""Cross-domain validation across the six benchmark data sets.

For each data set this runs, under both a balanced control and an
imbalanced regime (minority class kept at ``--ratio``):

1. split-half PCA selection stability (Jaccard + Spearman),
2. a PRR sweep over K comparing PCA-based, NISP and random screening,

and saves result tables (and PRR figures) under ``--out``.

Example
-------
    python scripts/run_cross_domain.py --datasets Fashion-MNIST ISOLET \
        --ratio 0.1 --out results/cross_domain
"""

from __future__ import annotations

import argparse
import os

import pandas as pd

from pcasnn import ExperimentConfig, set_global_seed, split_and_imbalance
from pcasnn.datasets import REGISTRY
from pcasnn.experiments import (
    run_stability_analysis,
    train_baseline_and_rank,
    run_prr_sweep,
)
from pcasnn.plotting import plot_prr_curve


def run_one(name: str, x_all, y_all, data_type: str, ratio: float,
            cfg: ExperimentConfig, out_dir: str, make_plots: bool) -> None:
    import numpy as np

    num_classes = len(np.unique(y_all))
    ds_dir = os.path.join(out_dir, name)
    os.makedirs(ds_dir, exist_ok=True)

    for mode, r in [("Balanced", 1.0), (f"Imbalanced_{ratio}", ratio)]:
        print(f"\n=== {name} | {mode} (ratio={r}) ===")
        set_global_seed(cfg.base_seed)
        (x_tr, y_tr), (x_val, y_val), (x_te_bal, y_te_bal), (x_te_imb, y_te_imb) = \
            split_and_imbalance(
                x_all, y_all, r, cfg.test_size, cfg.val_size_in_trainval,
                cfg.minority_class_label, cfg.min_keep_per_split, cfg.base_seed,
            )

        # 1. Stability
        stab_summary, _ = run_stability_analysis(
            x_tr, y_tr, x_val, y_val, num_classes, cfg
        )
        stab_summary.to_csv(
            os.path.join(ds_dir, f"{name}_{mode}_stability.csv"), index=False
        )
        print(stab_summary.to_string(index=False, float_format="%.4f"))

        # 2. Baseline + ranking + PRR sweep
        weights, pca_idx, nisp_idx, thresholds, timings = train_baseline_and_rank(
            x_tr, y_tr, x_val, y_val, num_classes, cfg
        )
        # For the balanced control both test sets coincide.
        test_sets = {"imb": (x_te_imb, y_te_imb)} if r < 1.0 else {"bal": (x_te_bal, y_te_bal)}
        df = run_prr_sweep(
            x_tr, y_tr, x_val, y_val, test_sets, weights, pca_idx, nisp_idx,
            cfg, data_type=data_type,
        )
        df.to_csv(os.path.join(ds_dir, f"{name}_{mode}_prr.csv"), index=False)
        print(f"  ranking time: PCA={timings['pca']:.3f}s NISP={timings['nisp']:.3f}s")

        if make_plots:
            lab = "imb" if r < 1.0 else "bal"
            for metric in ["F1", "ACC"]:
                plot_prr_curve(
                    df, f"PRR_{metric}_{lab}",
                    f"PRR ({metric}) - {name} [{mode}]", thresholds,
                    save_path=os.path.join(ds_dir, f"{name}_{mode}_PRR_{metric}.png"),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="+", default=list(REGISTRY.keys()),
                        choices=list(REGISTRY.keys()),
                        help="Which benchmarks to run (default: all six).")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Minority keep-ratio for the imbalanced regime.")
    parser.add_argument("--out", default="results/cross_domain",
                        help="Output directory.")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip figure generation.")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke-test settings (not for paper results).")
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.quick:
        from dataclasses import replace
        cfg = replace(cfg, use_full_scan=False, classifier_epochs=5,
                      siamese_epochs=3, n_runs=1, num_test_sets=2, stability_runs=2)

    os.makedirs(args.out, exist_ok=True)
    for name in args.datasets:
        loader, data_type = REGISTRY[name]
        print(f"\nLoading {name} ...")
        x_all, y_all = loader()
        run_one(name, x_all, y_all, data_type, args.ratio, cfg, args.out,
                make_plots=not args.no_plots)


if __name__ == "__main__":
    main()
