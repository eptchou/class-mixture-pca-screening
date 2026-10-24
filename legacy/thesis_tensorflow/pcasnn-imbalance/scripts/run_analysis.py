#!/usr/bin/env python
"""Statistical analysis of multi-ratio stability results.

Reads the per-run ``stability_ratio_*.csv`` files written by
``run_multi_ratio.py`` and produces:

1. a Kruskal--Wallis table (H, p, epsilon-squared) testing whether each
   stability metric differs across imbalance ratios, with Dunn's post hoc
   (Holm correction) saved for any significant metric;
2. optionally, the PCA-vs-NISP chi-square independence / overlap table,
   which requires retraining (pass ``--independence`` to compute it).

Example
-------
    python scripts/run_analysis.py --in results/multi_ratio \
        --out results/multi_ratio/analysis
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from pcasnn.experiments import kruskal_with_posthoc


METRICS = [
    "jaccard_Elbow Point", "jaccard_Var 85%", "jaccard_Var 90%",
    "jaccard_Var 95%", "jaccard_Var 99%", "spearman_rho",
]


def load_stability(in_dir: str) -> dict:
    """Return {ratio: DataFrame} from stability_ratio_*.csv files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(in_dir, "stability_ratio_*.csv"))):
        m = re.search(r"stability_ratio_(.+)\.csv$", os.path.basename(path))
        if m:
            out[float(m.group(1))] = pd.read_csv(path)
    if not out:
        raise FileNotFoundError(f"No stability_ratio_*.csv found in {in_dir}")
    return out


def run_kruskal(stability: dict, out_dir: str) -> None:
    rows = []
    for metric in METRICS:
        groups = {r: df[metric].values for r, df in stability.items() if metric in df}
        h, p, eps2, dunn = kruskal_with_posthoc(groups, do_posthoc=True)
        rows.append({"Metric": metric, "H": h, "p_value": p, "epsilon_squared": eps2})
        print(f"{metric:22s}  H={h:8.4f}  p={p:.4g}  eps^2={eps2:.4f}"
              + ("  *sig" if p < 0.05 else ""))
        if dunn is not None:
            dunn.to_csv(os.path.join(out_dir, f"dunn_{metric.replace(' ', '_')}.csv"))
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "kruskal_wallis.csv"), index=False)
    print(f"\nSaved Kruskal--Wallis table to {out_dir}/kruskal_wallis.csv")


def run_independence(out_dir: str, ratios, stability_runs: int) -> None:
    """Retrain per ratio and test PCA-vs-NISP selection independence."""
    from dataclasses import replace

    from pcasnn import ExperimentConfig, set_global_seed
    from pcasnn.datasets import load_fashion_mnist
    from pcasnn.data import split_and_imbalance
    from pcasnn.experiments import train_baseline_and_rank, pca_vs_nisp_independence

    cfg = replace(ExperimentConfig(), multiratio_stability_runs=stability_runs)
    x_all, y_all = load_fashion_mnist()
    num_classes = len(np.unique(y_all))
    thresholds_order = ["Elbow Point", "Var 85%", "Var 90%", "Var 95%", "Var 99%"]

    summary = []
    for ratio in ratios:
        print(f"\nIndependence test, ratio={ratio}")
        agg = {t: {"overlap": [], "sig": 0, "k": []} for t in thresholds_order}
        for r in range(cfg.multiratio_stability_runs):
            set_global_seed(1000 + r)
            (x_tr, y_tr), (x_val, y_val), _, _ = split_and_imbalance(
                x_all, y_all, ratio, cfg.test_size, cfg.val_size_in_trainval,
                cfg.minority_class_label, cfg.min_keep_per_split, seed_base=1000 + r,
            )
            _, pca_idx, nisp_idx, thr, _ = train_baseline_and_rank(
                x_tr, y_tr, x_val, y_val, num_classes, cfg
            )
            for name in thresholds_order:
                k = thr[name]
                p_val, overlap = pca_vs_nisp_independence(
                    pca_idx, nisp_idx, k, cfg.first_layer_neurons
                )
                agg[name]["overlap"].append(overlap)
                agg[name]["k"].append(k)
                if not np.isnan(p_val) and p_val < 0.05:
                    agg[name]["sig"] += 1
        for name in thresholds_order:
            n = cfg.multiratio_stability_runs
            summary.append({
                "ratio": ratio, "threshold": name,
                "avg_overlap": np.mean(agg[name]["overlap"]),
                "avg_neurons": np.mean(agg[name]["k"]),
                "sig_count": f"{agg[name]['sig']}/{n}",
                "sig_rate_%": 100 * agg[name]["sig"] / n,
            })
    pd.DataFrame(summary).to_csv(
        os.path.join(out_dir, "pca_vs_nisp_independence.csv"), index=False)
    print(f"\nSaved independence table to {out_dir}/pca_vs_nisp_independence.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="in_dir", default="results/multi_ratio")
    parser.add_argument("--out", dest="out_dir", default="results/multi_ratio/analysis")
    parser.add_argument("--independence", action="store_true",
                        help="Also compute PCA-vs-NISP independence (retrains).")
    parser.add_argument("--ratios", nargs="+", type=float,
                        default=[0.3, 0.1, 0.02, 0.01, 0.002])
    parser.add_argument("--independence-runs", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stability = load_stability(args.in_dir)
    run_kruskal(stability, args.out_dir)
    if args.independence:
        run_independence(args.out_dir, args.ratios, args.independence_runs)


if __name__ == "__main__":
    main()
