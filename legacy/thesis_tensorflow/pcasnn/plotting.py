"""Plotting helpers for performance-retention (PRR) curves.

Kept separate from the experiment logic so that figures can be regenerated
from saved result tables without rerunning any training.
"""

from __future__ import annotations

from typing import Dict, Optional

import matplotlib.pyplot as plt
import pandas as pd

METHOD_PALETTE = {"PCA-based": "#e11d48", "NISP": "#0284c7", "Random": "gray"}
METHOD_MARKERS = {"PCA-based": "o", "NISP": "s", "Random": "^"}
THRESHOLD_STYLE = {
    "Elbow Point": {"color": "#8b5cf6", "linestyle": "--"},
    "Var 85%": {"color": "#10b981", "linestyle": "-."},
    "Var 90%": {"color": "#f59e0b", "linestyle": "-."},
    "Var 95%": {"color": "#d946ef", "linestyle": "-."},
    "Var 99%": {"color": "#06b6d4", "linestyle": "-."},
}


def plot_prr_curve(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    thresholds: Optional[Dict[str, int]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
):
    """Plot PRR versus K for the three screening strategies.

    Parameters
    ----------
    df : DataFrame
        Must contain columns ``Method``, ``K`` and ``value_col``.
    value_col : str
        The PRR column to plot (e.g. ``"PRR_F1_imb"``).
    thresholds : dict, optional
        Threshold-name -> K, drawn as labelled vertical lines.
    """
    plt.figure(figsize=(10, 6))
    for method, sub in df.groupby("Method"):
        plt.plot(sub["K"], sub[value_col], label=method,
                 color=METHOD_PALETTE.get(method), marker=METHOD_MARKERS.get(method),
                 linewidth=2, markersize=4)
    plt.axhline(y=1.0, color="black", linestyle=":", linewidth=1.5, label="Full model")

    if thresholds:
        for name, k_val in thresholds.items():
            style = THRESHOLD_STYLE.get(name, {"color": "gray", "linestyle": "-."})
            plt.axvline(x=k_val, color=style["color"], linestyle=style["linestyle"],
                        linewidth=1.5, alpha=0.9, label=f"{name} (K={k_val})")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Number of neurons (K)")
    plt.ylabel(value_col.replace("_", " "))
    plt.legend(loc="lower right", ncol=2, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()
