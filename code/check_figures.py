"""Independent check of every plotted value against raw result files.

Recomputes the plotted quantities from the raw CSVs with fresh code, then
extracts the data actually drawn by aggregate.py's plotting functions and
compares point by point. Also re-derives the mixture-shift numbers for one
network with an alternative formula (observation-weighted covariance).
"""
import glob, json, sys, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from pathlib import Path
R = os.environ.get("ADAC_RESULTS_DIR", str(Path(__file__).resolve().parent / "results"))
CRIT = ["PCA", "CB-PCA", "Variance", "NISP", "Random"]
LOWK = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
fails = []


def raw(tag):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/{tag}/retention_seed*.csv"))])
    full = d[d.criterion == "Full"][["seed", "pair_f1", "min_recall", "bal_acc"]].rename(
        columns=lambda c: c if c == "seed" else "full_" + c)
    d = d[d.criterion != "Full"].merge(full, on="seed")
    d["prr"] = d.pair_f1 / d.full_pair_f1
    return d, full


def close(a, b, what, tol=1e-9):
    if not np.allclose(a, b, atol=tol, rtol=0):
        fails.append(f"{what}: max diff {np.max(np.abs(np.asarray(a) - np.asarray(b))):.3g}")


# ---------------------------------------------------------------- retention curves
sys.argv = ["aggregate.py"]
import aggregate as A   # executes the pipeline; we reuse only its plotting function
plt.close("all")
for tag in ["fmnist_r0.1", "mfeat_r0.1", "reuters_nat", "fmnist_bal", "mfeat_bal", "reuters_bal"]:
    d, full = raw(tag)
    assert d.seed.nunique() == 10 and len(d) == 10 * 15 * 5, tag
    for metric_raw, metric_plot in [("prr", "prr_f1"), ("min_recall", "min_recall"), ("bal_acc", "bal_acc")]:
        fig, ax = plt.subplots()
        A.curve(ax, A.RET[tag][0], metric_plot)
        INV = {v: k for k, v in A.LABEL.items()}
        lines = [l for l in ax.get_lines() if l.get_label() in INV]
        polys = ax.collections
        for line, poly in zip(lines, polys):
            c = INV[line.get_label()]
            sub = d[d.criterion == c].groupby("K")[metric_raw]
            m, se = sub.mean(), sub.std(ddof=1) / np.sqrt(sub.count())
            close(line.get_xdata(), m.index.values, f"{tag} {metric_raw} {c} x")
            close(line.get_ydata(), m.values, f"{tag} {metric_raw} {c} mean")
            v = poly.get_paths()[0].vertices
            for k, mu, s in zip(m.index, m.values, se.values):
                ys = v[np.isclose(v[:, 0], k), 1]
                close([ys.max(), ys.min()], [mu + s, mu - s], f"{tag} {metric_raw} {c} band K={k}")
        plt.close(fig)
    # unpruned reference lines
    for col in ["min_recall", "bal_acc"]:
        close(A.RET[tag][1][col].mean(), full["full_" + col].mean(), f"{tag} unpruned {col}")

# ---------------------------------------------------------------- multi-ratio figure
for rho, tag in A.RATIO_TAGS.items():
    d, full = raw(tag)
    low = d[d.K.isin(LOWK)].groupby(["criterion", "seed"])[["prr", "min_recall", "bal_acc"]].mean()
    for metric_raw, metric_plot in [("prr", "prr_f1"), ("min_recall", "min_recall"), ("bal_acc", "bal_acc")]:
        for c in CRIT:
            v = low.loc[c][metric_raw]
            m, s = A.summ[(tag, metric_plot)][0][c]
            close([m, s], [v.mean(), v.std(ddof=1)], f"ratio {rho} {metric_raw} {c}")

# ---------------------------------------------------------------- mixture figure: re-derive one network
import jax.numpy as jnp, engine as E
mx = pd.read_csv(f"{R}/fmnist_bal/mixture_shift.csv")
seed = 3
z = np.load(f"{R}/fmnist_bal/model_seed{seed}.npz")
p = {k: jnp.asarray(z[k]) for k in ["W1", "b1", "g", "be", "W2", "b2", "W3", "b3"]}
s = dict(mean=jnp.asarray(z["bn_mean"]), var=jnp.asarray(z["bn_var"]))
D = E.load_condition("fmnist", 1.0, seed)
Ab, _ = E.activations(p, s, D["Xva_bal"]); yb = D["yva_bal"]
counts = np.bincount(yb)


def wcov(pi):   # observation-weighted covariance, weights pi_c / n_c (alternative formula)
    w = (pi / counts)[yb]; w = w / w.sum()
    mu = w @ Ab
    X = Ab - mu
    return (X * w[:, None]).T @ X


def score(S):
    wv, V = np.linalg.eigh(S); wv, V = np.clip(wv[::-1], 0, None), V[:, ::-1]
    return (np.abs(V) * (wv / wv.sum())[None, :]).sum(1)


pi0 = np.full(10, 0.1); s0 = score(wcov(pi0))
for rho in [0.3, 0.1, 0.02, 0.01, 0.002]:
    pi = np.ones(10); pi[0] = rho; pi /= pi.sum()
    st = score(wcov(pi))
    row = mx[(mx.seed == seed) & np.isclose(mx.ratio, rho)].iloc[0]
    o0, ot = np.argsort(-s0, kind="stable"), np.argsort(-st, kind="stable")
    j16 = len(set(o0[:16]) & set(ot[:16])) / len(set(o0[:16]) | set(ot[:16]))
    close([spearmanr(st, s0)[0], np.abs(st - s0).max(), j16],
          [row.rho, row.dscore_max, row.J_16], f"mixture seed {seed} rho {rho}", tol=1e-6)
G = mx.groupby("ratio")[["rho", "J_16", "J_32"]].agg(["mean", "std"])
k = json.load(open(f"{R}/key_numbers.json"))
for rho in G.index:
    close(G.loc[rho, ("rho", "mean")], k["mixture"][str(rho)]["rho"], f"mixture mean rho {rho}", tol=1e-4)
ref = pd.read_csv(f"{R}/fmnist_bal/stab_split.csv").groupby("seed").mean(numeric_only=True)
print("split-half reference lines: rho %.4f  J16 %.4f  J32 %.4f" % (ref.PCA_rho.mean(), ref.PCA_J16.mean(), ref.PCA_J32.mean()))

print("FAILURES:" if fails else "ALL CHECKS PASSED", len(fails))
for f in fails[:20]:
    print("  ", f)
