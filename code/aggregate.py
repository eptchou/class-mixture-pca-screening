"""Aggregate results into LaTeX tables, figures and a JSON of key numbers."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
R = os.environ.get("ADAC_RESULTS_DIR", str(Path(__file__).resolve().parent / "results"))
PAPER = os.environ.get("ADAC_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "source_multifile"))
os.makedirs(f"{PAPER}/tables", exist_ok=True); os.makedirs(f"{PAPER}/figures", exist_ok=True)

CRIT = ["PCA", "CB-PCA", "Variance", "NISP", "Random"]
LABEL = {"PCA": "PCA", "CB-PCA": "CB", "Variance": "Variance", "NISP": "NISP", "Random": "Random"}
COL = {"PCA": "#2a78d6", "CB-PCA": "#eb6834", "Variance": "#1baf7a", "NISP": "#eda100", "Random": "#e87ba4"}
MRK = {"PCA": "o", "CB-PCA": "s", "Variance": "^", "NISP": "D", "Random": "v"}
LS = {"PCA": "-", "CB-PCA": "--", "Variance": "-.", "NISP": ":", "Random": (0, (5, 1, 1, 1))}
TAGS = {  # tag: (display, dataset, condition)
    "fmnist_bal": ("Fashion-MNIST", "balanced"), "fmnist_r0.1": ("Fashion-MNIST", r"$\rho=0.1$"),
    "mfeat_bal": ("Multiple Features", "balanced"), "mfeat_r0.1": ("Multiple Features", r"$\rho=0.1$"),
    "reuters_bal": ("Reuters-21578", "balanced"), "reuters_nat": ("Reuters-21578", "natural"),
}
RATIO_TAGS = {0.3: "fmnist_r0.3", 0.1: "fmnist_r0.1", 0.02: "fmnist_r0.02", 0.01: "fmnist_r0.01", 0.002: "fmnist_r0.002"}
LOWK = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
KEY = {}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "legend.fontsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6",
                     "grid.linewidth": 0.5, "font.family": "serif"})


def load_ret(tag):
    fs = sorted(glob.glob(f"{R}/{tag}/retention_seed*.csv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    full = d[d.criterion == "Full"].set_index("seed")
    d = d[d.criterion != "Full"].copy()
    d["prr_f1"] = d.pair_f1 / d.seed.map(full.pair_f1)
    d["prr_acc"] = d.pair_acc / d.seed.map(full.pair_acc)
    return d, full


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ({s:.{nd}f})"


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


# ------------------------------------------------------------------ load
RET = {t: load_ret(t) for t in list(TAGS) + list(RATIO_TAGS.values())}
RET = {t: v for t, v in RET.items() if v is not None}
nseeds = {t: v[1].shape[0] for t, v in RET.items()}
KEY["n_seeds"] = nseeds

# ------------------------------------------------------------------ data table
rows = []
DATA_TAGS = list(TAGS.items()) + [(RATIO_TAGS[r], ("Fashion-MNIST", r"$\rho=" + str(r) + "$")) for r in (0.3, 0.02, 0.01, 0.002)]
for t, (dn, cn) in DATA_TAGS:
    ms = sorted(glob.glob(f"{R}/{t}/meta_seed*.json"))
    if not ms:
        continue
    m = json.load(open(ms[0]))
    ctr = m["class_counts_train"]; cte = m["class_counts_test"]; cva = m["class_counts_val"]
    rows.append((dn, cn, len(ctr), m["n_train"], m["n_val"], m["n_test"], ctr[0], cva[0], cte[0], 100 * ctr[0] / sum(ctr)))
with open(f"{PAPER}/tables/tab_data.tex", "w") as f:
    for i, r in enumerate(rows):
        if i == 6:
            f.write("\\midrule\n")
        prev = f"{r[9]:.2f}" if r[9] < 1 else f"{r[9]:.1f}"
        f.write((f"{r[0]} & {r[1]} & {r[2]} & {r[3]:,} & {r[4]:,} & {r[5]:,} & {r[6]:,} & {r[7]:,} & {r[8]:,} & ".replace(",", "{,}")) + prev + " \\\\\n")
ratio_rows = []
for rho, t in RATIO_TAGS.items():
    ms = sorted(glob.glob(f"{R}/{t}/meta_seed*.json"))
    if ms:
        m = json.load(open(ms[0])); ratio_rows.append((rho, m["class_counts_train"][0], m["class_counts_test"][0], m["n_val"]))
KEY["ratio_minority_counts"] = ratio_rows

# ------------------------------------------------------------------ full-model table
with open(f"{PAPER}/tables/tab_full.tex", "w") as f:
    for t in list(TAGS) + [RATIO_TAGS[r] for r in (0.3, 0.02, 0.01, 0.002)]:
        if t not in RET:
            continue
        full = RET[t][1]
        dn, cn = TAGS[t] if t in TAGS else ("Fashion-MNIST", r"$\rho=" + t.split("_r")[1] + "$")
        cells = [fmt(full[c].mean(), full[c].std(ddof=1)) for c in ["pair_acc", "pair_f1", "min_recall", "bal_acc", "macro_f1"]]
        f.write(f"{dn} & {cn} & " + " & ".join(cells) + " \\\\\n")
        KEY[f"full_{t}"] = {c: [round(full[c].mean(), 4), round(full[c].std(ddof=1), 4)]
                            for c in ["pair_acc", "pair_f1", "min_recall", "bal_acc", "macro_f1"]}

# ------------------------------------------------------------------ low-K summaries + paired tests
METRICS = [("prr_f1", "PRR"), ("min_recall", "Minority recall"), ("bal_acc", "Balanced accuracy")]


def lowk(d, metric):
    sub = d[d.K.isin(LOWK)]
    return sub.groupby(["criterion", "seed"])[metric].mean().unstack(0)


summ = {}
for t in RET:
    d = RET[t][0]
    for mname, _ in METRICS + [("macro_f1", ""), ("pair_f1", "")]:
        L = lowk(d, mname)
        res = {c: (L[c].mean(), L[c].std(ddof=1)) for c in CRIT if c in L}
        tests = {}
        others = [c for c in CRIT if c != "PCA" and c in L]
        ps = []
        for c in others:
            diff = (L["PCA"] - L[c]).values
            n = len(diff); m = diff.mean(); se = diff.std(ddof=1) / np.sqrt(n)
            ci = stats.t.ppf(0.975, n - 1) * se
            p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
            tests[c] = dict(diff=m, lo=m - ci, hi=m + ci, p=p); ps.append(p)
        for c, pa in zip(others, holm(ps)):
            tests[c]["p_holm"] = pa
        summ[(t, mname)] = (res, tests)
        KEY[f"lowk_{t}_{mname}"] = {c: [round(v[0], 4), round(v[1], 4)] for c, v in res.items()}
        KEY[f"test_{t}_{mname}"] = {c: {k: round(float(v), 4) for k, v in tt.items()} for c, tt in tests.items()}


def star(p):
    return "^{\\ast}" if p < 0.05 else ""


with open(f"{PAPER}/tables/tab_lowk.tex", "w") as f:
    for t, (dn, cn) in TAGS.items():
        if (t, "prr_f1") not in summ:
            continue
        for i, c in enumerate(CRIT):
            cells = []
            for mname, _ in METRICS:
                res, tests = summ[(t, mname)]
                m, s = res[c]
                st = "" if c == "PCA" else star(tests[c]["p_holm"])
                cells.append(f"${m:.3f}{st}$ ({s:.3f})")
            lead = f"\\multirow{{5}}{{*}}{{\\shortstack[l]{{{dn}\\\\{cn}}}}}" if i == 0 else ""
            f.write(f"{lead} & {LABEL[c]} & " + " & ".join(cells) + " \\\\\n")
        f.write("\\addlinespace\n")

# ------------------------------------------------------------------ figures: retention curves
def curve(ax, d, metric, full=None, ref1=False):
    g = d.groupby(["criterion", "K"])[metric]
    for c in CRIT:
        m = g.mean().loc[c]; s = g.std(ddof=1).loc[c] / np.sqrt(g.count().loc[c])
        ax.plot(m.index, m.values, color=COL[c], marker=MRK[c], ms=3, lw=1.2, ls=LS[c], label=LABEL[c])
        ax.fill_between(m.index, m - s, m + s, color=COL[c], alpha=0.15, lw=0)
    if metric in ("min_recall", "bal_acc"):
        ax.set_ylim(0, 1.0)
    if ref1:
        ax.axhline(1.0, color="#555555", lw=0.8, ls=(0, (2, 2)))
    if full is not None:
        ax.axhline(full, color="#555555", lw=0.8, ls=(0, (2, 2)))
    ax.set_xscale("log", base=2); ax.set_xticks([1, 2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64"])


# K at which the seed-mean curve first reaches 95% of the unpruned value
for t in RET:
    d, full = RET[t]
    out = {}
    for c in CRIT:
        sub = d[d.criterion == c]
        prr = sub.groupby("K").prr_f1.mean()
        rel_bal = (sub.bal_acc / sub.seed.map(full.bal_acc)).groupby(sub.K).mean()
        k1 = prr[prr >= 0.95].index.min(); k2 = rel_bal[rel_bal >= 0.95].index.min()
        out[c] = dict(K95_prr=int(k1) if pd.notna(k1) else None, K95_balacc=int(k2) if pd.notna(k2) else None)
    KEY[f"k95_{t}"] = out
    # class-level retention at the low-K average: bal_acc relative to unpruned
    sub = d[d.K.isin(LOWK)]
    KEY[f"rel_balacc_lowk_{t}"] = {c: round(float((sub[sub.criterion == c].bal_acc /
                                    sub[sub.criterion == c].seed.map(full.bal_acc)).mean()), 4) for c in CRIT}


def retention_figure(tags, fname, recall_label="Minority-class recall"):
    tags = [t for t in tags if t in RET]
    if not tags:
        return
    fig, axes = plt.subplots(3, len(tags), figsize=(2.35 * len(tags), 6.0), squeeze=False)
    for j, t in enumerate(tags):
        d, full = RET[t]
        dn, cn = TAGS[t]
        curve(axes[0, j], d, "prr_f1", ref1=True)
        curve(axes[1, j], d, "min_recall", full=full.min_recall.mean())
        curve(axes[2, j], d, "bal_acc", full=full.bal_acc.mean())
        axes[0, j].set_title(f"{dn} ({cn.replace('$', '').replace(chr(92)+'rho', 'ρ')})")
        axes[2, j].set_xlabel("Retained units $K$")
    axes[0, 0].set_ylabel("PRR (pair $F_1$)"); axes[1, 0].set_ylabel(recall_label); axes[2, 0].set_ylabel("Balanced accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    fig.savefig(f"{PAPER}/figures/{fname}.pdf"); fig.savefig(f"{PAPER}/figures/{fname}.png", dpi=160)
    plt.close(fig)


retention_figure(["fmnist_r0.1", "mfeat_r0.1", "reuters_nat"], "fig_retention_imb")
retention_figure(["fmnist_bal", "mfeat_bal", "reuters_bal"], "fig_retention_bal", recall_label="Recall of designated class")

# ------------------------------------------------------------------ figure: multi-ratio
avail = [r for r in RATIO_TAGS if RATIO_TAGS[r] in RET]
if len(avail) >= 2:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    dodge = {c: 1.07 ** (i - 2) for i, c in enumerate(CRIT)}
    for ax, (mname, lab) in zip(axes, METRICS):
        for c in CRIT:
            ms = [summ[(RATIO_TAGS[r], mname)][0][c] for r in avail]
            nn = [nseeds[RATIO_TAGS[r]] for r in avail]
            ax.errorbar([r * dodge[c] for r in avail], [m for m, s in ms],
                        yerr=[s / np.sqrt(n) for (m, s), n in zip(ms, nn)], color=COL[c], marker=MRK[c], ms=3.5,
                        lw=1.2, ls=LS[c], capsize=1.5, label=LABEL[c])
        if mname != "prr_f1":
            fl = [RET[RATIO_TAGS[r]][1][mname].mean() for r in avail]
            ax.plot(avail, fl, color="#333333", lw=0.9, ls=(0, (2, 2)), marker="x", ms=3.5, label="Unpruned")
        ax.set_xscale("log"); ax.set_xticks(avail); ax.set_xticklabels([str(r).replace("0.", ".") for r in avail])
        ax.minorticks_off()
        ax.set_xlabel(r"Minority subsampling fraction $\rho$"); ax.set_title(lab)
    axes[0].set_ylabel(r"Mean over $K\leq32$")
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(f"{PAPER}/figures/fig_ratio.pdf"); fig.savefig(f"{PAPER}/figures/fig_ratio.png", dpi=160)
    plt.close(fig)

# ------------------------------------------------------------------ stability tables
def stab_summary(t):
    fs, fn = f"{R}/{t}/stab_split.csv", f"{R}/{t}/stab_nogueira.csv"
    if not (os.path.exists(fs) and os.path.exists(fn)):
        return None
    s = pd.read_csv(fs).groupby("seed").mean(numeric_only=True)
    n = pd.read_csv(fn)
    out = {}
    for c in ["PCA", "CB-PCA", "Variance", "NISP"]:
        out[c] = dict(rho=(s[f"{c}_rho"].mean(), s[f"{c}_rho"].std(ddof=1)),
                      J16=(s[f"{c}_J16"].mean(), s[f"{c}_J16"].std(ddof=1)),
                      J32=(s[f"{c}_J32"].mean(), s[f"{c}_J32"].std(ddof=1)),
                      phi16=(n[(n.criterion == c) & (n.K == 16)].phi.mean(), n[(n.criterion == c) & (n.K == 16)].phi.std(ddof=1)),
                      phi32=(n[(n.criterion == c) & (n.K == 32)].phi.mean(), n[(n.criterion == c) & (n.K == 32)].phi.std(ddof=1)))
    return out, s


with open(f"{PAPER}/tables/tab_stab.tex", "w") as f:
    for t, (dn, cn) in TAGS.items():
        ss = stab_summary(t)
        if ss is None:
            continue
        out, _ = ss
        for i, c in enumerate(["PCA", "CB-PCA", "Variance", "NISP"]):
            o = out[c]
            lead = f"\\multirow{{4}}{{*}}{{\\shortstack[l]{{{dn}\\\\{cn}}}}}" if i == 0 else ""
            cells = [fmt(*o[k]) for k in ["rho", "J16", "J32", "phi16", "phi32"]]
            f.write(f"{lead} & {LABEL[c]} & " + " & ".join(cells) + " \\\\\n")
            KEY[f"stab_{t}_{c}"] = {k: [round(v[0], 4), round(v[1], 4)] for k, v in o.items()}
        f.write("\\addlinespace\n")

# multi-ratio stability of the PCA ranking: thesis-style thresholds + chance across ratios
thr_names = ["Elbow", "Var85", "Var90", "Var95", "Var99"]
per_ratio = {}
for r in avail:
    ss = stab_summary(RATIO_TAGS[r])
    if ss is not None:
        per_ratio[r] = ss[1]
if len(per_ratio) >= 2:
    with open(f"{PAPER}/tables/tab_ratio_stab.tex", "w") as f:
        for r, s in per_ratio.items():
            cells = []
            for tn in thr_names:
                cells.append(f"{s[f'PCA_thrJ_{tn}'].mean():.3f} [{s[f'PCA_thrChance_{tn}'].mean():.2f}]")
            cells.append(f"{s['PCA_rho'].mean():.4f}")
            cells.append(f"{s['CB-PCA_rho'].mean():.4f}")
            f.write(f"{r} & " + " & ".join(cells) + " \\\\\n")
        # Blocked ratio tests are computed in analysis_extra.py; do not use
        # an independent-samples Kruskal test on the shared seed blocks.
        cols = [f"PCA_thrJ_{tn}" for tn in thr_names] + ["PCA_rho", "CB-PCA_rho"]
        KEY["ratio_stab_means"] = {str(r): {c: round(s[c].mean(), 4) for c in cols} for r, s in per_ratio.items()}

# ------------------------------------------------------------------ PCA-NISP overlap and score correlations
ov = []
ALLTAGS = list(dict.fromkeys(list(TAGS) + list(RATIO_TAGS.values())))
for t in ALLTAGS:
    fo = f"{R}/{t}/overlap_pca_nisp.csv"
    if os.path.exists(fo):
        o = pd.read_csv(fo); o["tag"] = t; ov.append(o)
if ov:
    ov = pd.concat(ov)
    g = ov.groupby(["tag", "threshold"]).agg(K=("K", "mean"), overlap=("overlap", "mean"), expected=("expected", "mean"),
                                             kappa=("kappa", "mean"), frac_sig=("p_hyper", lambda x: (x < 0.05).mean()))
    g.round(3).to_csv(f"{R}/overlap_summary.csv")
    cd = ov[ov.tag.isin(list(TAGS))]
    KEY['overlap_crossdomain'] = dict(kappa_mean=round(cd.kappa.mean(), 3), frac_sig=round((cd.p_hyper < 0.05).mean(), 3), n=int(len(cd)), excess=round((cd.overlap - cd.expected).mean(), 2))
    KEY["overlap_all"] = dict(kappa_mean=round(ov.kappa.mean(), 3), kappa_max=round(ov.kappa.max(), 3),
                              frac_sig=round((ov.p_hyper < 0.05).mean(), 3), n=int(len(ov)),
                              ratio_overlap_minus_expected=round((ov.overlap - ov.expected).mean(), 2))
cr = []
for t in ALLTAGS:
    fc = f"{R}/{t}/score_correlations.csv"
    if os.path.exists(fc):
        c = pd.read_csv(fc); c["tag"] = t; cr.append(c)
if cr:
    cr = pd.concat(cr)
    KEY["score_corr"] = cr.groupby("tag")[["rho_PCA_Variance", "rho_PCA_CB-PCA", "rho_PCA_NISP"]].mean().round(3).to_dict("index")
    KEY["score_corr_range"] = {c: [round(cr[c].min(), 3), round(cr[c].max(), 3)] for c in ["rho_PCA_Variance", "rho_PCA_CB-PCA", "rho_PCA_NISP"]}

# ------------------------------------------------------------------ mixture shift
fm = f"{R}/fmnist_bal/mixture_shift.csv"
if os.path.exists(fm):
    mx = pd.read_csv(fm)
    G = mx.groupby("ratio").agg(["mean", "std"])
    KEY["mixture"] = {str(r): {c: round(G.loc[r, (c, "mean")], 4) for c in
                               ["dpi_l1", "dSigma_op", "bound_3B", "dscore_max", "rho", "J_16", "J_32", "J_K85", "noise_split_median"]}
                      for r in G.index}
    KEY["mixture_guaranteed_any"] = bool(mx.filter(like="guaranteed").values.any())
    ref = stab_summary("fmnist_bal")
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.3))
    rs = sorted(mx.ratio.unique())
    ax = axes[0]
    ax.errorbar(rs, G.loc[rs, ("rho", "mean")], yerr=G.loc[rs, ("rho", "std")], color=COL["PCA"], marker="o", ms=3.5, capsize=2, label="Mixture shift")
    if ref:
        ax.axhline(ref[0]["PCA"]["rho"][0], color="#555555", ls=(0, (2, 2)), lw=0.9, label="Split-half (sampling)")
    ax.set_xscale("log"); ax.set_xticks(rs); ax.set_xticklabels([str(r) for r in rs]); ax.set_xlabel(r"Minority subsampling fraction $\rho$")
    ax.set_title("Spearman correlation"); ax.set_ylim(0.87, 1.0); ax.legend(frameon=False, loc="lower right")
    ax = axes[1]
    for key_, mk, ls_, lab, refk in [("J_16", "o", "-", "$K=16$", "J16"), ("J_32", "s", "--", "$K=32$", "J32")]:
        ax.errorbar(rs, G.loc[rs, (key_, "mean")], yerr=G.loc[rs, (key_, "std")], marker=mk, ms=3.5, capsize=2,
                    color=COL["PCA"], ls=ls_, label=lab)
        if ref:
            yv = ref[0]["PCA"][refk][0]
            ax.axhline(yv, color="#555555", ls=(0, (2, 2)), lw=0.8)
            ax.text(0.40, yv + 0.006, lab, fontsize=6.5, color="#555555", va="bottom", ha="left")
    ax.set_xscale("log"); ax.set_xticks(rs); ax.set_xticklabels([str(r) for r in rs]); ax.set_xlabel(r"Minority subsampling fraction $\rho$")
    ax.set_title("Jaccard index of top-$K$ sets"); ax.set_ylim(0.45, 0.92); ax.set_xlim(0.0015, 1.1)
    ax.plot([], [], color="#555555", ls=(0, (2, 2)), lw=0.8, label="Split-half")
    ax.legend(frameon=False, loc="lower right", ncol=3, columnspacing=1.0, handlelength=1.8)
    fig.tight_layout()
    fig.savefig(f"{PAPER}/figures/fig_mixture.pdf"); fig.savefig(f"{PAPER}/figures/fig_mixture.png", dpi=160)
    plt.close(fig)

# ------------------------------------------------------------------ timing / epochs
tm = f"{R}/fmnist_r0.1/timing.csv"
if os.path.exists(tm):
    KEY["timing_median_sec"] = pd.read_csv(tm).drop(columns="rep").median().round(3).to_dict()
ep = pd.concat([RET[t][0].assign(tag=t) for t in RET])
KEY["epochs_frac_early_stop"] = round((ep.epochs < 10).mean(), 3)

json.dump(KEY, open(f"{R}/key_numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: KEY[k] for k in list(KEY)[:6]}, indent=1, default=float))
print("tags:", nseeds)
