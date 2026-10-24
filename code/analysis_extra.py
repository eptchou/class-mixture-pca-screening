"""Additional analyses requested in review:
 (1) Friedman tests across imbalance ratios (seeds shared across ratios -> blocked design)
 (2) sensitivity of the low-K summary to the K range
 (3) exact Holm-adjusted p-values for all paired comparisons (K <= 32)
 (4) minority prevalence implied by rho; CB vs PCA agreement on balanced data
"""
import glob, json, os
import numpy as np, pandas as pd
from scipy import stats

from pathlib import Path
R = os.environ.get("ADAC_RESULTS_DIR", str(Path(__file__).resolve().parent / "results"))
P = str(Path(os.environ.get("ADAC_PAPER_DIR", str(Path(__file__).resolve().parents[1] / "source_multifile"))) / "tables")
Path(P).mkdir(parents=True, exist_ok=True)
CRIT = ["PCA", "CB-PCA", "Variance", "NISP", "Random"]
TAGS = ["fmnist_bal", "fmnist_r0.1", "mfeat_bal", "mfeat_r0.1", "reuters_bal", "reuters_nat",
        "fmnist_r0.3", "fmnist_r0.02", "fmnist_r0.01", "fmnist_r0.002"]
NAME = {"fmnist_bal": "F-MNIST, balanced", "fmnist_r0.1": r"F-MNIST, $\rho=0.1$", "mfeat_bal": "MF, balanced",
        "mfeat_r0.1": r"MF, $\rho=0.1$", "reuters_bal": "Reuters, balanced", "reuters_nat": "Reuters, natural",
        "fmnist_r0.3": r"F-MNIST, $\rho=0.3$", "fmnist_r0.02": r"F-MNIST, $\rho=0.02$",
        "fmnist_r0.01": r"F-MNIST, $\rho=0.01$", "fmnist_r0.002": r"F-MNIST, $\rho=0.002$"}
RATIO_TAGS = {0.3: "fmnist_r0.3", 0.1: "fmnist_r0.1", 0.02: "fmnist_r0.02", 0.01: "fmnist_r0.01", 0.002: "fmnist_r0.002"}
KEY = {}
LAB = {"PCA": "PCA", "CB-PCA": "CB", "Variance": "Variance", "NISP": "NISP", "Random": "Random"}


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def load(tag):
    d = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{R}/{tag}/retention_seed*.csv"))])
    full = d[d.criterion == "Full"].set_index("seed")
    d = d[d.criterion != "Full"].copy()
    d["prr_f1"] = d.pair_f1 / d.seed.map(full.pair_f1)
    return d


DATA = {t: load(t) for t in TAGS}


def paired(d, metric, Ks):
    L = d[d.K.isin(Ks)].groupby(["criterion", "seed"])[metric].mean().unstack(0)
    res = {}
    others = ["CB-PCA", "Variance", "NISP", "Random"]
    ps = []
    for c in others:
        diff = (L[c] - L["PCA"]).values          # other minus PCA
        p = stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
        res[c] = dict(diff=diff.mean(), p=p); ps.append(p)
    for c, pa in zip(others, holm(ps)):
        res[c]["p_holm"] = pa
    return res


# ---------------------------------------------------------------- (1) Friedman across ratios
rows = []
cols = [f"PCA_thrJ_{t}" for t in ["Elbow", "Var85", "Var90", "Var95", "Var99"]] + ["PCA_rho", "CB-PCA_rho"]
M = {r: pd.read_csv(f"{R}/{t}/stab_split.csv").groupby("seed").mean(numeric_only=True) for r, t in RATIO_TAGS.items()}
fr = {}
ps = []
for c in cols:
    X = np.column_stack([M[r][c].sort_index().values for r in RATIO_TAGS])   # seeds x ratios
    chi2, p = stats.friedmanchisquare(*X.T)
    n, k = X.shape; W = chi2 / (n * (k - 1))                               # Kendall's W
    fr[c] = dict(chi2=chi2, p=p, W=W); ps.append(p)
for c, pa in zip(cols, holm(ps)):
    fr[c]["p_holm"] = pa
KEY["friedman_ratio"] = {c: {k: round(float(v), 4) for k, v in d.items()} for c, d in fr.items()}
# also: are seeds shared across ratios? report correlation of seed-level PCA_rho across ratios
X = np.column_stack([M[r]["CB-PCA_rho"].sort_index().values for r in RATIO_TAGS])
with open(f"{P}/tab_ratio_friedman.tex", "w") as f:
    f.write("Friedman $\\chi^2_4$ & " + " & ".join(f"{fr[c]['chi2']:.2f}" for c in cols) + " \\\\\n")
    f.write("Holm $p$ & " + " & ".join(f"{fr[c]['p_holm']:.3f}" if fr[c]['p_holm'] >= 0.001 else "$<$0.001" for c in cols) + " \\\\\n")
    f.write("Kendall's $W$ & " + " & ".join(f"{fr[c]['W']:.2f}" for c in cols) + " \\\\\n")

# ---------------------------------------------------------------- (2) K-range sensitivity
RANGES = {"K<=16": [1, 2, 3, 4, 6, 8, 10, 12, 16], "K<=32": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32],
          "K<=64": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64],
          "all K": [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96]}
sens = {}
lines = []
for t in TAGS:
    cells = []
    for rn, Ks in RANGES.items():
        r1 = paired(DATA[t], "min_recall", Ks)["CB-PCA"]
        r2 = paired(DATA[t], "prr_f1", Ks)["Random"]
        sens[(t, rn)] = dict(cb_minrec=r1, rand_prr=r2)
        cells.append(r1)
    KEY.setdefault("sensitivity", {})[t] = {rn: dict(cb_minus_pca_minrec=round(sens[(t, rn)]["cb_minrec"]["diff"], 4),
                                                     p_holm=round(sens[(t, rn)]["cb_minrec"]["p_holm"], 4),
                                                     random_minus_pca_prr=round(sens[(t, rn)]["rand_prr"]["diff"], 4),
                                                     p_holm_prr=round(sens[(t, rn)]["rand_prr"]["p_holm"], 4))
                                            for rn in RANGES}


def sgn(x):
    return "0.000" if abs(x) < 0.0005 else f"{x:+.3f}"


def cell(r):
    st = "^{\\ast}" if r["p_holm"] < 0.05 else ""
    return f"${sgn(r['diff'])}{st}$ ({r['p_holm']:.3f})"


with open(f"{P}/tab_sensitivity.tex", "w") as f:
    for t in TAGS:
        f.write(NAME[t] + " & " + " & ".join(cell(sens[(t, rn)]["cb_minrec"]) for rn in RANGES) + " & "
                + " & ".join(cell(sens[(t, rn)]["rand_prr"]) for rn in ["K<=16", "K<=32", "K<=64"]) + " \\\\\n")

# ---------------------------------------------------------------- (3) exact p-values, all comparisons (K <= 32)
with open(f"{P}/tab_pvalues.tex", "w") as f:
    for t in TAGS:
        res = {m: paired(DATA[t], m, RANGES["K<=32"]) for m in ["prr_f1", "min_recall", "bal_acc"]}
        for i, c in enumerate(["CB-PCA", "Variance", "NISP", "Random"]):
            lead = f"\\multirow{{4}}{{*}}{{{NAME[t]}}}" if i == 0 else ""
            cs = [f"${sgn(res[m][c]['diff'])}$ & {res[m][c]['p_holm']:.3f}" for m in ["prr_f1", "min_recall", "bal_acc"]]
            f.write(f"{lead} & {LAB[c]} & " + " & ".join(cs) + " \\\\\n")
        if t != TAGS[-1]:
            f.write("\\addlinespace\n")

# ---------------------------------------------------------------- (4) prevalence and CB/PCA agreement on balanced data
prev = {}
for t in TAGS:
    m = json.load(open(sorted(glob.glob(f"{R}/{t}/meta_seed*.json"))[0]))
    c = m["class_counts_train"]; prev[t] = round(c[0] / sum(c), 5)
KEY["minority_prevalence_train"] = prev
agree = {}
for t in ["fmnist_bal", "mfeat_bal", "reuters_bal"]:
    cr = pd.read_csv(f"{R}/{t}/score_correlations.csv")
    d = DATA[t]; L = d[d.K.isin(RANGES["K<=32"])].groupby(["criterion", "seed"])[["min_recall", "bal_acc", "prr_f1"]].mean()
    diff = (L.loc["CB-PCA"] - L.loc["PCA"]).abs().max()
    same = (d[d.criterion == "PCA"].set_index(["seed", "K"]).min_recall == d[d.criterion == "CB-PCA"].set_index(["seed", "K"]).min_recall).mean()
    agree[t] = dict(min_spearman=round(cr["rho_PCA_CB-PCA"].min(), 5), max_abs_diff_lowk=diff.round(4).to_dict(),
                    frac_identical_fits=round(float(same), 3))
KEY["cb_pca_balanced_agreement"] = agree
# ---------------------------------------------------------------- (5) logistic-regression probe (robustness)
PROBE_TAGS = ["fmnist_r0.1", "mfeat_r0.1", "reuters_nat"]
if all(len(glob.glob(f"{R}/{t}/probe_seed*.csv")) == 10 for t in PROBE_TAGS):
    KEY["probe"] = {}
    with open(f"{P}/tab_probe.tex", "w") as f:
        for t in PROBE_TAGS:
            pr = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f"{R}/{t}/probe_seed*.csv"))])
            full = pr[pr.criterion == "Full"].set_index("seed")
            pr = pr[pr.criterion != "Full"]
            proto = DATA[t]
            res = {m: paired(pr, m, RANGES["K<=32"]) for m in ["probe_min_recall", "probe_bal_acc"]}
            lev = pr[pr.K.isin(RANGES["K<=32"])].groupby(["criterion", "seed"])[["probe_min_recall", "probe_bal_acc"]].mean().groupby("criterion").mean()
            plev = proto[proto.K.isin(RANGES["K<=32"])].groupby(["criterion", "seed"])[["min_recall", "bal_acc"]].mean().groupby("criterion").mean()
            KEY["probe"][t] = dict(full=dict(min_recall=round(full.probe_min_recall.mean(), 4), bal_acc=round(full.probe_bal_acc.mean(), 4)),
                                   lowk={c: dict(min_recall=round(lev.loc[c, "probe_min_recall"], 4), bal_acc=round(lev.loc[c, "probe_bal_acc"], 4),
                                                 proto_min_recall=round(plev.loc[c, "min_recall"], 4), proto_bal_acc=round(plev.loc[c, "bal_acc"], 4))
                                         for c in CRIT},
                                   tests={m: {c: {k: round(float(v), 4) for k, v in r.items()} for c, r in res[m].items()} for m in res})
            for i, c in enumerate(CRIT):
                lead = f"\\multirow{{5}}{{*}}{{{NAME[t]}}}" if i == 0 else ""
                if c == "PCA":
                    cs = [f"{lev.loc[c, 'probe_min_recall']:.3f} & ---", f"{lev.loc[c, 'probe_bal_acc']:.3f} & ---"]
                else:
                    cs = [f"{lev.loc[c, m]:.3f} & ${sgn(res[m][c]['diff'])}$ ({res[m][c]['p_holm']:.3f})" for m in ["probe_min_recall", "probe_bal_acc"]]
                f.write(f"{lead} & {LAB[c]} & " + " & ".join(cs) + " \\\\\n")
            f.write(f" & Unpruned & {full.probe_min_recall.mean():.3f} & & {full.probe_bal_acc.mean():.3f} & \\\\\n")
            if t != PROBE_TAGS[-1]:
                f.write("\\addlinespace\n")
    print(json.dumps(KEY["probe"], indent=0))
# ---------------------------------------------------------------- (6) stratified vs random split halves
STRAT = TAGS
if all(os.path.exists(f"{R}/{t}/stab_stratified.csv") for t in STRAT):
    values = {}
    for t in STRAT:
        d = pd.read_csv(f"{R}/{t}/stab_stratified.csv").groupby("seed").mean(numeric_only=True)
        dp = d.PCA_stratified - d.PCA_random
        p = stats.wilcoxon(dp).pvalue if np.any(dp != 0) else 1.0
        values[t] = dict(PCA_random=float(d.PCA_random.mean()),
                         PCA_stratified=float(d.PCA_stratified.mean()), p_PCA=float(p))
    for t, p in zip(STRAT, holm([values[t]["p_PCA"] for t in STRAT])):
        values[t]["p_holm"] = float(p)
    KEY["stratified"] = values
    with open(f"{P}/tab_stratified.tex", "w") as f:
        for t in STRAT:
            v = values[t]
            f.write(NAME[t] + " & " + " & ".join(f"{v[k]:.3f}" for k in
                    ["PCA_random", "PCA_stratified", "p_holm"]) + r" \\" + "\n")
# ---------------------------------------------------------------- (7) spectral conditions of Propositions 1-2
if os.path.exists(f"{R}/spectral_check.csv"):
    sp = pd.read_csv(f"{R}/spectral_check.csv")
    KEY["spectral"] = dict(r_min=int(sp.r.min()), gmin_rel_range=[float(sp.gmin_rel.min()), float(sp.gmin_rel.max())],
                           ratio16_min=float(sp.ratio16.min()), ratio32_min=float(sp.ratio32.min()),
                           n_networks=int(len(sp)))
    print(KEY["spectral"])
json.dump(KEY, open(f"{R}/key_numbers_extra.json", "w"), indent=1, default=float)
print(json.dumps(KEY["friedman_ratio"], indent=0)); print(json.dumps(prev)); print(json.dumps(agree, indent=0))
for t in TAGS:
    print(t, KEY["sensitivity"][t])
