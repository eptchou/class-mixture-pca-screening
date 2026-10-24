"""Independent recomputation of the main comparisons from the seed-level CSVs.

Reads code/results/<condition>/retention_seed*.csv only. For each condition it
averages each fit over K <= 32 within seed, then compares every criterion with
PCA by paired differences over the ten seeds (Wilcoxon signed-rank and exact
sign tests, Holm adjustment over the four comparisons within each condition and
measure; unadjusted 95% t-intervals). It also reports the change in the mean
recall of the non-minority classes, (C * dBA - dMinority) / (C - 1), which
follows from balanced accuracy being the mean of the class recalls.
No model is trained.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
R = Path(os.environ.get("ADAC_RESULTS_DIR", ROOT / "code" / "results"))
KS = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32]
OTHERS = ["CB", "Variance", "NISP", "Random"]
CONDS = {"fmnist_bal": 10, "fmnist_r0.1": 10, "mfeat_bal": 10, "mfeat_r0.1": 10,
         "reuters_bal": 4, "reuters_nat": 4, "fmnist_r0.3": 10, "fmnist_r0.02": 10,
         "fmnist_r0.01": 10, "fmnist_r0.002": 10}
IMBALANCED = ["fmnist_r0.3", "fmnist_r0.1", "fmnist_r0.02", "fmnist_r0.01",
              "fmnist_r0.002", "mfeat_r0.1", "reuters_nat"]


def holm(p):
    p = np.asarray(p, float)
    out, run = np.empty(len(p)), 0.0
    for r, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out


def tci(x):
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def load(tag):
    d = pd.concat(pd.read_csv(f) for f in sorted(glob.glob(str(R / tag / "retention_seed*.csv"))))
    d["criterion"] = d.criterion.replace("CB-PCA", "CB")
    full = d[d.criterion == "Full"].set_index("seed")
    d = d[d.criterion != "Full"].copy()
    d["prr"] = d.pair_f1 / d.seed.map(full.pair_f1)
    return d, full


total_pos = 0
for tag, C in CONDS.items():
    d, full = load(tag)
    low = d[d.K.isin(KS)].groupby(["seed", "criterion"])[["prr", "min_recall", "bal_acc"]].mean()
    print(f"\n== {tag}  unpruned: pair F1 {full.pair_f1.mean():.4f}, minority recall "
          f"{full.min_recall.mean():.4f}, BA {full.bal_acc.mean():.4f}")
    for m in ["prr", "min_recall", "bal_acc"]:
        x = low[m].unstack("criterion")
        diffs = {k: x[k] - x["PCA"] for k in OTHERS}
        pw = [stats.wilcoxon(v).pvalue if (v != 0).any() else 1.0 for v in diffs.values()]
        ps = []
        for v in diffs.values():
            nz = v[v != 0]
            ps.append(stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue if len(nz) else 1.0)
        hw, hs = holm(pw), holm(ps)
        means = "  ".join(f"{k} {x[k].mean():.3f}" for k in ["PCA"] + OTHERS)
        print(f"  {m:10s} {means}")
        for j, k in enumerate(OTHERS):
            lo, hi = tci(diffs[k])
            print(f"    {k:8s}-PCA {diffs[k].mean():+.3f}  95% CI ({lo:+.3f}, {hi:+.3f})  "
                  f"signed-rank Holm p {hw[j]:.3f}  sign Holm p {hs[j]:.3f}  "
                  f"+/0/- {(diffs[k] > 0).sum()}/{(diffs[k] == 0).sum()}/{(diffs[k] < 0).sum()}")
        if m == "min_recall" and tag in IMBALANCED:
            total_pos += int((diffs["CB"] > 0).sum())
    if tag in IMBALANCED:
        e = d[d.K.isin(KS) & d.criterion.isin(["PCA", "CB"])].copy()
        e["other"] = (C * e.bal_acc - e.min_recall) / (C - 1)
        v = e.groupby(["seed", "criterion"]).other.mean().unstack()
        v = v["CB"] - v["PCA"]
        lo, hi = tci(v)
        print(f"  mean recall of the other classes, CB-PCA {v.mean():+.4f}  95% CI ({lo:+.4f}, {hi:+.4f})")

print(f"\nPositive CB-PCA minority-recall differences over the seven imbalanced conditions: {total_pos} of 70")
