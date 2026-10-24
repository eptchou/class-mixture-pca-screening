"""Second class-level evaluation: class-weighted logistic-regression probe on the
Siamese embeddings. Re-creates exactly the same fitted networks as run_condition.py
(same seeds and recorded numerical environment) and checks that the stored
prototype metrics are reproduced before computing the probe metrics.

usage: python run_probe.py <name> <ratio> <tag> [seeds...]
"""
import sys, os, time, json
import numpy as np, pandas as pd
import jax.numpy as jnp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score
import engine as E

name, ratio, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
seeds = [int(s) for s in sys.argv[4:]] or list(range(1, 11))
OUT = f"{E.RESULTS}/{tag}"
os.makedirs(OUT, exist_ok=True)
if os.name == "posix":
    import fcntl
    _probe_lock = open(f"{OUT}/.probe.lock", "a")
    fcntl.flock(_probe_lock, fcntl.LOCK_EX)
n_train, n_val, n_test = (20000, 4000, 4000) if name in ("fmnist", "quickdraw") else (5000, 1000, 1000)
PROBE_N = 10000                       # training embeddings used to fit the probe

for seed in seeds:
    out = f"{OUT}/probe_seed{seed}.csv"
    if os.path.exists(out):
        print("skip", out); continue
    t0 = time.time()
    stored = pd.read_csv(f"{OUT}/retention_seed{seed}.csv")
    D = E.load_condition(name, ratio, seed)
    checkpoint = f"{OUT}/model_seed{seed}.npz"
    if os.path.exists(checkpoint):
        z = np.load(checkpoint)
        p = {k: jnp.asarray(z[k]) for k in ["W1", "b1", "g", "be", "W2", "b2", "W3", "b3"]}
        s = dict(mean=jnp.asarray(z["bn_mean"]), var=jnp.asarray(z["bn_var"]))
    else:
        p, s, _ = E.train_classifier(D, seed)
    A1, A2 = E.activations(p, s, D["Xva"])
    scores, _ = E.all_scores(A1, A2, D["yva"], p["W2"], np.random.default_rng(seed), np.arange(D["C"]))
    rng_probe = np.random.default_rng(77 + seed)
    sub = rng_probe.choice(len(D["ytr"]), min(PROBE_N, len(D["ytr"])), replace=False)
    # keep every minority training observation in the probe's training set
    sub = np.union1d(sub, np.where(D["ytr"] == D["minority"])[0])
    rows, maxdev = [], 0.0

    def fit_eval(crit, K, idx, fit_seed):
        global maxdev
        fr, tp = E.transfer(p, s, idx, np.random.default_rng(fit_seed))
        tp, ep = E.train_snn(fr, tp, D, n_train, n_val, fit_seed)
        pm = E.prototype_metrics(fr, tp, D, D["Xte_bal"], D["yte_bal"])
        ref = stored[(stored.criterion == crit) & (stored.K == K)].iloc[0]
        maxdev = max(maxdev, abs(pm["min_recall"] - ref.min_recall), abs(pm["bal_acc"] - ref.bal_acc))
        Etr = np.asarray(E._embed_eval(fr, tp, jnp.asarray(D["Xtr"][sub])))
        Ete = np.asarray(E._embed_eval(fr, tp, jnp.asarray(D["Xte_bal"])))
        sc = StandardScaler().fit(Etr)
        clf = LogisticRegression(class_weight="balanced", max_iter=500, C=1.0)
        clf.fit(sc.transform(Etr), D["ytr"][sub])
        pred = clf.predict(sc.transform(Ete)); y = D["yte_bal"]; m = D["minority"]
        rows.append(dict(seed=seed, criterion=crit, K=K,
                         probe_min_recall=recall_score(y, pred, labels=[m], average="macro", zero_division=0),
                         probe_bal_acc=balanced_accuracy_score(y, pred),
                         probe_macro_f1=f1_score(y, pred, average="macro", zero_division=0)))

    fit_eval("Full", 128, np.arange(128), 10_000 * seed + 128)
    for K in E.K_GRID:
        for crit in E.CRITERIA:
            if crit == "Random":
                idx = np.sort(np.random.default_rng(10_000 * seed + 100 * K + 7).choice(128, K, replace=False))
            else:
                idx = E.order(scores[crit])[:K]
            fit_eval(crit, K, idx, 10_000 * seed + K)
    # A failed numerical check must not silently publish a new probe result.
    if maxdev > 1e-8:
        raise RuntimeError(f"Prototype reproduction failed for seed {seed}: {maxdev:.3g}. "
                           "Use the recorded environment or regenerate the paired retention results.")
    json.dump(dict(seed=seed, criterion_fits=len(rows), max_prototype_difference=maxdev, checked_metrics=["min_recall", "bal_acc"]), open(out.replace(".csv", ".verified.json"), "w"), indent=2)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"{tag} seed {seed} probe done in {time.time()-t0:.0f}s; max deviation from stored prototype metrics = {maxdev:.2e}", flush=True)
