"""Run the retention experiment for one data condition.

usage: python run_condition.py <name> <ratio> <tag> [seeds...]
  name  : fmnist | mfeat | reuters_nat | reuters_bal
  ratio : minority keep-ratio (1.0 = no subsampling)
  tag   : output folder name
"""
import sys, os, json, time, pickle
import numpy as np
import pandas as pd
import jax.numpy as jnp
import engine as E

name, ratio, tag = sys.argv[1], float(sys.argv[2]), sys.argv[3]
seeds = [int(s) for s in sys.argv[4:]] or list(range(1, 11))
OUT = f"{E.RESULTS}/{tag}"
os.makedirs(OUT, exist_ok=True)
# Serialise concurrent launches of the same condition; completed seeds resume.
if os.name == "posix":
    import fcntl
    _condition_lock = open(f"{OUT}/.condition.lock", "a")
    fcntl.flock(_condition_lock, fcntl.LOCK_EX)
n_train, n_val, n_test = (20000, 4000, 4000) if name in ("fmnist", "quickdraw") else (5000, 1000, 1000)

for seed in seeds:
    csv = f"{OUT}/retention_seed{seed}.csv"
    if os.path.exists(csv):
        print("skip", csv); continue
    t0 = time.time()
    D = E.load_condition(name, ratio, seed)
    p, s, vacc = E.train_classifier(D, seed)
    A1, A2 = E.activations(p, s, D["Xva"])
    rng = np.random.default_rng(seed)
    scores, lam = E.all_scores(A1, A2, D["yva"], p["W2"], rng, np.arange(D["C"]))
    thr = E.thresholds_from_lam(lam)
    np.savez_compressed(f"{OUT}/model_seed{seed}.pending.npz", **{k: np.asarray(v) for k, v in p.items()},
             bn_mean=np.asarray(s["mean"]), bn_var=np.asarray(s["var"]),
             **{f"score_{k}": v for k, v in scores.items()}, lam=lam, val_acc=vacc)
    # Validate a complete ZIP before publishing the checkpoint path.
    with np.load(f"{OUT}/model_seed{seed}.pending.npz") as check:
        for key in check.files:
            _ = check[key]
    os.replace(f"{OUT}/model_seed{seed}.pending.npz", f"{OUT}/model_seed{seed}.npz")
    meta = dict(seed=seed, val_acc=vacc, thresholds=thr, n_train=len(D["ytr"]),
                n_val=len(D["yva"]), n_test=len(D["yte"]),
                class_counts_train=np.bincount(D["ytr"]).tolist(),
                class_counts_val=np.bincount(D["yva"]).tolist(),
                class_counts_test=np.bincount(D["yte"]).tolist())
    json.dump(meta, open(f"{OUT}/meta_seed{seed}.json", "w"), indent=1)

    rows = []
    def fit_eval(crit, K, idx, fit_seed):
        t1 = time.time()
        fr, tp = E.transfer(p, s, idx, np.random.default_rng(fit_seed))
        tp, ep = E.train_snn(fr, tp, D, n_train, n_val, fit_seed)
        acc, f1 = E.pair_metrics(fr, tp, D["Xte"], D["yte"], n_test)
        acc_b, f1_b = E.pair_metrics(fr, tp, D["Xte_bal"], D["yte_bal"], n_test)
        pm = E.prototype_metrics(fr, tp, D, D["Xte_bal"], D["yte_bal"])
        rows.append(dict(seed=seed, criterion=crit, K=K, pair_acc=acc, pair_f1=f1,
                         pair_acc_bt=acc_b, pair_f1_bt=f1_b, epochs=ep,
                         fit_sec=time.time() - t1, **pm))
        if os.environ.get("ADAC_PROGRESS") == "1":
            print("FIT_DONE", tag, seed, crit, K, ep, round(time.time()-t1, 3), flush=True)

    fit_eval("Full", 128, np.arange(128), 10_000 * seed + 128)
    for K in E.K_GRID:
        for ci, crit in enumerate(E.CRITERIA):
            if crit == "Random":
                idx = np.sort(np.random.default_rng(10_000 * seed + 100 * K + 7).choice(128, K, replace=False))
            else:
                idx = E.order(scores[crit])[:K]
            # same SNN seed for all criteria at a given (seed, K): paired comparison
            fit_eval(crit, K, idx, 10_000 * seed + K)
    pd.DataFrame(rows).to_csv(csv, index=False)
    print(f"{tag} seed {seed} done in {time.time()-t0:.0f}s; val_acc={vacc:.4f}; thr={thr}", flush=True)
