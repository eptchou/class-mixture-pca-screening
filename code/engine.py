"""Corrected re-implementation of the PCA-screening + Siamese pipeline (JAX, CPU).

Differences from the thesis notebook (all deliberate, documented in the paper):
  * BatchNorm parameters and moving statistics of the retained first-layer units are
    transferred to the Siamese subnetwork (the notebook re-initialised them).
  * The full (K = 128) and pruned Siamese models use the same training protocol
    (early stopping on validation loss, patience 2, best weights restored).
  * Scalers / TF-IDF are fitted on the training split only.
  * Ten training seeds; class-level (minority) metrics from prototype classification.
"""
import os, time, pickle
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import optax
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, balanced_accuracy_score

ROOT = Path(__file__).resolve().parent
DATA = os.environ.get("ADAC_DATA_DIR", str(ROOT / "data"))
RESULTS = os.environ.get("ADAC_RESULTS_DIR", str(ROOT / "results"))
P1, P2, P3 = 128, 64, 32
BN_EPS, BN_MOM = 1e-3, 0.99
BATCH = 128
ADAM = dict(learning_rate=1e-3, b1=0.9, b2=0.999, eps=1e-7)   # Keras defaults
TEST_SIZE, VAL_SIZE = 0.10, 0.30
K_GRID = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96]
CRITERIA = ["PCA", "Variance", "NISP", "Random", "CB-PCA"]


# =====================================================================
# Data
# =====================================================================
def downsample_class(X, y, cls, ratio, rng, min_keep=5):
    keep = []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        if c == cls and ratio < 1.0:
            n = min(max(int(len(idx) * ratio), min_keep), len(idx))
            idx = rng.choice(idx, n, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    rng.shuffle(keep)
    return X[keep], y[keep]


def load_condition(name, ratio, seed):
    """Return dict with train/val/test (condition) and balanced test arrays."""
    rng_split = 42
    if name == "fmnist":
        d = np.load(f"{DATA}/fmnist.npz"); X = d["X"].astype(np.float32) / 255.0; y = d["y"]
        minority = 0; kind = "image"
    elif name == "mfeat":
        d = np.load(f"{DATA}/mfeat.npz"); X = d["X"].astype(np.float32); y = d["y"]
        minority = 0; kind = "numeric"
    elif name == "quickdraw":
        d = np.load(f"{DATA}/quickdraw10.npz", allow_pickle=True)
        X = d["X"].astype(np.float32) / 255.0
        y = np.unique(d["y"], return_inverse=True)[1].astype(np.int64)   # = LabelEncoder
        minority = 0; kind = "image"
    elif name == "isolet":
        d = np.load(f"{DATA}/isolet.npz", allow_pickle=True)
        X = d["X"].astype(np.float32)
        y = np.unique(d["y"].astype(str), return_inverse=True)[1].astype(np.int64)
        minority = 0; kind = "numeric"
    elif name == "news20":
        r = pickle.load(open(f"{DATA}/news20.pkl", "rb"))
        X = np.array(r["data"], dtype=object); y = np.asarray(r["target"]).astype(np.int64)
        minority = 0; kind = "text"
    elif name.startswith("reuters"):
        r = pickle.load(open(f"{DATA}/reuters.pkl", "rb"))
        lab = np.array(r["first_cat"]); docs = np.array(r["data"], dtype=object)
        vc = {c: (lab == c).sum() for c in np.unique(lab)}
        top3 = sorted(vc, key=lambda c: -vc[c])[:3]
        names = ["crude", "acq", "earn", "Others"]          # minority (smallest) first
        assert set(top3) == {"earn", "acq", "crude"}
        lab4 = np.array([c if c in top3 else "Others" for c in lab])
        y = np.array([names.index(c) for c in lab4], dtype=np.int64)
        X = docs; minority = 0; kind = "text"
        if name == "reuters_bal":                          # undersample to smallest class
            rng = np.random.default_rng(42); m = np.bincount(y).min()
            keep = np.concatenate([rng.choice(np.where(y == c)[0], m, replace=False)
                                   for c in range(4)])
            rng.shuffle(keep); X, y = X[keep], y[keep]
    else:
        raise ValueError(name)

    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=rng_split)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=VAL_SIZE, stratify=ytv, random_state=rng_split)

    # Preserve the pre-subsampling test and validation inputs. Fit every
    # data-dependent transformation on the retained training inputs only.
    Xte_bal, yte_bal = Xte, yte
    Xva_bal, yva_bal = Xva, yva
    if ratio < 1.0:
        rng = np.random.default_rng(1000 + seed)
        Xtr, ytr = downsample_class(Xtr, ytr, minority, ratio, rng)
        Xva, yva = downsample_class(Xva, yva, minority, ratio, rng)
        Xte, yte = downsample_class(Xte, yte, minority, ratio, rng)
    if kind == "numeric":
        sc = StandardScaler().fit(Xtr)
        Xtr, Xva, Xte, Xte_bal, Xva_bal = (
            sc.transform(a).astype(np.float32)
            for a in (Xtr, Xva, Xte, Xte_bal, Xva_bal))
    elif kind == "text":
        tf = TfidfVectorizer(max_features=5000, stop_words="english").fit(Xtr)
        Xtr, Xva, Xte, Xte_bal, Xva_bal = (
            tf.transform(a).toarray().astype(np.float32)
            for a in (Xtr, Xva, Xte, Xte_bal, Xva_bal))
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte,
                Xte_bal=Xte_bal, yte_bal=yte_bal, Xva_bal=Xva_bal, yva_bal=yva_bal,
                minority=minority, kind=kind,
                C=int(len(np.unique(y))))


# =====================================================================
# Networks
# =====================================================================
def glorot(rng, fi, fo):
    lim = np.sqrt(6.0 / (fi + fo))
    return jnp.asarray(rng.uniform(-lim, lim, size=(fi, fo)).astype(np.float32))


def clf_init(rng, d, C):
    p = dict(W1=glorot(rng, d, P1), b1=jnp.zeros(P1), g=jnp.ones(P1), be=jnp.zeros(P1),
             W2=glorot(rng, P1, P2), b2=jnp.zeros(P2), W3=glorot(rng, P2, C), b3=jnp.zeros(C))
    s = dict(mean=jnp.zeros(P1), var=jnp.ones(P1))
    return p, s


def clf_apply(p, s, x, train):
    h = x @ p["W1"] + p["b1"]
    if train:
        mu, var = h.mean(0), h.var(0)
        hn = (h - mu) / jnp.sqrt(var + BN_EPS)
        s = dict(mean=BN_MOM * s["mean"] + (1 - BN_MOM) * mu, var=BN_MOM * s["var"] + (1 - BN_MOM) * var)
    else:
        hn = (h - s["mean"]) / jnp.sqrt(s["var"] + BN_EPS)
    a1 = jax.nn.relu(p["g"] * hn + p["be"])
    a2 = jax.nn.relu(a1 @ p["W2"] + p["b2"])
    return a2 @ p["W3"] + p["b3"], s, a1, a2


_opt = optax.adam(**ADAM)


@jax.jit
def _clf_epoch(p, s, o, X, y, perm):
    def loss_fn(p, s, xb, yb):
        logits, ns, _, _ = clf_apply(p, s, xb, True)
        return optax.softmax_cross_entropy_with_integer_labels(logits, yb).mean(), ns

    def step(carry, ib):
        p, s, o = carry
        (l, ns), g = jax.value_and_grad(loss_fn, has_aux=True)(p, s, X[ib], y[ib])
        u, o = _opt.update(g, o, p)
        return (optax.apply_updates(p, u), ns, o), l

    (p, s, o), ls = jax.lax.scan(step, (p, s, o), perm)
    return p, s, o, ls.mean()


@jax.jit
def _clf_predict(p, s, X):
    logits, _, a1, a2 = clf_apply(p, s, X, False)
    return logits, a1, a2


def batched(fn, X, *args, bs=4096):
    outs = [fn(*args, X[i:i + bs]) for i in range(0, len(X), bs)]
    return [np.concatenate([np.asarray(o[j]) for o in outs]) for j in range(len(outs[0]))]


def train_classifier(D, seed, epochs=15, patience=3):
    rng = np.random.default_rng(seed)
    X, y = jnp.asarray(D["Xtr"]), jnp.asarray(D["ytr"])
    p, s = clf_init(rng, X.shape[1], D["C"]); o = _opt.init(p)
    n = len(y); nb = n // BATCH
    best, best_acc, wait = None, -1.0, 0
    for ep in range(epochs):
        perm = jnp.asarray(rng.permutation(n)[: nb * BATCH].reshape(nb, BATCH))
        p, s, o, _ = _clf_epoch(p, s, o, X, y, perm)
        logits, _, _ = batched(lambda p_, s_, x: _clf_predict(p_, s_, x), D["Xva"], p, s)
        acc = float((logits.argmax(1) == D["yva"]).mean())
        if acc > best_acc:
            best_acc, best, wait = acc, (p, s), 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best[0], best[1], best_acc


def activations(p, s, X):
    _, a1, a2 = batched(lambda p_, s_, x: _clf_predict(p_, s_, x), X, p, s)
    return a1.astype(np.float64), a2.astype(np.float64)


# =====================================================================
# Ranking criteria
# =====================================================================
def eig_desc(S):
    w, V = np.linalg.eigh(S)
    w, V = w[::-1], V[:, ::-1]
    return np.clip(w, 0, None), V           # columns of V are eigenvectors


def pca_score_from_cov(S):
    w, V = eig_desc(S)
    lam = w / w.sum()
    return (np.abs(V) * lam[None, :]).sum(1), lam   # Score_i = sum_k lam_k |V_ik|


def pooled_cov(A):
    return np.cov(A, rowvar=False, ddof=1)


def balanced_cov(A, y, classes):
    """Equal-mixture plug-in covariance; classes is the fixed full class set.

    Within-class sample covariance uses ddof=1. Missing/singleton classes
    are errors, rather than a change to the estimand or a zero imputation.
    The finite-sample bias is stated in the manuscript, eq. (CB bias).
    """
    A, y, classes = np.asarray(A), np.asarray(y), np.asarray(classes)
    if len(A) != len(y) or len(classes) != len(np.unique(classes)):
        raise ValueError("Invalid activation/label/class dimensions")
    if not np.isin(y, classes).all():
        raise ValueError("Observed labels outside the fixed class set")
    counts = np.array([(y == c).sum() for c in classes])
    if np.any(counts < 2):
        raise ValueError(f"CB covariance requires >=2 per fixed class: {counts}")
    mus = np.stack([A[y == c].mean(0) for c in classes])
    mbar = mus.mean(0)
    return sum(np.cov(A[y == c], rowvar=False, ddof=1)
               + np.outer(mu - mbar, mu - mbar)
               for c, mu in zip(classes, mus)) / len(classes)


def inf_fs_scores(A, alpha=0.2, rng=None):
    rng = rng or np.random.default_rng(0)
    An = A + rng.normal(0, 1e-8, A.shape)
    sd = An.std(0); sd = sd / sd.max() if sd.max() > 0 else sd
    sigma = np.maximum.outer(sd, sd)
    rho = spearmanr(An, axis=0)[0]; rho = np.nan_to_num(rho, nan=0.0)
    Amat = alpha * sigma + (1 - alpha) * (1 - np.abs(rho))
    np.fill_diagonal(Amat, 0)
    r = 0.9 / np.max(np.abs(np.linalg.eigvals(Amat)))
    Sinf = np.linalg.inv(np.eye(len(Amat)) - r * Amat) - np.eye(len(Amat))
    return np.real(Sinf.sum(1))


def thresholds_from_lam(lam):
    cum = np.cumsum(lam)
    x = np.arange(1, len(lam) + 1); yv = lam
    num = np.abs((yv[-1] - yv[0]) * x - (x[-1] - x[0]) * yv + x[-1] * yv[0] - yv[-1] * x[0])
    den = np.sqrt((yv[-1] - yv[0]) ** 2 + (x[-1] - x[0]) ** 2)
    out = {"Elbow": int(x[np.argmax(num / den)])}
    for t in (0.85, 0.90, 0.95, 0.99):
        out[f"Var{int(t*100)}"] = int(np.argmax(cum >= t) + 1)
    return out


def all_scores(A1, A2, yv, W2, rng, classes):
    S = pooled_cov(A1)
    pca, lam = pca_score_from_cov(S)
    cb, _ = pca_score_from_cov(balanced_cov(A1, yv, classes))
    nisp = np.abs(np.asarray(W2)) @ inf_fs_scores(A2, 0.2, rng)
    return dict(PCA=pca, Variance=np.diag(S).copy(), NISP=nisp, **{"CB-PCA": cb}), lam


def order(score):
    return np.argsort(-score, kind="stable")


# =====================================================================
# Siamese network
# =====================================================================
def embed(fr, tp, x, train, key):
    h = x @ fr["W1"] + fr["b1"]
    a1 = jax.nn.relu(fr["g"] * (h - fr["mean"]) / jnp.sqrt(fr["var"] + BN_EPS) + fr["be"])
    a2 = jax.nn.relu(a1 @ tp["W2"] + tp["b2"])
    a3 = jax.nn.relu(a2 @ tp["W3"] + tp["b3"])
    if train:
        keep = jax.random.bernoulli(key, 0.8, a3.shape)
        a3 = jnp.where(keep, a3 / 0.8, 0.0)
    return a3


def head(tp, ea, eb):
    return (jnp.abs(ea - eb) @ tp["wh"] + tp["bh"])[:, 0]


@jax.jit
def _snn_epoch(tp, o, fr, X, ia, ib, yl, key):
    def loss_fn(tp, xa, xb, yb, k):
        k1, k2 = jax.random.split(k)
        lg = head(tp, embed(fr, tp, xa, True, k1), embed(fr, tp, xb, True, k2))
        return optax.sigmoid_binary_cross_entropy(lg, yb).mean()

    keys = jax.random.split(key, ia.shape[0])

    def step(carry, b):
        tp, o = carry
        a, bb, yy, k = b
        l, g = jax.value_and_grad(loss_fn)(tp, X[a], X[bb], yy, k)
        u, o = _opt.update(g, o, tp)
        return (optax.apply_updates(tp, u), o), l

    (tp, o), ls = jax.lax.scan(step, (tp, o), (ia, ib, yl, keys))
    return tp, o, ls.mean()


@jax.jit
def _embed_eval(fr, tp, X):
    return embed(fr, tp, X, False, None)


@jax.jit
def _pair_loss(tp, Ea, Eb, yl):
    return optax.sigmoid_binary_cross_entropy(head(tp, Ea, Eb), yl).mean()


def make_pairs(y, n_pairs, rng):
    classes = np.unique(y); C = len(classes)
    idx = [np.where(y == c)[0] for c in classes]
    valid = np.array([j for j in range(C) if len(idx[j]) > 1])
    n_pos, n_neg = (n_pairs + 1) // 2, n_pairs // 2
    cp = rng.choice(valid, n_pos)
    a = np.empty(n_pos, int); b = np.empty(n_pos, int)
    for j in np.unique(cp):
        m = cp == j; L = len(idx[j]); k = m.sum()
        i1 = rng.integers(L, size=k); i2 = rng.integers(L - 1, size=k); i2 = i2 + (i2 >= i1)
        a[m], b[m] = idx[j][i1], idx[j][i2]
    c1 = rng.integers(C, size=n_neg); c2 = (c1 + 1 + rng.integers(C - 1, size=n_neg)) % C
    na = np.array([idx[j][rng.integers(len(idx[j]))] for j in c1])
    nb_ = np.array([idx[j][rng.integers(len(idx[j]))] for j in c2])
    ia = np.concatenate([a, na]); ib = np.concatenate([b, nb_])
    lab = np.concatenate([np.ones(n_pos), np.zeros(n_neg)]).astype(np.float32)
    return ia, ib, lab


def transfer(p, s, idx, rng):
    idx = np.asarray(idx)
    fr = dict(W1=p["W1"][:, idx], b1=p["b1"][idx], g=p["g"][idx], be=p["be"][idx],
              mean=s["mean"][idx], var=s["var"][idx])
    tp = dict(W2=p["W2"][idx, :], b2=p["b2"], W3=glorot(rng, P2, P3), b3=jnp.zeros(P3),
              wh=glorot(rng, P3, 1), bh=jnp.zeros(1))
    return fr, tp


def train_snn(fr, tp, D, n_train, n_val, seed, epochs=10, patience=2):
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    X = jnp.asarray(D["Xtr"])
    ia, ib, yl = make_pairs(D["ytr"], n_train, rng)
    va, vb, vy = make_pairs(D["yva"], n_val, rng)
    Xva = jnp.asarray(D["Xva"])
    o = _opt.init(tp)
    nb = n_train // BATCH
    best, best_loss, wait, ep_used = tp, np.inf, 0, 0
    for ep in range(epochs):
        perm = rng.permutation(n_train)[: nb * BATCH].reshape(nb, BATCH)
        key, sub = jax.random.split(key)
        tp, o, _ = _snn_epoch(tp, o, fr, X, jnp.asarray(ia[perm]), jnp.asarray(ib[perm]),
                              jnp.asarray(yl[perm]), sub)
        E = _embed_eval(fr, tp, Xva)
        vl = float(_pair_loss(tp, E[va], E[vb], jnp.asarray(vy)))
        ep_used = ep + 1
        if vl < best_loss:
            best_loss, best, wait = vl, tp, 0
        else:
            wait += 1
            if wait >= patience:
                break
    return best, ep_used


def pair_metrics(fr, tp, X, y, n_pairs, n_sets=10):
    E = _embed_eval(fr, tp, jnp.asarray(X))
    accs, f1s = [], []
    for t in range(n_sets):
        a, b, lab = make_pairs(y, n_pairs, np.random.default_rng(2000 + t))
        pred = (np.asarray(head(tp, E[a], E[b])) >= 0).astype(int)
        accs.append((pred == lab).mean()); f1s.append(f1_score(lab, pred, zero_division=0))
    return float(np.mean(accs)), float(np.mean(f1s))


def prototype_metrics(fr, tp, D, Xeval, yeval):
    """Classify by the SNN head score against class centroids of training embeddings."""
    Etr = np.asarray(_embed_eval(fr, tp, jnp.asarray(D["Xtr"])))
    cls = np.unique(D["ytr"])
    cent = np.stack([Etr[D["ytr"] == c].mean(0) for c in cls])
    Ee = np.asarray(_embed_eval(fr, tp, jnp.asarray(Xeval)))
    wh = np.asarray(tp["wh"])[:, 0]
    sc = np.stack([(np.abs(Ee - c) * wh).sum(1) for c in cent], 1)
    pred = cls[sc.argmax(1)]
    m = D["minority"]
    return dict(min_recall=float(recall_score(yeval, pred, labels=[m], average="macro", zero_division=0)),
                min_f1=float(f1_score(yeval, pred, labels=[m], average="macro", zero_division=0)),
                bal_acc=float(balanced_accuracy_score(yeval, pred)),
                macro_f1=float(f1_score(yeval, pred, average="macro", zero_division=0)))
