"""Loaders for the six benchmark data sets.

Each loader returns ``(X, y)`` with ``X`` a float32 feature matrix and
``y`` integer labels, ready for :func:`pcasnn.data.split_and_imbalance`.
Downloads are cached by the underlying libraries (Keras, scikit-learn,
NLTK) in their default locations. A ``data_type`` string ("image" or
"numerical") is also returned where the experiment scripts need it to
size the pair counts.

Note on Reuters: the raw label set is large and highly skewed, so the
three most frequent categories are kept and the rest merged into a single
"Others" class, giving a naturally imbalanced four-class problem.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def load_fashion_mnist() -> Tuple[np.ndarray, np.ndarray]:
    """Fashion-MNIST: 70,000 samples, 784 features, 10 classes."""
    from tensorflow.keras.datasets import fashion_mnist

    (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
    x = np.concatenate((x_tr, x_te), axis=0).reshape(-1, 784) / 255.0
    y = np.concatenate((y_tr, y_te), axis=0)
    return x.astype(np.float32), y.astype(int)


def load_quickdraw(samples_per_class: int = 7000,
                   seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """QuickDraw: 10 categories sampled to match Fashion-MNIST's size.

    Downloads the official Google numpy bitmaps on first use.
    """
    import os
    import urllib.request

    from sklearn.preprocessing import LabelEncoder

    categories = [
        "airplane", "apple", "banana", "bear", "bed",
        "bee", "bird", "book", "bus", "car",
    ]
    base = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap"

    x_list, y_list = [], []
    rng = np.random.default_rng(seed)
    for cat in categories:
        fname = f"{cat}.npy"
        if not os.path.exists(fname):
            urllib.request.urlretrieve(f"{base}/{cat}.npy", fname)
        data = np.load(fname)
        idx = rng.permutation(len(data))[:samples_per_class]
        x_list.append(data[idx])
        y_list.extend([cat] * samples_per_class)

    x = (np.concatenate(x_list, axis=0) / 255.0).astype(np.float32)
    y = LabelEncoder().fit_transform(y_list)
    return x, y


def load_isolet() -> Tuple[np.ndarray, np.ndarray]:
    """ISOLET: spoken-letter acoustic features, 617 dims, 26 classes."""
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    isolet = fetch_openml(data_id=300, as_frame=False, parser="auto")
    x = StandardScaler().fit_transform(isolet.data).astype(np.float32)
    y = LabelEncoder().fit_transform(isolet.target)
    return x, y


def load_multiple_features() -> Tuple[np.ndarray, np.ndarray]:
    """UCI Multiple Features: 2,000 samples, six feature groups, 10 classes."""
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    base = "https://archive.ics.uci.edu/ml/machine-learning-databases/mfeat/"
    files = ["mfeat-fac", "mfeat-fou", "mfeat-kar", "mfeat-mor", "mfeat-pix", "mfeat-zer"]
    blocks = [pd.read_csv(base + f, sep=r"\s+", header=None).values for f in files]
    x = np.hstack(blocks)
    y = np.repeat(np.arange(10), 200)
    x = StandardScaler().fit_transform(x).astype(np.float32)
    return x, y


def load_20newsgroups(max_features: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    """20 Newsgroups: TF-IDF features (dense), 20 classes."""
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    news = fetch_20newsgroups(subset="all", remove=("headers", "footers", "quotes"))
    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    x = vec.fit_transform(news.data).toarray().astype(np.float32)
    y = LabelEncoder().fit_transform(news.target)
    return x, y


def load_reuters(max_features: int = 2000) -> Tuple[np.ndarray, np.ndarray]:
    """Reuters-21578: naturally imbalanced, top-3 classes + Others.

    Downloads the NLTK Reuters corpus on first use.
    """
    import nltk
    import pandas as pd
    from nltk.corpus import reuters
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder

    nltk.download("reuters", quiet=True)
    nltk.download("punkt", quiet=True)

    doc_ids = reuters.fileids()
    texts = [reuters.raw(d) for d in doc_ids]
    first_cat = np.array(
        [reuters.categories(d)[0] if reuters.categories(d) else "unknown" for d in doc_ids]
    )

    top3 = pd.Series(first_cat).value_counts().index[:3].tolist()
    y_str = np.array([c if c in top3 else "Others" for c in first_cat])

    vec = TfidfVectorizer(max_features=max_features, stop_words="english")
    x = vec.fit_transform(texts).toarray().astype(np.float32)
    y = LabelEncoder().fit_transform(y_str)
    return x, y


# name -> (loader, data_type) for the cross-domain driver
REGISTRY = {
    "Fashion-MNIST": (load_fashion_mnist, "image"),
    "QuickDraw": (load_quickdraw, "image"),
    "ISOLET": (load_isolet, "numerical"),
    "Multiple-Features": (load_multiple_features, "numerical"),
    "20-Newsgroups": (load_20newsgroups, "numerical"),
    "Reuters-21578": (load_reuters, "numerical"),
}
