"""Prepare raw datasets into compact npz/pickle files used by the experiments."""
import gzip, zipfile, re, pickle, os
from pathlib import Path
import numpy as np
import pandas as pd

OUT = os.environ.get("ADAC_DATA_DIR", str(Path(__file__).resolve().parent / "data"))
RAW = str(Path(OUT) / "raw")
Path(OUT).mkdir(parents=True, exist_ok=True)


def read_idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[:4], "big")
    ndim = magic & 0xFF
    dims = [int.from_bytes(data[4 + 4 * i: 8 + 4 * i], "big") for i in range(ndim)]
    return np.frombuffer(data, dtype=np.uint8, offset=4 + 4 * ndim).reshape(dims)


# ---------------- Fashion-MNIST ----------------
Xtr = read_idx(f"{RAW}/train-images-idx3-ubyte.gz").reshape(-1, 784)
ytr = read_idx(f"{RAW}/train-labels-idx1-ubyte.gz")
Xte = read_idx(f"{RAW}/t10k-images-idx3-ubyte.gz").reshape(-1, 784)
yte = read_idx(f"{RAW}/t10k-labels-idx1-ubyte.gz")
X = np.concatenate([Xtr, Xte]); y = np.concatenate([ytr, yte]).astype(np.int64)
np.savez_compressed(f"{OUT}/fmnist.npz", X=X, y=y)
print("fmnist", X.shape, np.bincount(y))

# ---------------- Multiple Features ----------------
names = ["mfeat_factors", "mfeat_fourier", "mfeat_karhunen",
         "mfeat_morphological", "mfeat_pixel", "mfeat_zernike"]
blocks, targets = [], []
for n in names:
    d = pd.read_csv(f"{RAW}/{n}.tsv.gz", sep="\t")
    targets.append(d["target"].to_numpy())
    blocks.append(d.drop(columns="target").to_numpy(dtype=np.float64))
for t in targets[1:]:
    assert np.array_equal(t, targets[0]), "row order differs between feature groups"
Xm = np.hstack(blocks); ym = targets[0].astype(np.int64)
assert np.array_equal(ym, np.repeat(np.arange(10), 200)), "unexpected class order"
np.savez_compressed(f"{OUT}/mfeat.npz", X=Xm, y=ym)
print("mfeat", Xm.shape, np.bincount(ym), [b.shape[1] for b in blocks])

# ---------------- Reuters-21578 (NLTK ApteMod) ----------------
z = zipfile.ZipFile(f"{RAW}/reuters.zip")
cats = {}
for line in z.read("reuters/cats.txt").decode("latin-1").splitlines():
    parts = line.split()
    if parts:
        cats[parts[0]] = parts[1:]
docs, labels, ids = [], [], []
for fid, cs in cats.items():
    txt = z.read(f"reuters/{fid}").decode("latin-1")
    docs.append(txt); labels.append(cs[0] if cs else "unknown"); ids.append(fid)
pickle.dump({"data": docs, "first_cat": labels, "all_cats": [cats[i] for i in ids], "ids": ids},
            open(f"{OUT}/reuters.pkl", "wb"))
vc = pd.Series(labels).value_counts()
print("reuters docs", len(docs)); print(vc.head(6))
