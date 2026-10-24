# PCA-Guided Neuron Screening for Siamese Networks under Class Imbalance

Code for the study *"PCA-Guided Neuron Screening for Siamese Neural
Networks: Preserving Minority-Class Features under Class Imbalance."*

The method applies principal component analysis (PCA) to the hidden-layer
activations of a pre-trained classifier, scores each neuron by its
absolute PCA loadings weighted by the explained-variance ratios, and
transfers the top-ranked neurons into a Siamese neural network whose
pairwise distance metric decouples classification from the class
proportions. This repository reproduces both experiments in the paper:

1. **Cross-domain validation** over six benchmarks (Fashion-MNIST,
   QuickDraw, ISOLET, Multiple Features, 20 Newsgroups, Reuters-21578),
   comparing PCA-based screening against NISP and random selection.
2. **Multi-ratio robustness** on Fashion-MNIST across imbalance ratios
   0.3, 0.1, 0.02, 0.01 and 0.002, with Kruskal–Wallis / Dunn tests and a
   PCA-vs-NISP chi-square independence analysis.

## Installation

```bash
git clone https://github.com/<your-account>/pcasnn-imbalance.git
cd pcasnn-imbalance
python -m venv .venv && source .venv/bin/activate   # optional
pip install -e .            # or: pip install -r requirements.txt
```

The research ran on Python 3.12 with TensorFlow 2.20.0. A GPU is
recommended for the full sweeps but not required.

## Repository layout

```
pcasnn/                 importable library (single source of truth)
  config.py             ExperimentConfig: all hyperparameters
  seeding.py            deterministic seeding
  data.py               stratified split + controlled imbalance
  datasets.py           loaders for the six benchmarks
  screening.py          PCA importance, Inf-FS (NISP), elbow, Jaccard
  models.py             baseline MLP, pruned subnet, Siamese network
  pairs.py              pair construction + Siamese evaluation
  experiments.py        stability, PRR sweep, KW/Dunn, chi-square
  plotting.py           PRR curve figures
scripts/
  run_cross_domain.py   Experiment 1
  run_multi_ratio.py    Experiment 2 (training + per-run stability)
  run_analysis.py       KW/Dunn tables + PCA-vs-NISP independence
```

## Reproducing the experiments

**Quick smoke test** (tiny settings, to check the pipeline runs):

```bash
python scripts/run_cross_domain.py --datasets Fashion-MNIST --quick
```

**Experiment 1 — cross-domain validation** (all six data sets, ratio 0.1):

```bash
python scripts/run_cross_domain.py --out results/cross_domain
```

Per data set this writes stability and PRR tables (`*_stability.csv`,
`*_prr.csv`) and PRR figures under `results/cross_domain/<dataset>/`.

**Experiment 2 — multi-ratio robustness** on Fashion-MNIST:

```bash
python scripts/run_multi_ratio.py --out results/multi_ratio
python scripts/run_analysis.py --in results/multi_ratio \
    --out results/multi_ratio/analysis --independence
```

`run_multi_ratio.py` writes `stability_ratio_*.csv` and `prr_ratio_*.csv`
per ratio. `run_analysis.py` then produces `kruskal_wallis.csv`, the
significant-metric `dunn_*.csv` tables, and (with `--independence`)
`pca_vs_nisp_independence.csv`.

## Using the library directly

```python
import numpy as np
from pcasnn import (ExperimentConfig, set_global_seed, split_and_imbalance,
                    create_mlp_classifier, pca_importance_and_thresholds)

cfg = ExperimentConfig()
set_global_seed(cfg.base_seed)

# X, y are your feature matrix and integer labels
(x_tr, y_tr), (x_val, y_val), *_ = split_and_imbalance(X, y, ratio=0.1)

model = create_mlp_classifier(x_tr.shape[1:], num_classes=len(np.unique(y)))
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_tr, y_tr, validation_data=(x_val, y_val),
          epochs=cfg.classifier_epochs, batch_size=cfg.batch_size, verbose=0)

importance, ranked_neurons, thresholds = pca_importance_and_thresholds(model, x_val)
print("Retain top-K at each cutoff:", thresholds)
```

## Method summary

Given the first hidden layer's activation matrix on the validation set,
PCA yields components with loadings `V_ki` and explained-variance ratios
`lambda_k`. Each neuron `i` receives the screening score

```
score(i) = sum_k |V_ki| * lambda_k
```

Neurons are ranked by this score; the top `K` (chosen at the elbow point
or an 85/90/95/99% cumulative-variance cutoff) and their pre-trained
weights are frozen into the first layer of the Siamese subnetwork, and the
network is fine-tuned on same/different-class pairs. Because the objective
is pairwise similarity rather than a class-prior-dependent boundary,
minority-class discrimination is preserved under compression.

## Notes on reproducibility

- Splitting is always performed on balanced data first, then imbalance is
  imposed per split, so the test distribution stays under experimenter
  control and there is no leakage between splits.
- Seeds are fixed and deterministic-op flags are set, but exact bitwise
  reproducibility on GPU also depends on the CUDA/cuDNN build.
- Results are saved as CSV by default; re-enable Excel export by
  installing `openpyxl` and adjusting the `to_csv` calls if desired.

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff), or cite the accompanying paper once
published.
