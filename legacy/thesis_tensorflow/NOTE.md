# Thesis implementation (TensorFlow) — not used for the article

This folder keeps the original TensorFlow/Keras implementation written for the master's thesis, refactored into the `pcasnn` package. **No number, table or figure in the article comes from this code.** The article's results were produced by the JAX implementation in `../../code/`.

The main differences, which the article's implementation corrects or changes:

1. **Batch normalisation.** The Siamese subnetwork creates a new `BatchNormalization(trainable=False)` layer and does not copy the trained scale, shift or moving statistics (`pcasnn/models.py`), so the transferred units do not compute the activations that were scored.
2. **Preprocessing.** `StandardScaler` (ISOLET, Multiple Features) and `TfidfVectorizer` (20 Newsgroups, Reuters) are fitted on all observations before the train/validation/test split (`pcasnn/datasets.py`).
3. **Reuters-21578.** 2,000 TF–IDF terms; after label encoding, class 0 is *Others*, the largest class, so the designated "minority" class is not a minority class. The article uses 5,000 terms and *crude* as the minority class.
4. **Replication.** Three runs per retention size (`pcasnn/config.py`); the article uses ten seeds with paired designs.
5. **Evaluation.** Pair-level accuracy and F1 only; the article adds class-level metrics (minority recall, balanced accuracy) and a logistic-regression probe.
6. **Scope.** Six data sets (including QuickDraw, ISOLET and 20 Newsgroups) and Kruskal–Wallis/Dunn and chi-square analyses, which the article does not use.

The `README.md`, `CITATION.cff` and `LICENSE` in this folder describe the thesis version and are kept unchanged for the record.
