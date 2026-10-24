# Class-mixture dependence in PCA-based hidden-unit screening

Code, seed-level numerical results and figures for the article

> **Class-mixture dependence in PCA-based hidden-unit screening: reproducibility and minority-class performance**
> Elizabeth P. Chou and Pei-Ying Tsai (2026). Submitted to *Advances in Data Analysis and Classification*.

The study ranks the 128 first-layer units of a trained multilayer perceptron by a PCA score (absolute eigenvector coefficients weighted by explained-variance ratios), transfers the top-ranked units to a Siamese network and evaluates

1. what the score measures (its relation to activation variance and to redundant units);
2. how the screening target depends on the class composition of the scoring data, with the network held fixed;
3. the reproducibility of the ranking under class-stratified half-samples; and
4. pair-level and class-level performance after transfer, including minority-class recall, for the PCA score, a class-balanced (CB) variant, activation variance, NISP and random selection.

Data: Fashion-MNIST, Multiple Features (UCI) and Reuters-21578 (ApteMod), ten seeds per condition.

## Repository layout

| Path | Content |
|---|---|
| `code/` | JAX implementation used for every number in the article (see `code/README.md`) |
| `code/results/` | Seed-level outputs: retention CSVs, probe CSVs, split-half diagnostics, mixture-shift results and the 110 classifier checkpoints (`model_seed*.npz`) |
| `code/provenance/` | Execution logs, environment records, download manifest and validation records |
| `tools/` | Rebuild of all tables and figures from the saved outputs; independent verification script |
| `environment/` | Pinned package versions, OS, thread settings and data checksums |
| `figures/` | The four figures of the article (PDF) |
| `TABLES_AND_FIGURES.md` | Which script and which result files produce each table and figure |
| `PROVENANCE.md` | How the results were produced, what was and was not saved |
| `legacy/thesis_tensorflow/` | The original TensorFlow implementation from the master's thesis; **not** used for the article (see the note in that folder) |

## Quick start: rebuild the tables and figures (no training)

Requires Python 3.12 and the packages in `requirements.txt` (only NumPy, pandas, SciPy and Matplotlib are needed for this step).

```bash
python -m pip install -r requirements.txt
bash tools/rebuild_summaries.sh       # writes rebuilt/source_multifile/{tables,figures}
python tools/verify_summaries.py      # independent recomputation of the main comparisons
```

`rebuild_summaries.sh` copies `code/results` to `rebuilt/results` and regenerates every table (LaTeX rows), figure and statistical test. It trains no model and does not overwrite the saved outputs.

## Full rerun (training)

See `code/README.md`. A full rerun trains 100 classifiers and 10,640 Siamese networks on CPU. Use a new, empty results directory; existing seed files are skipped.

## Data

Prepared data are not redistributed. `code/download_data.py` downloads the public sources and checks their SHA-256 values (`code/data/download_manifest.json`, `environment/data_sources_and_raw_checksums.json`) and then runs `code/prep_data.py`. Fashion-MNIST is distributed by Zalando Research, Multiple Features through the UCI Machine Learning Repository (here via PMLB), and Reuters-21578 (ApteMod) through NLTK data; their own licences and terms of use apply.

## Citation

See `CITATION.cff`. Please cite the article once published.

## Licence

Code: MIT (`LICENSE`). Figures and numerical results in this repository: CC BY 4.0.
