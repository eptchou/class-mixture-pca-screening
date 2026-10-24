# Code

JAX re-implementation of the original TensorFlow pipeline, with the corrections described in the article (batch-normalisation parameters and statistics transferred to the Siamese subnetwork, identical training protocol for pruned and unpruned networks, preprocessing fitted on the retained training data only, Reuters minority class set to *crude*).

## Main settings

- First hidden layer: 128 units with batch normalisation and ReLU; ranking uses validation activations.
- Criteria: PCA score, class-balanced PCA score (CB; fixed full class set, within-class covariance with divisor `n_c - 1`, classes with fewer than two observations rejected), activation variance, NISP (one-step propagation from the second hidden layer, InfFS mixing parameter 0.2) and random selection.
- Retention sizes K in {1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 96} plus the unpruned reference K = 128.
- Ten seeds per condition. At a fixed seed and K, all criteria share the initialisation of newly introduced layers and the training pairs.
- Multiple Features: standardisation fitted after minority subsampling (`run_preprocessing_sensitivity.py before` runs the alternative order).
- Split-half reproducibility: ten complementary class-stratified splits per classifier; the first half receives `floor(n_c/2)` observations of each class.

## Files

| File | Purpose |
|---|---|
| `engine.py` | Data conditions, classifier, ranking scores, unit transfer, Siamese training and evaluation |
| `run_condition.py`, `jobA.sh`, `jobB.sh` | Primary fits for the ten conditions |
| `diagnostics.py`, `stability.py`, `run_stability.sh` | Split-half reproducibility, spectral checks, fixed-network mixture shift, score correlations, PCA-NISP overlap |
| `run_probe.py` | Class-weighted logistic-regression probe (re-trains the Siamese networks with the same seeds and checks the prototype metrics against the primary CSVs) |
| `run_preprocessing_sensitivity.py` | Multiple Features with standardisation fitted before subsampling |
| `aggregate.py`, `analysis_extra.py`, `revision_sensitivity.py` | Tables, figures and tests of the article |
| `audit_fresh_results.py` | Seed-level statistical audit and cardinality checks |
| `benchmark_timing.py` | Ranking times |
| `check_figures.py` | Independent check of plotted values (needs the data and checkpoints) |
| `download_data.py`, `prep_data.py` | Download with checksum verification; data preparation |

Environment variables `ADAC_DATA_DIR`, `ADAC_RESULTS_DIR` and `ADAC_PAPER_DIR` override the data, results and output directories (defaults are relative to this folder).

## Full rerun

```bash
python -m pip install -r ../requirements.txt
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python download_data.py                         # also runs prep_data.py
export ADAC_RESULTS_DIR="$PWD/rerun_results"      # must be new and empty
bash jobA.sh
bash jobB.sh
python diagnostics.py
python run_probe.py fmnist 0.1 fmnist_r0.1
python run_probe.py mfeat 0.1 mfeat_r0.1
python run_probe.py reuters_nat 1.0 reuters_nat
python run_preprocessing_sensitivity.py before
python benchmark_timing.py
python aggregate.py
python analysis_extra.py
python revision_sensitivity.py
python audit_fresh_results.py
```

Results are deterministic for a given seed and software environment; bitwise agreement across hardware or library builds is not expected.
