# Provenance of the numerical results

## One run, one environment

All results in the article come from a single run of the code in `code/`, started on 2026-10-24 (UTC), in one software environment:

- Python 3.12.14 on Linux 6.18.55 (x86_64, glibc 2.39), CPU only
- jax 0.12.0, jaxlib 0.12.0, optax 0.2.8, numpy 2.3.5, scipy 1.17.0, scikit-learn 1.8.0, pandas 2.2.3, matplotlib 3.10.8
- `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`; workers pinned to four CPUs

The SHA-256 of `code/engine.py` recorded at the start of training (`environment/recorded_at_training_start.json`) equals that of the file in this repository.

| Component | Count |
|---|---|
| Classifiers (10 conditions × 10 seeds) | 100 |
| Primary Siamese fits (10 conditions × 10 seeds × 76) | 7,600 |
| Probe analysis (3 conditions; Siamese networks re-trained with the same seeds) | 2,280 |
| Multiple Features, standardisation fitted before subsampling (10 classifiers) | 760 |

Siamese training stopped early (fewer than 10 epochs) in 359 of the 7,600 primary fits (4.7%).

## Determinism checks

- Re-training the 100 classifiers in the same environment gave byte-identical parameter arrays (1,000 arrays compared).
- The 760 Multiple Features fits of each preprocessing order were identical in two runs.
- The probe runs re-train the Siamese networks; their prototype minority recall and balanced accuracy agree with the primary CSVs to within 1.2 × 10⁻¹⁶.

Bitwise agreement across hardware or library builds is not expected.

## What was not saved

- Final Siamese weights were not saved; only performance records are kept.
- The split-half index arrays were not saved during the run. `tools/reconstruct_indices.py` regenerates them from the same function, data and seeds; Spearman and Jaccard values recomputed from the reconstructed indices agree with the saved values to within 1.2 × 10⁻¹⁶.
- The value of `XLA_FLAGS` at training time was not recorded; the launch scripts did not set it.

## Recovery

The checkpoint of the Reuters (balanced) seed-4 classifier was found empty after the run. It was regenerated in a separate directory; the 76 Siamese fits of that seed were repeated and gave identical performance records and epochs. Logs are in `code/provenance/`.

## Earlier batch

During manuscript preparation, an earlier set of Siamese results was produced in a different software environment. It is not included here and is not used in the article; `code/provenance/audit.log` and `code/results/audit_fresh.json` record the comparison (the CB-minus-PCA minority-recall conclusions were identical in both).

## Relation to the thesis code

`legacy/thesis_tensorflow/` is the original TensorFlow implementation from the master's thesis. It was not used for any number in the article. See `legacy/thesis_tensorflow/NOTE.md` for the differences.
