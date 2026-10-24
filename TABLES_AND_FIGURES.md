# Tables and figures

Numbering follows the submitted manuscript. All generators read `code/results/` (or `$ADAC_RESULTS_DIR`) and write to `$ADAC_PAPER_DIR`; `tools/rebuild_summaries.sh` runs them in order.

| Item | Content | Generator | Result files |
|---|---|---|---|
| Table 1 | Data conditions | `code/aggregate.py` | design quantities from `code/engine.py` and `code/prep_data.py` |
| Table 2 | Network architectures | hand-written; implemented in `code/engine.py` | none |
| Table 3 | Unpruned networks (K = 128) | `code/aggregate.py` | `*/retention_seed*.csv` (rows `criterion == Full`) |
| Table 4 | Split-half reproducibility | `code/aggregate.py` | `*/stab_split_seed*.csv`, `*/stab_nogueira_seed*.csv` |
| Table 5 | Averages over K <= 32 | `code/aggregate.py` | `*/retention_seed*.csv` |
| Table 6 | Multi-ratio reproducibility and Friedman tests | `code/aggregate.py`, `code/analysis_extra.py` | `fmnist_r*/stab_split*.csv`, `*/stab_nogueira_seed*.csv` |
| Table 7 | Sensitivity to the range of K | `code/analysis_extra.py` | `*/retention_seed*.csv` |
| Table 8 | All paired comparisons with PCA | `code/analysis_extra.py` | `*/retention_seed*.csv` |
| Table 9 | Random versus stratified halves (PCA) | `code/analysis_extra.py` | `*/stab_stratified.csv` |
| Table 10 | Logistic-regression probe | `code/analysis_extra.py` | `fmnist_r0.1/`, `mfeat_r0.1/`, `reuters_nat/` `probe_seed*.csv` |
| Table 11 | Signed-rank versus sign tests | `tools/generate_sign_table.py` | `*/retention_seed*.csv` |
| Table 12 | Raw and baseline-adjusted retention | `code/revision_sensitivity.py` | `*/retention_seed*.csv` |
| Table 13 | Order of standardisation and subsampling (MF) | `code/revision_sensitivity.py` | `mfeat_r0.1/` and `sensitivity/mfeat_before/` `retention_seed*.csv` |
| Figure 1 | Class-mixture shift with a fixed network | `code/aggregate.py` | `fmnist_bal/mixture_shift.csv`, `fmnist_bal/diagnostics_seed*.json` |
| Figure 2 | Retention curves, imbalanced conditions | `code/aggregate.py` | `fmnist_r0.1/`, `mfeat_r0.1/`, `reuters_nat/` `retention_seed*.csv` |
| Figure 3 | Multi-ratio summaries | `code/aggregate.py` | `fmnist_r*/retention_seed*.csv` |
| Figure 4 | Retention curves, balanced conditions | `code/aggregate.py` | `fmnist_bal/`, `mfeat_bal/`, `reuters_bal/` `retention_seed*.csv` |

Numbers quoted in the text come from `code/results/key_numbers*.json`, the tables above, the spectral and overlap files (`spectral_check.csv`, `overlap_summary.csv`, `nisp_weight_diagnostic.csv`) and `tools/verify_summaries.py`. The last script also computes two derived quantities reported in Section 4.5: the number of positive seed-level CB-minus-PCA differences in minority recall (61 of 70) and the change in the mean recall of the non-minority classes, (C·ΔBA − Δminority recall)/(C − 1).

## Rounding

Values are rounded to three decimals from binary floating-point means. Where an exact mean ends in 5 at the fourth decimal, the printed value can differ from decimal rounding. The one case in the manuscript is the unpruned Multiple Features (ρ = 0.1) balanced accuracy, whose exact mean is 1967/2000 = 0.9835; the manuscript reports 0.984, whereas `aggregate.py` prints 0.983.
