# CBC anemia classification: data audit and corrected statistical benchmark

Code to reproduce all results in:

> Nithya S, Sasikala S. *Oversampling Leakage and Threshold-Derived Labels Inflate Reported Accuracy in CBC-Based Anemia Classification: A Data Audit and Corrected Statistical Benchmark.* (under review)

## Data
The dataset is **not** redistributed here. Download `diagnosed_cbc_data_v4.csv` from the Kaggle dataset
"Anemia Types Classification" (E. Aboelnaga): https://www.kaggle.com/datasets/ehababoelnaga/anemia-types-classification
and place it in this folder. The script prints the file's SHA-256 so you can confirm you have the same version.

Expected SHA-256: `f7dee8d4fe500aec6e8e4bfbeb6df6b83bc86bef03d540721e0aeaad2759fcdb`

## Run
```bash
pip install -r requirements.txt
python anemia_reanalysis.py --csv diagnosed_cbc_data_v4.csv --seed 123 --out results
```
Add `--fast` for a quick 5×2 smoke test. A full run takes roughly 30–60 minutes on a laptop (nested tuning is the slowest step).

## Outputs (folder `results/`) and where they appear in the paper
| File | Paper |
|---|---|
| `cleaning_log.json`, `step2_audit_table.csv` | Section 3.2, Supplementary Table S1 |
| `imputed_flag_by_class.csv` | Figure 3, chi-square test |
| `cv_summary_seed20260815.csv`, `pairwise_tests_seed20260815.csv` | Tables 4–6 |
| `cv_summary_seed123.csv`, `pairwise_tests_seed123.csv` | Supplementary Tables S2–S3 |
| `per_class_RandomForest_seed123.csv`, `fig1_confusion_RandomForest.png` | Table 7, Figure 1 |
| `ablation_data_quality.csv` | Table 8, Figure 2 |
| `shallow_tree_recoverability.csv`, `tree_depth5_rules.txt` | Table 9, Figure 4 |
| `nested_cv_tuned_baselines.csv` | Table 9 (tuned baselines) |
| `tree_vs_ensembles_summary.csv`, `tree_vs_ensembles_tests.csv` | Section 4.6 (depth-5 tree vs ensembles) |
| `hyperparameters_and_versions.json` | Software versions and model settings |

Seeds: primary partitions 20260815; secondary partitions, ablation, trees and tuning 123.

## Licence
Code: MIT (see `LICENSE`). The dataset remains under its own Kaggle licence.
