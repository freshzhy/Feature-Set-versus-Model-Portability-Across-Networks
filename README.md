# Attack-Specific Feature Signatures in Network Intrusion Detection: Feature-Set versus Model Portability Across Networks

Code and reference outputs for the paper

> H. Zhang, M. Wang. *Attack-Specific Feature Signatures in Network Intrusion Detection: Feature-Set versus Model Portability Across Networks.* Submitted to *Journal of Network and Computer Applications*, 2026.

## Repository layout

```
.
├── run.sh                       single entry point for the whole pipeline
├── requirements.txt
├── code/
│   ├── preprocess.py            NFv2 preprocessing (Section 3.1.1)
│   ├── rq1_signatures.py        RQ1: one-vs-rest XGBoost + TreeSHAP signatures
│   ├── rq1_distinctiveness.py   RQ1: intra-dataset distinctiveness (Table III, Fig. 3)
│   ├── rq1_stability.py         RQ1: signature stability across 30 seeds (Section 4.1.4)
│   ├── rq1_dedup_robustness.py  RQ1/RQ2 signatures on exact-deduplicated data (Table S6)
│   ├── rq2_similarity.py        RQ2: cross-dataset Jaccard / Spearman (Table IV)
│   ├── rq3_diagnostics.py       raw class counts, duplicate and IP-group diagnostics
│   ├── rq3_transfer.py          RQ3: feature-set and model portability, statistics
│   ├── rq3_jsd.py               Jensen–Shannon divergence as a predictor (Fig. 8)
│   ├── plot_figures_rq1_rq2.py  Figs. 1, 2, 4, 5 and S1
│   └── plot_figures_rq3.py      Figs. 6–9 and S2
├── data/
│   ├── raw/                     place the four NFv2 CSV files here (not redistributed)
│   └── processed/               written by the preprocessing step
└── results/                     reference outputs behind every table and figure
```

## Setup

The results in the paper were produced with **Python 3.12 and XGBoost 3.2.0** (`tree_method = hist`) on macOS (Apple Silicon).

```bash
./run.sh setup                   # creates .venv and installs requirements.txt
```

Download NF-UNSW-NB15-v2, NF-CSE-CIC-IDS2018-v2, NF-ToN-IoT-v2 and NF-BoT-IoT-v2 from https://staff.itee.uq.edu.au/marius/NIDS_datasets/ and save them as `data/raw/<dataset>.csv`.

## Reproducing the results

```bash
./run.sh preprocess              # data/raw  -> data/processed
./run.sh rq1                     # signatures, distinctiveness, stability
./run.sh rq2                     # cross-dataset similarity
./run.sh rq3                     # diagnostics, transfer experiment (30 seeds), JSD
./run.sh robustness              # deduplication, selection protocol, scenario-feature ablation
./run.sh figures                 # results/figures/
```

`./run.sh all` runs every step in order. The RQ3 and robustness steps take several hours; run them in the background with `./run.sh bg rq3` (or `bg robustness`, `bg all`), check progress with `./run.sh status`, and stop with `kill $(cat results/logs/<step>.pid)`. Restarting a step resumes from the completed tasks. On macOS the background job keeps the machine awake with `caffeinate`.

Parallelism of the RQ3 experiments is set with the `WORKERS` and `THREADS` environment variables (default: `(cores − 2)/4` parallel tasks with 4 XGBoost threads each), e.g. `WORKERS=4 THREADS=4 ./run.sh bg rq3`. Every script also documents its options under `python code/<script>.py --help`.

`rq3_transfer.py` has the subcommands `prep`, `run`, `status` and `aggregate`. The robustness analyses are selected with `--dedup` (exact-deduplicated data) or `--variant trainshap | reselect | noscenario` (SHAP on the training split; re-selection on deduplicated data; ports, TTL and TCP-window features removed).

## Where each reported result comes from

| Paper item | File(s) in `results/` |
|---|---|
| Table I (dataset characteristics) | `preprocess_logs/*.log`, `rq3_diagnostics/raw_class_counts.json` |
| Fig. 1, Table S1 (per-class performance) | `rq1_signatures/*_rq1_model_metrics.csv` |
| Fig. 2, Fig. S1 (signatures) | `rq1_signatures/*_rq1_shap_matrix.csv`, `*_rq1_top10_signatures.csv` |
| Table III, Fig. 3 (intra-dataset distinctiveness) | `rq1_distinctiveness/summary.csv`, `*_distinctiveness_matrix.csv` |
| Section 4.1.4 (stability across seeds) | `rq1_stability/signature_stability_summary.csv` |
| Table IV, Figs. 4–5 (cross-dataset similarity) | `rq2_similarity/rq2_transferability_report.csv`, `rq2_jaccard_*.csv`, `rq2_spearman_*.csv` |
| Table V-a (within-dataset compression) | `rq3_main/config_summary.csv` (conditions Full-40, Target-10) |
| Table V-b, Section 4.3.2 (configuration-level tests) | `rq3_main/config_level_tests.csv` |
| Table S2, Figs. 6 and S2 (per-configuration results) | `rq3_main/config_summary.csv`, `per_config_tests.csv`, `raw.csv` |
| Fig. 7 (environment), Fig. 8 (predictors) | `rq3_main/per_config_tests.csv`, `rq3_jsd/jsd_matrix.csv`, `rq2_similarity/` |
| Table VI, Fig. 9 (zero-shot transfer) | `rq3_main/config_summary.csv` (`*-Direct` conditions), `per_config_tests.csv` |
| Table S3 (deduplicated data) | `rq3_dedup/config_level_tests.csv` |
| Table S4 (rows retained after deduplication) | `rq3_dedup/dedup_report.csv` |
| Table S5 (selection protocol, scenario features) | `rq3_trainshap/`, `rq3_dedup_reselect/`, `rq3_noscenario/` |
| Table S6, Section 4.3.5 (signatures on deduplicated data) | `rq1_dedup_robustness/signature_dedup.csv`, `intra_distinctiveness.csv`, `cross_dataset.csv` |
| All figures as published | `figures/Fig*.pdf` / `.png` |

Each `summary.txt` is a human-readable digest of the corresponding experiment. In the result tables, `std` is the standard test set and `nat` the prevalence-matched test set.

## Notes on reproducibility

- Seeds 42–71 are used for all repeated splits; robustness analyses use seeds 42–51.
- XGBoost fits use row and column subsampling, so a different thread configuration can change individual fits slightly. The paper quantifies this refit variability for the RQ1 signatures (Section 4.1.4); aggregate conclusions are unaffected.
- Large caches (`results/**/cache/`, per-task `parts/`) and the processed data are excluded from the repository and are regenerated by the pipeline.
