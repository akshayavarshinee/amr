# AMR-GenomeGuard – System Architecture

> Research and surveillance decision-support system.
> Not a clinical treatment recommender.

---

## Phase 1 Architecture (3-Way Evaluation & Canonical Prep)

```
BV-BRC AMR CSV (raw, read-only: 235,525 rows)
        │
        ▼
┌─────────────────────────────┐
│   Dataset Inspection        │  inspect_dataset.py
│   - schema validation       │  → reports/dataset_inspection_report.json
│   - label distribution      │
│   - leakage risk flags      │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Preprocessing Pipeline    │  preprocess.py
│   - organism/ab filter      │  → data/processed/clean_dataset.csv
│   - label normalization     │  → data/processed/feature_matrix.csv
│   - conflict policy & dedup │  → data/processed/train.csv
│   - leakage exclusion       │  → data/processed/val.csv
│   - categorical encoding    │  → data/processed/test.csv
│   - 3-way stratified split  │  → reports/preprocessing_report.json
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Feature Matrix            │
│   Metadata columns only     │
│   (leaking columns excluded)│
└────────────┬────────────────┘
             │
      ┌──────┼──────────────┐
      │      │              │
      ▼      ▼              ▼
┌────────┐ ┌──────────┐ ┌──────────┐
│ Rule   │ │Logistic  │ │XGBoost   │
│Baseline│ │Regression│ │(primary) │  train.py
└────┬───┘ └────┬─────┘ └────┬─────┘
     │          │             │
     └──────────┼─────────────┘
                │
                ▼ (Tuned & Selected on Validation Set)
┌─────────────────────────────┐
│   Evaluation Layer          │  evaluate.py
│   - accuracy / F1 / MCC     │  → reports/evaluation_report.json
│   - ROC-AUC / PR-AUC        │  → reports/model_comparison.csv
│   - confusion matrix plots  │  → reports/confusion_matrix_*.png
│   - ONE untouched Test eval │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Model Artifact Bundle     │  → models/artifacts/xgboost_phase1.pkl
│   (XGBoost, Encoders,       │
│    Feature Order, Threshold)│
└─────────────────────────────┘
```

---

## Target Scope

| Dimension     | Value                       |
|---------------|-----------------------------|
| Organism      | *Escherichia coli*          |
| Antibiotic    | Ciprofloxacin               |
| Labels        | Resistant (1) / Susceptible (0) |
| Dataset       | BV-BRC `BVBRC_genome_amr.csv` |
| Model         | XGBoost (calibrated)        |
| Baselines     | Majority-class rule, Logistic Regression |
| Data Split    | Train (70%) / Validation (15%) / Test (15%) |

---

## Leakage Risk – Excluded Features

The BV-BRC dataset contains columns that encode the outcome or computational estimation of resistance. These are excluded from all model features to prevent label leakage:

- `computational_method`
- `computational_method_version`
- `computational_method_performance`
- `evidence`
- `measurement`, `measurement_sign`, `measurement_value`, `measurement_unit`
- `resistant_phenotype` (target)
- `genome_name`, `source`, `pubmed`, `taxon_id`

---

## Positioning Statement

> AMR-GenomeGuard turns bacterial genome metadata into monitored, explainable, version-controlled AMR phenotype predictions, with CLI automation for high-throughput surveillance workflows.
