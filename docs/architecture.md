# AMR-GenomeGuard – System Architecture

> Research and surveillance decision-support system.
> Not a clinical treatment recommender.

---

## Phase 1 Architecture (current)

```
BV-BRC AMR CSV (raw, read-only)
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
│   - deduplication           │  → data/processed/train.csv
│   - leakage exclusion       │  → data/processed/test.csv
│   - categorical encoding    │  → reports/preprocessing_report.json
│   - stratified 80/20 split  │
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
                ▼
┌─────────────────────────────┐
│   Evaluation Layer          │  evaluate.py
│   - accuracy / F1 / MCC     │  → reports/evaluation_report.json
│   - ROC-AUC / PR-AUC        │  → reports/model_comparison.csv
│   - confusion matrix plots  │  → reports/confusion_matrix_*.png
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│   Model Artifact            │  → models/artifacts/xgboost_phase1.pkl
│   XGBoost (calibrated)      │
└─────────────────────────────┘
```

---

## Target Scope (Phase 1)

| Dimension     | Value                       |
|---------------|-----------------------------|
| Organism      | Escherichia coli            |
| Antibiotic    | Ciprofloxacin               |
| Labels        | Resistant (1) / Susceptible (0) |
| Dataset       | BV-BRC BVBRC_genome_amr.csv |
| Model         | XGBoost (calibrated)        |
| Baselines     | Majority-class rule, Logistic Regression |

---

## Leakage Risk – Excluded Features

The BV-BRC dataset contains columns that encode the method used to generate the
phenotype label (e.g., `Computational Method = "SIR XGBoost Model"`). These are
excluded from all model features to prevent label leakage:

- `computational_method`
- `computational_method_version`
- `computational_method_performance`
- `evidence`
- `measurement`, `measurement_sign`, `measurement_value`, `measurement_unit`

---

## Later Phases (planned)

```
Phase 1 Model Artifact
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Phase 2 – Experiment Tracking & Versioning           │
│  MLflow model registry, DVC dataset versioning        │
│  dataset hash, feature schema version, model version  │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 3 – AutoML + Model Optimisation                │
│  FLAML / Optuna hyperparameter search                 │
│  class imbalance handling, threshold tuning           │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 4 – API Deployment                             │
│  FastAPI: /predict  /batch-predict  /health           │
│  /models/current  /monitoring/drift  /retrain         │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 5 – CLI Batch + Live Processing                │
│  amr-guard predict  batch  watch                      │
│  parallel workers, CSV / Excel / JSON export          │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 6 – Monitoring & Drift Detection               │
│  feature drift, prediction drift, performance decay   │
│  Evidently AI / custom Streamlit dashboard            │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 7 – Automated Retraining                       │
│  drift trigger → validate → train → compare           │
│  → promote only if better → log → alert               │
└───────────────────┬───────────────────────────────────┘
                    │
                    ▼
┌───────────────────────────────────────────────────────┐
│  Phase 8 – CI/CD + Dockerisation                      │
│  GitHub Actions: lint → test → build → push → deploy  │
│  Docker Compose: api + mlflow + dashboard + db        │
└───────────────────────────────────────────────────────┘
```

---

## Prediction Response (future API)

```json
{
  "sample_id": "ECOLI_001",
  "prediction": "Resistant",
  "probability": 0.87,
  "confidence": "High",
  "model_version": "ecoli-cipro-v1",
  "evidence": ["gyrA mutation", "qnr gene"],
  "warnings": []
}
```

---

## One-line Positioning

> AMR-GenomeGuard turns bacterial genome metadata into monitored, explainable,
> version-controlled AMR phenotype predictions, with CLI automation for
> high-throughput surveillance workflows.
