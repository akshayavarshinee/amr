# AMR-GenomeGuard

**MLOps-enabled antibiotic resistance phenotype prediction from bacterial genome data.**

> Research and surveillance decision-support system. Not a clinical treatment recommender.

---

## Problem

Existing AMR tools (CARD/RGI, AMRFinderPlus, ResFinder) detect known resistance genes and mutations.
They do not provide: phenotype prediction, uncertainty quantification, dataset/model version tracking,
drift monitoring, or audit logs.

AMR-GenomeGuard fills this gap: a learning system that predicts resistance phenotypes,
explains its evidence, tracks versions, monitors drift, and supports high-throughput batch processing.

---

## Phase 1 – Completed

| Step | Output |
|------|--------|
| Dataset inspection | `reports/dataset_inspection_report.json` |
| Preprocessing | `data/processed/clean_dataset.csv`, `train.csv`, `test.csv` |
| Feature matrix | `data/processed/feature_matrix.csv` |
| Baseline models | Rule baseline, Logistic Regression |
| Primary model | XGBoost (calibrated) → `models/artifacts/xgboost_phase1.pkl` |
| Evaluation | `reports/evaluation_report.json`, `model_comparison.csv` |
| Architecture doc | `docs/architecture.md` |

**Scope:** E. coli + Ciprofloxacin resistance prediction
**Dataset:** BV-BRC `BVBRC_genome_amr.csv` (89,033 rows)

---

## Quick Start

```bash
# Create environment (uv)
uv venv
uv pip install -e .

# Run full Phase 1 pipeline
python run_phase1.py --csv path/to/BVBRC_genome_amr.csv

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
amr-genomeguard/
├── src/
│   └── amr_guard/
│       ├── data/
│       │   ├── inspect_dataset.py   # schema inspection, leakage flags
│       │   └── preprocess.py        # filtering, encoding, train/test split
│       ├── models/
│       │   ├── train.py             # rule baseline, logistic regression, XGBoost
│       │   ├── evaluate.py          # metrics, confusion matrix plots
│       │   └── predict.py           # single and batch inference
│       └── utils/
├── tests/
│   ├── test_preprocessing.py
│   └── test_models.py
├── data/
│   ├── raw/                         # original files (not committed)
│   └── processed/                   # derived artifacts
├── models/
│   └── artifacts/                   # saved model files
├── reports/                         # evaluation reports and plots
├── docs/
│   └── architecture.md              # full system architecture
├── run_phase1.py                    # end-to-end pipeline runner
└── pyproject.toml
```

---

## Planned Phases

| Phase | Focus |
|-------|-------|
| 2 | Experiment tracking (MLflow), data versioning (DVC) |
| 3 | AutoML (FLAML/Optuna), model optimisation |
| 4 | FastAPI prediction service |
| 5 | CLI batch processing, CSV/Excel export |
| 6 | Monitoring + drift detection dashboard |
| 7 | Automated retraining pipeline |
| 8 | CI/CD (GitHub Actions) + Dockerisation |
