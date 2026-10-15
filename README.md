# AMR-GenomeGuard

**MLOps-enabled antibiotic resistance phenotype prediction from BV-BRC bacterial AMR metadata.**

> Research and surveillance decision-support system. Not a clinical treatment recommender.

---

## Model Scope & Data Disclaimer

**Current Model Scope:**
- **Organism:** *Escherichia coli* (Taxon ID 562)
- **Antibiotic:** Ciprofloxacin
- **Dataset:** BV-BRC `BVBRC_genome_amr.csv` (235,525 raw records; ~105.8k unique *E. coli* ciprofloxacin isolate records)
- **Features Used:** Non-leaking laboratory typing metadata fields (`lab_typing_method`, `lab_typing_method_version`, `lab_typing_platform`, `testing_standard`, `testing_standard_year`, `vendor`).

> [!NOTE]
> The current MVP model operates on BV-BRC AMR record metadata. It does **NOT** learn directly from whole-genome FASTA DNA sequences or genomic variant features. Genome IDs and direct phenotype-encoding columns (`computational_method`, `measurement_*`, `evidence`) are strictly excluded to prevent data leakage.

---

## 3-Way Train / Validation / Test Methodology

To eliminate test-set leakage:
- **Train Set (70%):** Fits baseline and XGBoost candidate models.
- **Validation Set (15%):** Used for Optuna hyperparameter tuning, model candidate selection (static vs. tuned XGBoost), and decision threshold optimization.
- **Test Set (15%):** Kept completely untouched until final evaluation. Evaluated **exactly once** on the final promoted model artifact.

---

## Pipeline Overview

| Step | Output |
|------|--------|
| Dataset inspection | `reports/dataset_inspection_report.json` |
| Preprocessing & Dedup | `data/processed/clean_dataset.csv`, `train.csv`, `val.csv`, `test.csv`, `preprocessing_report.json` |
| Feature matrix | `data/processed/feature_matrix.csv` |
| Baseline models | Rule baseline, Logistic Regression |
| Primary model artifact | XGBoost bundle (model, encoders, feature_names, threshold, metadata) → `models/artifacts/xgboost_phase1.pkl` |
| Evaluation | `reports/evaluation_report.json`, `model_comparison.csv` |
| System architecture | `docs/architecture.md` |

---

## Quick Start

```bash
# Install dependencies
pip install -e .

# Run full end-to-end pipeline (Preprocessing -> Train -> Val -> Test -> Save Artifact)
python run_phase1.py

# Run all unit and integration tests
pytest -v

# Launch FastAPI prediction service
python run_api.py

# Run CLI batch prediction
amr-guard predict --input data/processed/clean_dataset.csv --output reports/cli_results.csv
```

---

## Project Structure

```
amr-genomeguard/
├── src/
│   └── amr_guard/
│       ├── api/               # FastAPI prediction service & schemas
│       ├── cli/               # CLI batch processing
│       ├── data/              # inspection, duplicate resolution, preprocessing & canonical feature prep
│       ├── models/            # training, 3-way evaluation, Optuna tuning, prediction & retrain
│       ├── monitoring/        # prediction logging & PSI drift detection
│       ├── tracking/          # optional MLflow integration
│       └── utils/             # configuration loader
├── tests/                     # unit & integration tests
├── data/
│   ├── raw/                   # original read-only files
│   └── processed/             # cleaned 3-way split data & feature matrices
├── models/artifacts/          # saved model bundle (.pkl)
├── reports/                   # evaluation reports, decision threshold, prediction log
├── frontend/                  # React / Vite monitoring dashboard
├── run_phase1.py              # Phase 1 end-to-end pipeline runner
├── run_api.py                 # FastAPI service entry point
└── pyproject.toml
```

---

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| `GET` | `/health` | Service liveness and model artifact status |
| `GET` | `/models/current` | Active model version, path, threshold, and registered name |
| `POST` | `/predict` | Single-sample prediction using canonical feature preparation |
| `POST` | `/batch-predict` | Batch predictions for multiple metadata rows |
| `GET` | `/monitoring/drift` | Feature PSI drift report comparing predictions against training set |
| `POST` | `/retrain` | Triggers background retrain into scratch workspace & promotes if candidate improves |

---

## Monitoring Dashboard

```bash
cd frontend
npm install
npm run build
```

---

## Retraining Safety

The automated retraining engine (`amr_guard.models.retrain`) runs preprocessing and candidate model training inside a isolated scratch directory. The incumbent live model is only overwritten if the candidate's test score exceeds the incumbent's score by at least `min_improvement` (defined in `params.yaml`). On promotion, the previous model is backed up to `xgboost_phase1_prev.pkl` and the API model cache is invalidated automatically.
