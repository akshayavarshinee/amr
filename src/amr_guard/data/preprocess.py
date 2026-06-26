"""
Preprocessing pipeline for BV-BRC AMR data.

Steps:
  1. Load and rename columns
  2. Filter to E. coli + ciprofloxacin (or user-supplied scope)
  3. Keep only Resistant / Susceptible labels; encode as 1 / 0
  4. Deduplicate on genome_id
  5. Extract organism genus/species from genome_name
  6. Build feature matrix from non-leaking metadata columns
  7. Stratified 80/20 train/test split
  8. Save artifacts to data/processed/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from amr_guard.data.inspect_dataset import load_raw, LEAKAGE_RISK_COLUMNS

LABEL_MAP = {"Resistant": 1, "Susceptible": 0}

# Columns that cannot be used as model features (leakage or always empty)
EXCLUDED_FEATURES = set(LEAKAGE_RISK_COLUMNS) | {
    "resistant_phenotype",
    "genome_name",
    "source",
    "pubmed",
    "taxon_id",
}

CATEGORICAL_FEATURES = ["antibiotic", "organism", "lab_typing_method", "testing_standard", "vendor"]
NUMERIC_FEATURES: list[str] = []  # extended later if MIC columns are populated


def _extract_organism(genome_name: pd.Series) -> pd.Series:
    return genome_name.str.extract(r"^([A-Za-z]+ [a-z]+)", expand=False).fillna("Unknown")


def _encode_categoricals(df: pd.DataFrame, fit_encoders: dict | None = None) -> tuple[pd.DataFrame, dict]:
    encoders: dict[str, LabelEncoder] = fit_encoders or {}
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("Unknown").astype(str)
        if col not in encoders:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "Unknown")
            if "Unknown" not in known:
                le.classes_ = np.append(le.classes_, "Unknown")
            df[col] = le.transform(df[col])
    return df, encoders


def run(
    csv_path: str | Path,
    organism_filter: str = "Escherichia coli",
    antibiotic_filter: str = "ciprofloxacin",
    output_dir: str | Path = "data/processed",
    random_state: int = 42,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_raw(csv_path)

    # Extract organism
    df["organism"] = _extract_organism(df["genome_name"])

    # Filter scope
    mask_organism = df["organism"].str.lower().str.startswith(organism_filter.lower())
    mask_antibiotic = df["antibiotic"].str.lower().str.strip() == antibiotic_filter.lower()
    df = df[mask_organism & mask_antibiotic].copy()

    # Keep only clean labels
    df = df[df["resistant_phenotype"].isin(LABEL_MAP.keys())].copy()

    # Deduplicate on genome_id; keep first occurrence
    before_dedup = len(df)
    df.drop_duplicates(subset=["genome_id"], keep="first", inplace=True)
    after_dedup = len(df)

    # Encode target
    df["label"] = df["resistant_phenotype"].map(LABEL_MAP)

    # Drop all leaking and irrelevant columns
    feature_cols = [c for c in df.columns if c not in EXCLUDED_FEATURES and c != "label" and c != "genome_id"]

    features_df = df[["genome_id"] + feature_cols + ["label"]].copy()
    features_df, encoders = _encode_categoricals(features_df)

    # Fill any remaining NaNs with 0 (absent / unknown)
    feature_matrix = features_df.drop(columns=["genome_id", "label"])
    feature_matrix.fillna(0, inplace=True)
    labels = features_df["label"]

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        feature_matrix, labels, test_size=0.2, random_state=random_state, stratify=labels
    )

    # Save artifacts
    df[["genome_id", "genome_name", "antibiotic", "organism", "resistant_phenotype", "label"]].to_csv(
        output_dir / "clean_dataset.csv", index=False
    )
    feature_matrix.assign(label=labels).to_csv(output_dir / "feature_matrix.csv", index=False)
    X_train.assign(label=y_train).to_csv(output_dir / "train.csv", index=False)
    X_test.assign(label=y_test).to_csv(output_dir / "test.csv", index=False)

    summary = {
        "organism_filter": organism_filter,
        "antibiotic_filter": antibiotic_filter,
        "rows_after_filter": before_dedup,
        "rows_after_dedup": after_dedup,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "features": list(feature_matrix.columns),
        "label_balance_train": y_train.value_counts().to_dict(),
        "label_balance_test": y_test.value_counts().to_dict(),
        "excluded_leakage_columns": list(LEAKAGE_RISK_COLUMNS),
    }

    import json
    with open(output_dir / "preprocessing_report.json", "w") as f:
        json.dump({k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in summary.items()}, f, indent=2)

    print(f"Preprocessing complete. Train: {len(X_train)}, Test: {len(X_test)}")
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "encoders": encoders,
        "summary": summary,
    }


if __name__ == "__main__":
    run(r"C:\Users\2504787\Downloads\BVBRC_genome_amr.csv")
