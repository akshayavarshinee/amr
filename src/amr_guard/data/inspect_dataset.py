"""
Dataset inspection: loads the raw BV-BRC CSV and produces a summary report
without modifying the original file.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BVBRC_COLUMNS = {
    "Taxon ID": "taxon_id",
    "Genome ID": "genome_id",
    "Genome Name": "genome_name",
    "Antibiotic": "antibiotic",
    "Resistant Phenotype": "resistant_phenotype",
    "Measurement": "measurement",
    "Measurement Sign": "measurement_sign",
    "Measurement Value": "measurement_value",
    "Measurement Unit": "measurement_unit",
    "Laboratory Typing Method": "lab_typing_method",
    "Laboratory Typing Method Version": "lab_typing_method_version",
    "Laboratory Typing Platform": "lab_typing_platform",
    "Vendor": "vendor",
    "Testing Standard": "testing_standard",
    "Testing Standard Year": "testing_standard_year",
    "Computational Method": "computational_method",
    "Computational Method Version": "computational_method_version",
    "Computational Method Performance": "computational_method_performance",
    "Evidence": "evidence",
    "Source": "source",
    "PubMed": "pubmed",
}

LEAKAGE_RISK_COLUMNS = [
    "computational_method",
    "computational_method_version",
    "computational_method_performance",
    "evidence",
    "measurement",
    "measurement_sign",
    "measurement_value",
    "measurement_unit",
]


def load_raw(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    df.rename(columns=BVBRC_COLUMNS, inplace=True)
    df.columns = [c.strip() for c in df.columns]
    return df


def inspect(csv_path: str | Path, report_dir: str | Path | None = None) -> dict:
    df = load_raw(csv_path)

    total_rows = len(df)
    col_nulls = {col: int(df[col].isna().sum() + (df[col] == "").sum()) for col in df.columns}

    label_col = "resistant_phenotype"
    label_counts = df[label_col].value_counts(dropna=False).to_dict()
    label_counts = {str(k): int(v) for k, v in label_counts.items()}

    antibiotic_counts = df["antibiotic"].value_counts(dropna=False).head(20).to_dict()
    antibiotic_counts = {str(k): int(v) for k, v in antibiotic_counts.items()}

    organism_counts = df["genome_name"].str.extract(r"^([A-Za-z]+ [a-z]+)")[0].value_counts().head(20).to_dict()
    organism_counts = {str(k): int(v) for k, v in organism_counts.items()}

    # cross-tab: organism x antibiotic x label
    df["organism"] = df["genome_name"].str.extract(r"^([A-Za-z]+ [a-z]+)")[0]
    scope_counts = (
        df[df[label_col].isin(["Resistant", "Susceptible"])]
        .groupby(["organism", "antibiotic", label_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(40)
        .to_dict(orient="records")
    )

    leakage_flags = {
        col: int(df[col].notna().sum()) for col in LEAKAGE_RISK_COLUMNS if col in df.columns
    }

    report = {
        "total_rows": total_rows,
        "total_columns": len(df.columns),
        "columns": list(df.columns),
        "null_counts": col_nulls,
        "label_distribution": label_counts,
        "top_antibiotics": antibiotic_counts,
        "top_organisms": organism_counts,
        "scope_sample_counts": scope_counts,
        "leakage_risk_columns_populated": leakage_flags,
        "leakage_note": (
            "Columns like computational_method and evidence likely encode the source "
            "of the phenotype label (e.g. 'SIR XGBoost Model'). These MUST be excluded "
            "from model features to prevent leakage."
        ),
    }

    if report_dir:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "dataset_inspection_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"Inspection report saved to {out / 'dataset_inspection_report.json'}")

    return report


if __name__ == "__main__":
    import sys

    csv = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\2504787\Downloads\BVBRC_genome_amr.csv"
    result = inspect(csv, report_dir="reports")
    print(f"Total rows: {result['total_rows']}")
    print(f"Label distribution: {result['label_distribution']}")
    print(f"Top antibiotics: {list(result['top_antibiotics'].keys())[:5]}")
