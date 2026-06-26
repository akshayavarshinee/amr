"""Unit tests for the preprocessing pipeline."""

import io
import textwrap

import pandas as pd
import pytest

from amr_guard.data.preprocess import LABEL_MAP, LEAKAGE_RISK_COLUMNS, _extract_organism


SAMPLE_CSV = textwrap.dedent("""\
    Taxon ID,Genome ID,Genome Name,Antibiotic,Resistant Phenotype,Measurement,Measurement Sign,Measurement Value,Measurement Unit,Laboratory Typing Method,Laboratory Typing Method Version,Laboratory Typing Platform,Vendor,Testing Standard,Testing Standard Year,Computational Method,Computational Method Version,Computational Method Performance,Evidence,Source,PubMed
    562,562.1,Escherichia coli ECBS001,ciprofloxacin,Resistant,,,,,,,,,,,"SIR XGBoost Model","20250225","F1 score: 0.98","Computational Method",,
    562,562.2,Escherichia coli ECBS002,ciprofloxacin,Susceptible,,,,,,,,,,,"SIR XGBoost Model","20250225","F1 score: 0.98","Computational Method",,
    562,562.3,Escherichia coli ECBS003,ciprofloxacin,Resistant,,,,,,,,,,,"SIR XGBoost Model","20250225","F1 score: 0.98","Computational Method",,
    562,562.4,Escherichia coli ECBS004,ciprofloxacin,Intermediate,,,,,,,,,,,"SIR XGBoost Model","20250225","F1 score: 0.98","Computational Method",,
    562,562.5,Klebsiella pneumoniae KP001,ciprofloxacin,Resistant,,,,,,,,,,,"SIR XGBoost Model","20250225","F1 score: 0.98","Computational Method",,
""")


def _load_sample() -> pd.DataFrame:
    from amr_guard.data.inspect_dataset import BVBRC_COLUMNS
    df = pd.read_csv(io.StringIO(SAMPLE_CSV), dtype=str)
    df.rename(columns=BVBRC_COLUMNS, inplace=True)
    return df


def test_label_map_complete():
    assert "Resistant" in LABEL_MAP
    assert "Susceptible" in LABEL_MAP
    assert LABEL_MAP["Resistant"] == 1
    assert LABEL_MAP["Susceptible"] == 0


def test_extract_organism():
    names = pd.Series([
        "Escherichia coli ECBS001",
        "Klebsiella pneumoniae KP001",
        "Staphylococcus aureus SA01",
    ])
    result = _extract_organism(names)
    assert result.iloc[0] == "Escherichia coli"
    assert result.iloc[1] == "Klebsiella pneumoniae"
    assert result.iloc[2] == "Staphylococcus aureus"


def test_intermediate_labels_excluded():
    df = _load_sample()
    df["label"] = df["resistant_phenotype"].map(LABEL_MAP)
    clean = df[df["label"].notna()]
    assert "Intermediate" not in clean["resistant_phenotype"].values


def test_non_ecoli_excluded():
    df = _load_sample()
    df["organism"] = _extract_organism(df["genome_name"])
    ecoli = df[df["organism"].str.lower().str.startswith("escherichia coli")]
    assert "Klebsiella pneumoniae" not in ecoli["organism"].values


def test_leakage_columns_defined():
    assert len(LEAKAGE_RISK_COLUMNS) > 0
    assert "computational_method" in LEAKAGE_RISK_COLUMNS
    assert "evidence" in LEAKAGE_RISK_COLUMNS


def test_no_duplicate_genome_ids():
    df = _load_sample()
    df["organism"] = _extract_organism(df["genome_name"])
    df_clean = df[df["resistant_phenotype"].isin(LABEL_MAP.keys())]
    deduped = df_clean.drop_duplicates(subset=["genome_id"])
    assert deduped["genome_id"].nunique() == len(deduped)
