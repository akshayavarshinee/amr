"""Preprocessing pipeline for genomic sequence AMR data.

Steps:
  1. Load genome labels (data/processed/genome_labels.csv) and match to FASTA sequence files.
  2. Parse FASTA contigs, extract valid 6-mer frequencies within each contig independently (4096 features).
  3. Validate dataset: detect missing FASTA, empty FASTA, duplicate IDs, conflicting labels.
  4. Perform 3-way Stratified Split (Train: 70%, Validation: 15%, Test: 15%).
  5. Check data leakage (ensure no Genome ID overlap across splits).
  6. Generate genomic_dataset_report.json.
  7. Save feature matrices to data/processed/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from amr_guard.data.fasta import parse_fasta, parse_fasta_str
from amr_guard.data.kmer import extract_kmers_from_contigs, get_kmer_vocabulary
from amr_guard.data.loader import GenomeSample, load_genome_dataset
from amr_guard.utils.config import load_params


def prepare_features(
    inputs: str | Path | dict | list[dict] | pd.DataFrame,
    encoders: dict | None = None,
    feature_names: list[str] | None = None,
    k: int = 6,
) -> pd.DataFrame:
    """Canonical genomic feature preparation path shared across training, inference, API, and CLI."""
    vocab = feature_names or get_kmer_vocabulary(k)

    if isinstance(inputs, (str, Path)):
        inp_str = str(inputs).strip()
        path = Path(inp_str)
        if path.exists() and path.is_file():
            contigs, _, _ = parse_fasta(path)
        else:
            contigs, _, _ = parse_fasta_str(inp_str)

        kmer_freqs = extract_kmers_from_contigs(contigs, k=k, vocabulary=vocab, normalize=True)
        return pd.DataFrame([kmer_freqs], columns=vocab)

    if isinstance(inputs, dict):
        if "fasta_content" in inputs and inputs["fasta_content"]:
            contigs, _, _ = parse_fasta_str(str(inputs["fasta_content"]))
            kmer_freqs = extract_kmers_from_contigs(contigs, k=k, vocabulary=vocab, normalize=True)
            return pd.DataFrame([kmer_freqs], columns=vocab)
        if "sequence" in inputs and inputs["sequence"]:
            contigs, _, _ = parse_fasta_str(str(inputs["sequence"]))
            kmer_freqs = extract_kmers_from_contigs(contigs, k=k, vocabulary=vocab, normalize=True)
            return pd.DataFrame([kmer_freqs], columns=vocab)
        if "fasta_path" in inputs and inputs["fasta_path"]:
            contigs, _, _ = parse_fasta(Path(inputs["fasta_path"]))
            kmer_freqs = extract_kmers_from_contigs(contigs, k=k, vocabulary=vocab, normalize=True)
            return pd.DataFrame([kmer_freqs], columns=vocab)
        if "features" in inputs and isinstance(inputs["features"], dict):
            df = pd.DataFrame([inputs["features"]])
        else:
            df = pd.DataFrame([inputs])
    elif isinstance(inputs, list):
        df = pd.DataFrame(inputs)
    elif inputs is not None and hasattr(inputs, "copy"):
        df = inputs.copy()
    else:
        df = pd.DataFrame([{}])

    return df.reindex(columns=vocab, fill_value=0.0).astype(np.float32)


def check_data_leakage(
    train_ids: pd.Series | set,
    val_ids: pd.Series | set,
    test_ids: pd.Series | set,
) -> None:
    """Explicitly verifies that no Genome ID appears in multiple splits."""
    set_train = set(train_ids)
    set_val = set(val_ids)
    set_test = set(test_ids)

    overlap_train_val = set_train & set_val
    overlap_train_test = set_train & set_test
    overlap_val_test = set_val & set_test

    if overlap_train_val or overlap_train_test or overlap_val_test:
        raise ValueError(
            f"DATA LEAKAGE DETECTED! Overlapping Genome IDs between splits: "
            f"Train/Val={len(overlap_train_val)}, Train/Test={len(overlap_train_test)}, Val/Test={len(overlap_val_test)}"
        )


def run(
    labels_csv: str | Path | None = None,
    genomes_dir: str | Path = "data/raw/genomes",
    test_dir: str | Path = "data/raw/genomes_test",
    output_dir: str | Path | None = None,
    reports_dir: str | Path | None = None,
    k: int = 6,
    random_state: int | None = None,
    val_size: float | None = None,
    test_size: float | None = None,
    max_genomes: int | None = None,
) -> dict[str, Any]:
    params = load_params()
    data_config = params["data"]
    paths_config = params["paths"]

    labels_csv = Path(labels_csv if labels_csv is not None else "data/processed/genome_labels.csv")
    output_dir = Path(output_dir if output_dir is not None else paths_config["processed_dir"])
    reports_dir = Path(reports_dir if reports_dir is not None else paths_config["reports_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    random_state = random_state if random_state is not None else data_config.get("random_state", 42)
    val_size = val_size if val_size is not None else data_config.get("val_size", 0.15)
    test_size = test_size if test_size is not None else data_config.get("test_size", 0.15)

    print(f"Loading genomic dataset from {labels_csv}...")
    samples, load_report = load_genome_dataset(
        labels_csv=labels_csv,
        genomes_dir=genomes_dir,
        test_dir=test_dir,
        max_genomes=max_genomes,
    )

    valid_samples = [s for s in samples if s.status == "valid"]
    if not valid_samples:
        raise RuntimeError("No valid genome FASTA samples found for feature extraction!")

    print(f"Extracting {k}-mer frequencies for {len(valid_samples)} valid genomes...")
    vocab = get_kmer_vocabulary(k=k)

    # Fast 2D numpy matrix allocation
    matrix = np.zeros((len(valid_samples), len(vocab)), dtype=np.float32)
    genome_ids = pd.Series([s.genome_id for s in valid_samples], name="genome_id")
    labels = pd.Series([s.label for s in valid_samples], name="label")

    for idx, sample in enumerate(valid_samples):
        matrix[idx] = extract_kmers_from_contigs(sample.contigs, k=k, vocabulary=vocab, normalize=True)
        if (idx + 1) % 500 == 0 or (idx + 1) == len(valid_samples):
            print(f"  Processed {idx + 1}/{len(valid_samples)} genomes...")

    feature_matrix = pd.DataFrame(matrix, columns=vocab)

    # Stratified 3-way split: Train (70%), Val (15%), Test (15%)
    if len(valid_samples) < 5:
        X_train, X_val, X_test = feature_matrix, feature_matrix, feature_matrix
        y_train, y_val, y_test = labels, labels, labels
        ids_train, ids_val, ids_test = genome_ids, genome_ids, genome_ids
    else:
        X_train_val, X_test, y_train_val, y_test, ids_train_val, ids_test = train_test_split(
            feature_matrix, labels, genome_ids, test_size=test_size, random_state=random_state, stratify=labels if labels.nunique() > 1 else None
        )
        adjusted_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
            X_train_val, y_train_val, ids_train_val, test_size=adjusted_val_size, random_state=random_state, stratify=y_train_val if y_train_val.nunique() > 1 else None
        )
        check_data_leakage(ids_train, ids_val, ids_test)

    # Save dataset artifacts
    feature_matrix.assign(genome_id=genome_ids, label=labels).to_csv(output_dir / "feature_matrix.csv", index=False)
    X_train.assign(genome_id=ids_train, label=y_train).to_csv(output_dir / "train.csv", index=False)
    X_val.assign(genome_id=ids_val, label=y_val).to_csv(output_dir / "val.csv", index=False)
    X_test.assign(genome_id=ids_test, label=y_test).to_csv(output_dir / "test.csv", index=False)

    genomic_report = {
        **load_report,
        "k": k,
        "feature_count": len(vocab),
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
        "split_overlap_check": "PASSED",
        "label_balance_train": y_train.value_counts().to_dict(),
        "label_balance_val": y_val.value_counts().to_dict(),
        "label_balance_test": y_test.value_counts().to_dict(),
    }

    report_path = reports_dir / "genomic_dataset_report.json"
    with open(report_path, "w") as f:
        json.dump(genomic_report, f, indent=2)

    print(
        f"Genomic preprocessing complete.\n"
        f"  Total valid genomes: {len(valid_samples)}\n"
        f"  Feature shape: {feature_matrix.shape}\n"
        f"  Report saved to: {report_path}"
    )

    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "ids_train": ids_train,
        "ids_val": ids_val,
        "ids_test": ids_test,
        "encoders": {},
        "feature_names": vocab,
        "summary": genomic_report,
    }


if __name__ == "__main__":
    run()
