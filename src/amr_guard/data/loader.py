"""Genome dataset loader and manifest builder.

Matches Genome IDs from data/processed/genome_labels.csv to FASTA files in:
  1. data/raw/genomes/
  2. data/raw/genomes_test/ (fallback for testing)

Requirements:
- Each Genome ID corresponds to exactly ONE ML sample.
- All FASTA records/contigs for a genome are processed per contig without boundary concatenation.
- Detect missing FASTA files, empty files, malformed FASTA, duplicate IDs, conflicting labels.
- Produces a detailed manifest report without silently discarding missing genomes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

from amr_guard.data.fasta import parse_fasta

LABEL_MAP = {"Resistant": 1, "Susceptible": 0}


@dataclass
class GenomeSample:
    genome_id: str
    phenotype: str
    label: int
    fasta_path: Path
    sequence_length: int
    contig_count: int
    status: str  # "valid", "missing", "empty", "malformed"
    contigs: list[str] = field(default_factory=list)


def load_genome_dataset(
    labels_csv: str | Path = "data/processed/genome_labels.csv",
    genomes_dir: str | Path = "data/raw/genomes",
    test_dir: str | Path = "data/raw/genomes_test",
    max_genomes: int | None = None,
) -> tuple[list[GenomeSample], dict]:
    """Loads all genome samples matching labels_csv to FASTA sequence files.

    Returns:
        samples: list of GenomeSample objects
        report: dictionary summarizing valid, missing, empty, malformed, duplicate, and conflicting stats
    """
    labels_path = Path(labels_csv)
    if not labels_path.exists():
        raise FileNotFoundError(f"Genome labels file not found: {labels_path}")

    df = pd.read_csv(labels_path, dtype=str)
    df["Genome ID"] = df["Genome ID"].str.strip()
    df["phenotype"] = df["phenotype"].str.strip()

    if max_genomes is not None:
        df = df.head(max_genomes)

    total_rows = len(df)
    unique_ids = df["Genome ID"].nunique()

    # Pre-index directory files for O(1) lookup without hammering disk stat
    primary_dir = Path(genomes_dir)
    secondary_dir = Path(test_dir)

    fasta_index: dict[str, Path] = {}
    if primary_dir.exists():
        for p in primary_dir.glob("*.fna"):
            fasta_index[p.name] = p
    if secondary_dir.exists():
        for p in secondary_dir.glob("*.fna"):
            if p.name not in fasta_index:
                fasta_index[p.name] = p

    # Check duplicate Genome IDs and conflicting labels
    grouped_labels = df.groupby("Genome ID")["phenotype"].unique()
    conflicting_ids = set(grouped_labels[grouped_labels.apply(len) > 1].index)
    duplicate_ids = set(df[df["Genome ID"].duplicated()]["Genome ID"].unique())

    # Filter out conflicting genomes
    df_clean = df[~df["Genome ID"].isin(conflicting_ids)].drop_duplicates(subset=["Genome ID"]).copy()

    samples: list[GenomeSample] = []
    missing_count = 0
    empty_count = 0
    malformed_count = 0
    valid_count = 0

    seq_lengths = []
    contig_counts = []

    for _, row in df_clean.iterrows():
        g_id = row["Genome ID"]
        phenotype = row["phenotype"]
        label = LABEL_MAP.get(phenotype, -1)
        filename = f"{g_id}.fna"

        fasta_path = fasta_index.get(filename)

        if fasta_path is None:
            missing_count += 1
            samples.append(GenomeSample(
                genome_id=g_id,
                phenotype=phenotype,
                label=label,
                fasta_path=primary_dir / filename,
                sequence_length=0,
                contig_count=0,
                status="missing",
                contigs=[],
            ))
            continue

        if fasta_path.stat().st_size == 0:
            empty_count += 1
            samples.append(GenomeSample(
                genome_id=g_id,
                phenotype=phenotype,
                label=label,
                fasta_path=fasta_path,
                sequence_length=0,
                contig_count=0,
                status="empty",
                contigs=[],
            ))
            continue

        try:
            contigs, seq_len, c_count = parse_fasta(fasta_path)
            valid_count += 1
            seq_lengths.append(seq_len)
            contig_counts.append(c_count)
            samples.append(GenomeSample(
                genome_id=g_id,
                phenotype=phenotype,
                label=label,
                fasta_path=fasta_path,
                sequence_length=seq_len,
                contig_count=c_count,
                status="valid",
                contigs=contigs,
            ))
        except Exception:
            malformed_count += 1
            samples.append(GenomeSample(
                genome_id=g_id,
                phenotype=phenotype,
                label=label,
                fasta_path=fasta_path,
                sequence_length=0,
                contig_count=0,
                status="malformed",
                contigs=[],
            ))

    res_count = sum(1 for s in samples if s.status == "valid" and s.label == 1)
    sus_count = sum(1 for s in samples if s.status == "valid" and s.label == 0)

    report = {
        "total_labeled_genomes": total_rows,
        "unique_labeled_genomes": unique_ids,
        "conflicting_genomes_count": len(conflicting_ids),
        "duplicate_genomes_count": len(duplicate_ids),
        "valid_fasta_genomes": valid_count,
        "missing_fasta_genomes": missing_count,
        "empty_fasta_genomes": empty_count,
        "malformed_fasta_genomes": malformed_count,
        "resistant_genomes": res_count,
        "susceptible_genomes": sus_count,
        "total_contigs": int(sum(contig_counts)),
        "sequence_length_stats": {
            "mean": float(pd.Series(seq_lengths).mean()) if seq_lengths else 0.0,
            "min": int(pd.Series(seq_lengths).min()) if seq_lengths else 0,
            "max": int(pd.Series(seq_lengths).max()) if seq_lengths else 0,
            "total_bases": int(sum(seq_lengths)),
        },
        "contig_count_stats": {
            "mean": float(pd.Series(contig_counts).mean()) if contig_counts else 0.0,
            "min": int(pd.Series(contig_counts).min()) if contig_counts else 0,
            "max": int(pd.Series(contig_counts).max()) if contig_counts else 0,
        },
    }

    return samples, report
