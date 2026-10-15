"""Audit downloaded genomes cohort and generate cohort manifest."""

import pandas as pd
from pathlib import Path
from amr_guard.data.loader import load_genome_dataset

def main():
    print("=" * 60)
    print("Auditing downloaded FASTA genomes cohort...")
    print("=" * 60)

    samples, report = load_genome_dataset(
        labels_csv="data/processed/genome_labels.csv",
        genomes_dir="data/raw/genomes",
        test_dir="data/raw/genomes_test",
    )

    print("\nDataset Loading Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")

    valid_samples = [s for s in samples if s.status == "valid"]
    print(f"\nTotal Valid Usable Genomes: {len(valid_samples)}")

    manifest_data = [
        {
            "genome_id": s.genome_id,
            "phenotype": s.phenotype,
            "label": s.label,
            "fasta_path": str(s.fasta_path),
            "status": s.status,
            "sequence_length": s.sequence_length,
            "contig_count": s.contig_count,
        }
        for s in samples
    ]

    manifest_df = pd.DataFrame(manifest_data)
    manifest_path = Path("data/processed/genomic_cohort_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nCohort manifest saved to: {manifest_path}")

if __name__ == "__main__":
    main()
