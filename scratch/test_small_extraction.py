"""Small 10-genome end-to-end verification script for Step 2."""

from amr_guard.data.preprocess import run as run_preprocess

def main():
    print("=" * 60)
    print("Step 2 Verification: Small 10-Genome Feature Extraction")
    print("=" * 60)

    res = run_preprocess(
        labels_csv="data/processed/genome_labels.csv",
        genomes_dir="data/raw/genomes",
        test_dir="data/raw/genomes_test",
        max_genomes=10,
    )

    summary = res["summary"]

    print("\nExtraction Summary:")
    print(f"  Genomes loaded: {summary['valid_fasta_genomes']}")
    print(f"  Missing FASTA:  {summary['missing_fasta_genomes']}")
    print(f"  Total contigs:  {summary['total_contigs']}")
    print(f"  Sequence length stats: {summary['sequence_length_stats']}")
    print(f"  Feature Matrix Shape:  ({summary['valid_fasta_genomes']}, {len(res['feature_names'])})")
    print(f"  K-mer Feature Count:   {len(res['feature_names'])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
