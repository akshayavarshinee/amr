"""Stage 1: Streaming Genomic Feature Extraction & Verification Script.

Processes bacterial genome FASTAs in a memory-safe, stream-wise fashion.
Features:
- k = 6, exact 4096-element array per genome (float32)
- Direct base-to-integer bit-shift encoding (zero string dict overhead)
- Contigs processed independently (no contig boundary concatenation)
- Ignores non-ACGT bases
- Normalizes counts to sum to 1.0 per genome
- Disk-backed feature matrix (np.lib.format.open_memmap / npy)
- Full checkpoint and resume support
"""

import sys
import os
from pathlib import Path
import gc
import numpy as np
import pandas as pd

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from amr_guard.data.fasta import parse_fasta

# Lookup array mapping ASCII byte codes to integer 0..3 (A=0, C=1, G=2, T=3, everything else=255)
BYTE_MAP = np.full(256, 255, dtype=np.uint8)
BYTE_MAP[ord('A')] = 0
BYTE_MAP[ord('a')] = 0
BYTE_MAP[ord('C')] = 1
BYTE_MAP[ord('c')] = 1
BYTE_MAP[ord('G')] = 2
BYTE_MAP[ord('g')] = 2
BYTE_MAP[ord('T')] = 3
BYTE_MAP[ord('t')] = 3

def extract_genome_kmers_fast(fasta_path: Path) -> tuple[np.ndarray, int]:
    """Extracts normalized 4096 6-mer counts using ultra-fast vectorized numpy byte operations."""
    contigs, _, _ = parse_fasta(fasta_path)
    counts = np.zeros(4096, dtype=np.float64)
    total_valid = 0

    for contig in contigs:
        c_len = len(contig)
        if c_len < 6:
            continue
        
        # Convert string to numpy ASCII bytes array
        seq_bytes = np.frombuffer(contig.encode('ascii'), dtype=np.uint8)
        encoded = BYTE_MAP[seq_bytes]
        
        # Form 6-mer window matrices
        w0 = encoded[:-5]
        w1 = encoded[1:-4]
        w2 = encoded[2:-3]
        w3 = encoded[3:-2]
        w4 = encoded[4:-1]
        w5 = encoded[5:]
        
        # Filter windows that contain invalid bases (value 255)
        valid_mask = (w0 < 4) & (w1 < 4) & (w2 < 4) & (w3 < 4) & (w4 < 4) & (w5 < 4)
        if not np.any(valid_mask):
            continue
        
        # Compute integer indices [0..4095]
        indices = (
            (w0[valid_mask].astype(np.int32) << 10) |
            (w1[valid_mask].astype(np.int32) << 8)  |
            (w2[valid_mask].astype(np.int32) << 6)  |
            (w3[valid_mask].astype(np.int32) << 4)  |
            (w4[valid_mask].astype(np.int32) << 2)  |
            w5[valid_mask].astype(np.int32)
        )
        
        # Vectorized bin counts
        bincounts = np.bincount(indices, minlength=4096)
        counts += bincounts
        total_valid += len(indices)

    if total_valid > 0:
        counts /= total_valid

    return counts.astype(np.float32), total_valid


def run_stage_1(verify_only_25: bool = False):
    print("=" * 70)
    print("STAGE 1 — STREAMING GENOMIC FEATURE EXTRACTION")
    print("=" * 70)

    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    genomes_dir = project_root / "data" / "raw" / "genomes"
    labels_path = processed_dir / "genome_labels.csv"

    if not labels_path.exists():
        print(f"Error: {labels_path} not found.")
        return

    labels_df = pd.read_csv(labels_path)
    # Standardize column names (Genome ID -> genome_id)
    labels_df.columns = [c.strip().lower().replace(" ", "_") for c in labels_df.columns]
    labels_df['genome_id'] = labels_df['genome_id'].astype(str)

    # Convert binary phenotype to int: Resistant -> 1, Susceptible -> 0
    pheno_map = {"Resistant": 1, "Susceptible": 0, "1": 1, "0": 0}
    labels_df['phenotype'] = labels_df['phenotype'].astype(str).map(pheno_map).fillna(0).astype(np.int32)

    # Filter to genomes present on disk
    disk_fasta_map = {f.name.replace(".fna", "").replace(".fasta", ""): f for f in genomes_dir.glob("*.fna")}
    labels_df['fasta_path'] = labels_df['genome_id'].map(disk_fasta_map)
    valid_df = labels_df.dropna(subset=['fasta_path']).copy().reset_index(drop=True)

    # Freeze cohort check: top 2602
    if len(valid_df) > 2602:
        valid_df = valid_df.iloc[:2602].copy()

    total_expected = len(valid_df)
    if verify_only_25:
        print("\n*** RUNNING INITIAL VERIFICATION (FIRST 25 GENOMES) ***\n")
        valid_df = valid_df.iloc[:25].copy()
        total_expected = len(valid_df)

    print(f"Total genomes expected: {total_expected}")

    # Output paths
    features_path = processed_dir / ("genomic_features_verify25.npy" if verify_only_25 else "genomic_features.npy")
    labels_npy_path = processed_dir / ("genomic_labels_verify25.npy" if verify_only_25 else "genomic_labels.npy")
    manifest_path = processed_dir / ("genomic_manifest_verify25.csv" if verify_only_25 else "genomic_manifest.csv")

    # Allocate memmap
    shape = (total_expected, 4096)
    if not features_path.exists():
        features_mm = np.lib.format.open_memmap(features_path, mode='w+', dtype='float32', shape=shape)
    else:
        features_mm = np.lib.format.open_memmap(features_path, mode='r+', dtype='float32', shape=shape)

    # Checkpoint tracking
    processed_mask = np.zeros(total_expected, dtype=bool)
    valid_kmers_counts = np.zeros(total_expected, dtype=np.int64)

    if manifest_path.exists():
        m_df = pd.read_csv(manifest_path)
        if len(m_df) == total_expected and 'status' in m_df.columns:
            processed_mask = (m_df['status'] == 'completed').values
            if 'valid_kmers' in m_df.columns:
                valid_kmers_counts = m_df['valid_kmers'].values

    completed_count = int(np.sum(processed_mask))
    remaining_count = total_expected - completed_count
    print(f"Already processed: {completed_count} | Remaining: {remaining_count}")

    malformed_count = 0

    for idx, row in valid_df.iterrows():
        if processed_mask[idx]:
            continue

        gid = row['genome_id']
        fpath = Path(row['fasta_path'])

        try:
            feats, n_kmers = extract_genome_kmers_fast(fpath)
            features_mm[idx] = feats
            valid_kmers_counts[idx] = n_kmers
            processed_mask[idx] = True
        except Exception as e:
            print(f"Malformed/Error processing {gid} ({fpath.name}): {e}")
            features_mm[idx] = np.zeros(4096, dtype=np.float32)
            valid_kmers_counts[idx] = 0
            malformed_count += 1

        # Periodic flush & status report
        if (idx + 1) % 25 == 0 or (idx + 1) == total_expected:
            features_mm.flush()
            print(f"Progress: {idx + 1}/{total_expected} genomes processed...")
            gc.collect()

    features_mm.flush()

    # Save final labels & manifest
    valid_df['status'] = np.where(processed_mask, 'completed', 'failed')
    valid_df['valid_kmers'] = valid_kmers_counts
    valid_df[['genome_id', 'phenotype', 'valid_kmers', 'status']].to_csv(manifest_path, index=False)
    np.save(labels_npy_path, valid_df['phenotype'].values.astype(np.int32))

    # Reporting
    completed_idx = np.where(processed_mask)[0]
    total_valid_kmers = int(np.sum(valid_kmers_counts[completed_idx]))
    mean_kmers = float(np.mean(valid_kmers_counts[completed_idx])) if len(completed_idx) > 0 else 0
    min_kmers = int(np.min(valid_kmers_counts[completed_idx])) if len(completed_idx) > 0 else 0
    max_kmers = int(np.max(valid_kmers_counts[completed_idx])) if len(completed_idx) > 0 else 0

    print("\n" + "=" * 70)
    print("STAGE 1 EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"Total Genomes Expected:          {total_expected}")
    print(f"Genomes Successfully Processed:  {len(completed_idx)}")
    print(f"Genomes Remaining:               {total_expected - len(completed_idx)}")
    print(f"Malformed/Missing FASTAs:        {malformed_count}")
    print(f"Feature Matrix Shape:            {features_mm.shape}")
    print(f"Total Valid K-mers Extracted:    {total_valid_kmers:,}")
    print(f"K-mers per Genome (Mean/Min/Max): {mean_kmers:,.1f} / {min_kmers:,} / {max_kmers:,}")

    if verify_only_25:
        # Check finite and row sums
        subset_feats = np.array(features_mm[:len(completed_idx)])
        is_finite = np.all(np.isfinite(subset_feats))
        row_sums = np.sum(subset_feats, axis=1)
        valid_sums = np.allclose(row_sums, 1.0, atol=1e-4)

        print("\n--- VERIFICATION CHECKS (25 GENOMES) ---")
        print(f"All values finite:               {is_finite}")
        print(f"Row sums approximately 1.0:      {valid_sums} (Min sum: {np.min(row_sums):.5f}, Max sum: {np.max(row_sums):.5f})")
        print(f"Verification Checkpoint Passed:  {is_finite and valid_sums}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", help="Run 25-genome verification check")
    args = parser.parse_args()

    run_stage_1(verify_only_25=args.verify)
