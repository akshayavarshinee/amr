"""Stage 2: Stratified Splitter Script.

Loads persisted disk-backed feature matrix & labels.
Performs 70/15/15 stratified train/val/test split by Genome ID.
Saves split assignments to data/processed/genomic_splits.csv.
Performs strict validation (zero overlap, exact coverage, class balance).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

def run_stage_2():
    print("=" * 70)
    print("STAGE 2 — STRATIFIED SPLITTING (70/15/15)")
    print("=" * 70)

    processed_dir = project_root / "data" / "processed"
    manifest_path = processed_dir / "genomic_manifest.csv"
    labels_path = processed_dir / "genomic_labels.npy"
    splits_output_path = processed_dir / "genomic_splits.csv"

    if not manifest_path.exists() or not labels_path.exists():
        print(f"Error: Manifest ({manifest_path}) or Labels ({labels_path}) missing. Complete Stage 1 first.")
        return

    manifest_df = pd.read_csv(manifest_path)
    labels = np.load(labels_path)

    total_samples = len(manifest_df)
    print(f"Total samples loaded from manifest: {total_samples}")

    manifest_df['index'] = np.arange(total_samples)
    manifest_df['phenotype'] = labels

    # First split: Train (70%) vs Temp (30%)
    train_df, temp_df = train_test_split(
        manifest_df,
        test_size=0.30,
        random_state=42,
        stratify=manifest_df['phenotype']
    )

    # Second split: Val (15% of total = 50% of temp) vs Test (15% of total = 50% of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df['phenotype']
    )

    # Assign split labels
    manifest_df.loc[train_df.index, 'split'] = 'train'
    manifest_df.loc[val_df.index, 'split'] = 'val'
    manifest_df.loc[test_df.index, 'split'] = 'test'

    # Save to disk
    manifest_df[['genome_id', 'phenotype', 'split', 'index']].to_csv(splits_output_path, index=False)
    print(f"Saved split assignments to {splits_output_path}")

    # VERIFICATIONS
    train_ids = set(train_df['genome_id'])
    val_ids = set(val_df['genome_id'])
    test_ids = set(test_df['genome_id'])

    overlap_train_val = train_ids.intersection(val_ids)
    overlap_train_test = train_ids.intersection(test_ids)
    overlap_val_test = val_ids.intersection(test_ids)
    has_zero_overlap = len(overlap_train_val) == 0 and len(overlap_train_test) == 0 and len(overlap_val_test) == 0

    total_assigned = len(train_df) + len(val_df) + len(test_df)
    all_assigned_once = total_assigned == total_samples

    print("\n--- VERIFICATION CHECKS ---")
    print(f"Zero Genome ID Overlap:          {has_zero_overlap}")
    print(f"All Samples Assigned Exactly Once: {all_assigned_once} ({total_assigned}/{total_samples})")

    print("\n--- SPLIT DISTRIBUTION ---")
    print(f"Train Count: {len(train_df)} ({len(train_df)/total_samples:.1%}) | Resistant: {sum(train_df['phenotype']==1)}, Susceptible: {sum(train_df['phenotype']==0)}")
    print(f"Val Count:   {len(val_df)} ({len(val_df)/total_samples:.1%}) | Resistant: {sum(val_df['phenotype']==1)}, Susceptible: {sum(val_df['phenotype']==0)}")
    print(f"Test Count:  {len(test_df)} ({len(test_df)/total_samples:.1%}) | Resistant: {sum(test_df['phenotype']==1)}, Susceptible: {sum(test_df['phenotype']==0)}")

if __name__ == "__main__":
    run_stage_2()
