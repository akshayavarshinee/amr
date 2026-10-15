"""
Phase 1 pipeline entry point.

Usage:
    python run_phase1.py
    python run_phase1.py --csv path/to/BVBRC_genome_amr.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from amr_guard.data.inspect_dataset import inspect
from amr_guard.data.preprocess import run as preprocess
from amr_guard.models.train import run as train
from amr_guard.utils.config import load_params


def main():
    config = load_params()
    parser = argparse.ArgumentParser(description="AMR-GenomeGuard Phase 1 Pipeline")
    parser.add_argument(
        "--csv",
        default=config["data"]["raw_csv"],
        help="Path to BV-BRC AMR CSV file",
    )
    parser.add_argument("--organism", default=config["data"]["organism_filter"])
    parser.add_argument("--antibiotic", default=config["data"]["antibiotic_filter"])
    parser.add_argument("--output-data", default=config["paths"]["processed_dir"])
    parser.add_argument("--output-models", default=config["paths"]["models_dir"])
    parser.add_argument("--output-reports", default=config["paths"]["reports_dir"])
    args = parser.parse_args()

    print("=" * 60)
    print("AMR-GenomeGuard  |  Phase 1 Pipeline")
    print("=" * 60)

    # Step 1: Inspect
    print("\n[1/3] Inspecting dataset...")
    report = inspect(args.csv, report_dir=args.output_reports)
    print(f"      Total rows    : {report['total_rows']}")
    print(f"      Label dist    : {report['label_distribution']}")
    print(f"      Top antibiotics: {list(report['top_antibiotics'].keys())[:5]}")

    # Step 2: Preprocess
    print(f"\n[2/3] Preprocessing  ({args.organism} / {args.antibiotic})...")
    data = preprocess(
        csv_path=args.csv,
        organism_filter=args.organism,
        antibiotic_filter=args.antibiotic,
        output_dir=args.output_data,
    )
    s = data["summary"]
    print(f"      Raw rows          : {s['raw_rows']}")
    print(f"      Usable genomes    : {s['unique_genomes']}")
    print(f"      Conflicting genomes: {s['conflicting_label_genome_count']}")
    print(f"      Train / Val / Test: {s['train_size']} / {s['val_size']} / {s['test_size']}")
    print(f"      Features          : {s['features']}")

    if s["unique_genomes"] < 30:
        print("\nWARNING: fewer than 30 unique genomes found for the selected scope.")

    # Step 3: Train + Evaluate
    print("\n[3/3] Training models...")
    results = train(
        data["X_train"],
        data["X_val"],
        data["X_test"],
        data["y_train"],
        data["y_val"],
        data["y_test"],
        encoders=data["encoders"],
        feature_names=data["feature_names"],
        output_dir=args.output_models,
        reports_dir=args.output_reports,
    )

    print("\n" + "=" * 60)
    print("Phase 1 complete. Artifacts saved:")
    print(f"  data/    -> {args.output_data}")
    print(f"  models/  -> {args.output_models}")
    print(f"  reports/ -> {args.output_reports}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
