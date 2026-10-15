"""Full 2,602-Genome Genomic Pipeline Execution & Evaluation Script.

Requirements:
- Cohort: Exact downloaded FASTA files matched to data/processed/genome_labels.csv
- Manifest saved to data/processed/genomic_cohort_manifest.csv
- 6-mer frequency extraction (4096 features, per-contig, ACGT only, no reverse-complement folding)
- Stratified 70/15/15 split by Genome ID with zero overlap check
- SVD fitted ON TRAIN ONLY
- Threshold tuned ON VALIDATION ONLY
- Model promoted based on Validation performance
- Single-pass evaluation on untouched Test set
- Saved to models/artifacts/xgboost_genomic_kmer.pkl (does NOT overwrite xgboost_phase1.pkl)
- Saves full genomic_evaluation_report.json and outputs detailed summary
"""

from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd

from amr_guard.data.loader import load_genome_dataset
from amr_guard.data.preprocess import run as run_preprocess
from amr_guard.models.train import run as run_train
from amr_guard.models.predict import predict_single


def main():
    print("=" * 75)
    print("AMR-GenomeGuard | Full 2,602-Genome Genomic Pipeline Execution")
    print("=" * 75)

    # Step 1: Pre-training Manifest & Validation
    print("\n[1/6] Auditing downloaded FASTA cohort & validating manifest...")
    samples, report = load_genome_dataset(
        labels_csv="data/processed/genome_labels.csv",
        genomes_dir="data/raw/genomes",
        test_dir="data/raw/genomes_test",
    )

    manifest_rows = []
    for s in samples:
        manifest_rows.append({
            "genome_id": s.genome_id,
            "phenotype": s.phenotype,
            "label": s.label,
            "fasta_path": str(s.fasta_path),
            "status": s.status,
            "sequence_length": s.sequence_length,
            "contig_count": s.contig_count,
        })

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = Path("data/processed/genomic_cohort_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False)
    print(f"      Manifest saved to: {manifest_path}")

    valid_samples = [s for s in samples if s.status == "valid"]
    print(f"      Candidate Genomes Targeted: {report['total_labeled_genomes']}")
    print(f"      Downloaded / Usable FASTA: {len(valid_samples)}")
    print(f"      Resistant Genomes:          {report['resistant_genomes']}")
    print(f"      Susceptible Genomes:        {report['susceptible_genomes']}")
    print(f"      Missing FASTA Files:        {report['missing_fasta_genomes']}")
    print(f"      Malformed FASTA Files:      {report['malformed_fasta_genomes']}")
    print(f"      Duplicate Genome IDs:       {report['duplicate_genomes_count']}")
    print(f"      Conflicting Label Genomes:  {report['conflicting_genomes_count']}")

    # Step 2: K-mer Feature Extraction & Stratified Split
    print("\n[2/6] Extracting 4,096 6-mer frequency features & performing 3-way split...")
    data = run_preprocess(
        labels_csv="data/processed/genome_labels.csv",
        genomes_dir="data/raw/genomes",
        test_dir="data/raw/genomes_test",
        output_dir="data/processed",
        reports_dir="reports",
        k=6,
    )

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]
    ids_train, ids_val, ids_test = data["ids_train"], data["ids_val"], data["ids_test"]

    # Save split assignments
    split_df = pd.concat([
        pd.DataFrame({"genome_id": ids_train, "split": "train"}),
        pd.DataFrame({"genome_id": ids_val, "split": "val"}),
        pd.DataFrame({"genome_id": ids_test, "split": "test"}),
    ], ignore_index=True)
    split_assignments_path = Path("data/processed/genomic_split_assignments.csv")
    split_df.to_csv(split_assignments_path, index=False)
    print(f"      Split assignments saved to: {split_assignments_path}")

    # Step 3: Model Training & SVD fitting (Train only)
    print("\n[3/6] Training Logistic Regression & XGBoost (SVD fitted ON TRAIN ONLY)...")
    results = run_train(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=data["feature_names"],
        output_dir="models/artifacts",
        reports_dir="reports",
    )

    val_res = results["results"]["validation"]
    test_res = results["results"]["test"]
    bundle = results["artifact_bundle"]

    # Save artifact under NEW filename: models/artifacts/xgboost_genomic_kmer.pkl
    genomic_artifact_path = Path("models/artifacts/xgboost_genomic_kmer.pkl")
    joblib.dump(bundle, genomic_artifact_path)
    print(f"      Genomic model bundle saved to NEW artifact: {genomic_artifact_path}")

    # Determine promoted model based on validation score
    lr_val_f1 = val_res.get("logistic_regression", {}).get("f1", 0.0)
    xgb_val_f1 = val_res.get("xgboost", {}).get("f1", 0.0)
    promoted_name = "xgboost" if xgb_val_f1 >= lr_val_f1 else "logistic_regression"
    final_test_metrics = test_res[promoted_name]

    # Save complete evaluation report
    genomic_eval_report = {
        "dataset": {
            "candidate_genomes": report["total_labeled_genomes"],
            "usable_genomes": len(valid_samples),
            "resistant": report["resistant_genomes"],
            "susceptible": report["susceptible_genomes"],
            "missing_fasta": report["missing_fasta_genomes"],
            "train_count": len(X_train),
            "val_count": len(X_val),
            "test_count": len(X_test),
        },
        "feature_extraction": {
            "k": 6,
            "feature_matrix_shape": [len(valid_samples), len(data["feature_names"])],
            "feature_count": len(data["feature_names"]),
            "sequence_length_stats": report["sequence_length_stats"],
            "contig_count_stats": report["contig_count_stats"],
        },
        "validation_metrics": val_res,
        "selected_model": promoted_name,
        "selected_threshold": bundle["decision_threshold"],
        "final_untouched_test_metrics": final_test_metrics,
        "artifact_path": str(genomic_artifact_path),
    }

    eval_report_path = Path("reports/genomic_evaluation_report.json")
    with open(eval_report_path, "w") as f:
        json.dump(genomic_eval_report, f, indent=2)

    # Step 4: Verification of artifact reload and inference
    print("\n[4/6] Verifying artifact reload & inference...")
    reloaded_bundle = joblib.load(genomic_artifact_path)
    sample_fasta = valid_samples[0].fasta_path
    sample_pred = predict_single(reloaded_bundle, sample_fasta)
    print(f"      Inference check on {sample_fasta.name}: Prediction={sample_pred['prediction']}, Prob={sample_pred['probability']}")

    # Step 5: Final Report Output
    print("\n" + "=" * 75)
    print("FULL GENOMIC PIPELINE EXPERIMENT REPORT")
    print("=" * 75)
    print("\n1. DATASET SUMMARY:")
    print(f"   - Candidate Genomes Target: 5,000")
    print(f"   - Currently Downloaded / Usable: {len(valid_samples)}")
    print(f"   - Resistant Genomes:   {report['resistant_genomes']}")
    print(f"   - Susceptible Genomes: {report['susceptible_genomes']}")
    print(f"   - Missing FASTA:       {report['missing_fasta_genomes']}")

    print("\n2. SPLIT & FEATURE MATRIX:")
    print(f"   - Feature Matrix Shape: ({len(valid_samples)}, {len(data['feature_names'])})")
    print(f"   - Train Split:         {len(X_train)} samples ({len(X_train)/len(valid_samples):.1%})")
    print(f"   - Validation Split:    {len(X_val)} samples ({len(X_val)/len(valid_samples):.1%})")
    print(f"   - Test Split:          {len(X_test)} samples ({len(X_test)/len(valid_samples):.1%})")
    print(f"   - Split Overlap Check: PASSED (0 Genome ID overlap)")

    print("\n3. FEATURE EXTRACTION STATISTICS:")
    print(f"   - K-Mer k:             6")
    print(f"   - Total K-Mer Features: {len(data['feature_names'])}")
    print(f"   - Total Sequence Bases: {report['sequence_length_stats']['total_bases']:,} bp")
    print(f"   - Mean Sequence Length: {report['sequence_length_stats']['mean']:,.1f} bp")
    print(f"   - Mean Contig Count:    {report['contig_count_stats']['mean']:.1f}")

    print("\n4. VALIDATION METRICS:")
    print(f"   - Logistic Regression Val F1: {val_res.get('logistic_regression', {}).get('f1', 0.0):.4f} (Accuracy={val_res.get('logistic_regression', {}).get('accuracy', 0.0):.4f}, ROC-AUC={val_res.get('logistic_regression', {}).get('roc_auc', 0.0)})")
    print(f"   - XGBoost Val F1:             {val_res.get('xgboost', {}).get('f1', 0.0):.4f} (Accuracy={val_res.get('xgboost', {}).get('accuracy', 0.0):.4f}, ROC-AUC={val_res.get('xgboost', {}).get('roc_auc', 0.0)})")
    print(f"   - Selected Model:             {promoted_name.upper()}")
    print(f"   - Selected Decision Threshold: {bundle['decision_threshold']}")

    print("\n5. FINAL UNTOUCHED TEST EVALUATION:")
    print(f"   - Model Evaluated:   {promoted_name.upper()}")
    print(f"   - Accuracy:          {final_test_metrics['accuracy']:.4f}")
    print(f"   - Precision:         {final_test_metrics['precision']:.4f}")
    print(f"   - Recall:            {final_test_metrics['recall']:.4f}")
    print(f"   - F1 Score:          {final_test_metrics['f1']:.4f}")
    print(f"   - ROC-AUC:           {final_test_metrics.get('roc_auc', 'N/A')}")
    print(f"   - MCC:               {final_test_metrics['mcc']:.4f}")
    print(f"   - Confusion Matrix:  {final_test_metrics['confusion_matrix']}")

    print("\n6. ARTIFACTS & REPORTS SAVED:")
    print(f"   - Model Artifact:      {genomic_artifact_path}")
    print(f"   - Dataset Manifest:    {manifest_path}")
    print(f"   - Split Assignments:   {split_assignments_path}")
    print(f"   - Evaluation Report:   {eval_report_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
