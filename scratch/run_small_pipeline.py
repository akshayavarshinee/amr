"""Small 10-Genome End-to-End Pipeline Integration Test.

Workflow:
1. Load 10 test genomes from data/raw/genomes_test/ and data/processed/genome_labels.csv
2. Extract 4096 6-mer frequency feature matrix
3. Stratified 3-way split (Train / Validation / Test) and zero leakage check
4. Fit TruncatedSVD ONLY on training partition
5. Train Logistic Regression baseline & XGBoost model
6. Tune decision threshold on Validation set ONLY
7. Evaluate promoted model ONCE on untouched Test set
8. Save complete model artifact bundle
9. Reload artifact bundle and execute inference on a sample sequence
"""

from pathlib import Path
import joblib

from amr_guard.data.preprocess import run as run_preprocess
from amr_guard.models.train import run as run_train
from amr_guard.models.predict import predict_single


def main():
    print("=" * 70)
    print("Step 3: Small 10-Genome End-to-End Pipeline Integration Test")
    print("=" * 70)

    # 1. Preprocess 10 test genomes -> 4096-dim k-mer feature matrix & split
    print("\n[1/6] Running genomic feature extraction & 3-way split...")
    data = run_preprocess(
        labels_csv="data/processed/genome_labels.csv",
        genomes_dir="data/raw/genomes",
        test_dir="data/raw/genomes_test",
        max_genomes=10,
    )

    X_train, X_val, X_test = data["X_train"], data["X_val"], data["X_test"]
    y_train, y_val, y_test = data["y_train"], data["y_val"], data["y_test"]

    print(f"      Feature Matrix Shape: ({len(X_train) + len(X_val) + len(X_test)}, {len(data['feature_names'])})")
    print(f"      Split Sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 2. Train models (Logistic Regression + XGBoost with SVD fitted ON TRAIN ONLY)
    print("\n[2/6] Fitting models & TruncatedSVD (fitted ON TRAIN ONLY)...")
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

    print("\n[3/6] Validation Metrics & Threshold Selection:")
    print(f"      Tuned Decision Threshold: {bundle['decision_threshold']}")
    for model_name, m in val_res.items():
        print(f"      Val [{model_name}]: Accuracy={m['accuracy']}, F1={m['f1']}, MCC={m['mcc']}")

    print("\n[4/6] Final Untouched Test Evaluation:")
    for model_name, m in test_res.items():
        print(f"      Test [{model_name}]: Accuracy={m['accuracy']}, Precision={m['precision']}, Recall={m['recall']}, F1={m['f1']}")

    # 3. Artifact verification
    artifact_path = Path("models/artifacts/xgboost_phase1.pkl")
    print(f"\n[5/6] Verifying saved artifact bundle at: {artifact_path}")
    assert artifact_path.exists(), "Artifact bundle not saved!"

    reloaded_bundle = joblib.load(artifact_path)
    assert "model" in reloaded_bundle
    assert "svd" in reloaded_bundle
    assert "vocabulary" in reloaded_bundle
    assert len(reloaded_bundle["vocabulary"]) == 4096
    print(f"      Artifact verified: k={reloaded_bundle['k']}, Vocabulary size={len(reloaded_bundle['vocabulary'])}, Threshold={reloaded_bundle['decision_threshold']}")

    # 4. Inference verification
    print("\n[6/6] Testing inference reload & prediction...")
    test_fasta_path = Path("data/raw/genomes_test/562.191786.fna")
    pred_res = predict_single(reloaded_bundle, test_fasta_path)
    print("      Prediction Result:")
    print(f"        Genome File: {test_fasta_path.name}")
    print(f"        Prediction:  {pred_res['prediction']}")
    print(f"        Probability: {pred_res['probability']}")
    print(f"        Threshold:   {pred_res['threshold']}")
    print(f"        Confidence:  {pred_res['confidence']}")

    print("\n" + "=" * 70)
    print("Small 10-Genome End-to-End Pipeline Integration Test PASSED SUCCESSFULLY!")
    print("=" * 70)

    return {
        "feature_shape": (len(X_train) + len(X_val) + len(X_test), len(data['feature_names'])),
        "split_sizes": {"train": len(X_train), "val": len(X_val), "test": len(X_test)},
        "metrics": test_res,
        "threshold": bundle["decision_threshold"],
        "artifact_path": str(artifact_path),
    }


if __name__ == "__main__":
    main()
