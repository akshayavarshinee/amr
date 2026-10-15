"""Stage 3 & 4: Model Training, Threshold Tuning, & untouched Final Test Evaluation.

Reads persisted feature matrix (genomic_features.npy) & split assignments (genomic_splits.csv).
1. Fits StandardScaler on Train only (for LogisticRegression).
2. Fits TruncatedSVD on Train only (for XGBoost).
3. Evaluates and tunes threshold on Validation set only.
4. Evaluates ONCE on untouched Test set.
5. Saves model bundle to models/artifacts/xgboost_genomic_kmer.pkl (NEVER overwriting xgboost_phase1.pkl).
"""

import sys
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix
)
import xgboost as xgb

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
    }

def run_stage_3_and_4():
    print("=" * 70)
    print("STAGE 3 — MODEL TRAINING & VALIDATION THRESHOLD SELECTION")
    print("=" * 70)

    processed_dir = project_root / "data" / "processed"
    features_path = processed_dir / "genomic_features.npy"
    splits_path = processed_dir / "genomic_splits.csv"
    artifacts_dir = project_root / "models" / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    artifact_output_path = artifacts_dir / "xgboost_genomic_kmer.pkl"
    report_output_path = project_root / "reports" / "genomic_evaluation_report.json"
    report_output_path.parent.mkdir(parents=True, exist_ok=True)

    if not features_path.exists() or not splits_path.exists():
        print("Error: Missing features or splits. Complete Stage 1 and Stage 2 first.")
        return

    # Load persisted feature matrix & splits
    features_mm = np.load(features_path, mmap_mode='r')
    splits_df = pd.read_csv(splits_path)

    train_idx = splits_df[splits_df['split'] == 'train']['index'].values
    val_idx = splits_df[splits_df['split'] == 'val']['index'].values
    test_idx = splits_df[splits_df['split'] == 'test']['index'].values

    y_train = splits_df[splits_df['split'] == 'train']['phenotype'].values
    y_val = splits_df[splits_df['split'] == 'val']['phenotype'].values
    y_test = splits_df[splits_df['split'] == 'test']['phenotype'].values

    print(f"Dataset partitions loaded:")
    print(f"  Train: {len(train_idx)} samples | Val: {len(val_idx)} samples | Test: {len(test_idx)} samples")

    # Load features for train/val/test into float32 numpy arrays
    X_train = np.array(features_mm[train_idx], dtype=np.float32)
    X_val = np.array(features_mm[val_idx], dtype=np.float32)
    X_test = np.array(features_mm[test_idx], dtype=np.float32)

    # -------------------------------------------------------------
    # 1. LOGISTIC REGRESSION (StandardScaler fit on Train only)
    # -------------------------------------------------------------
    print("\n--- Training Logistic Regression Baseline ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    lr_model = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_model.fit(X_train_scaled, y_train)

    lr_val_prob = lr_model.predict_proba(X_val_scaled)[:, 1]
    lr_val_metrics = evaluate_predictions(y_val, lr_val_prob, threshold=0.5)
    print(f"Logistic Regression Validation AUC: {lr_val_metrics['roc_auc']:.4f} | F1: {lr_val_metrics['f1']:.4f}")

    # -------------------------------------------------------------
    # 2. XGBOOST (TruncatedSVD fit on Train only)
    # -------------------------------------------------------------
    print("\n--- Training XGBoost Classifier ---")
    # SVD fit strictly on Train partition
    n_components = min(128, len(X_train) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)

    xgb_model = xgb.XGBClassifier(
        n_estimators=150,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )
    xgb_model.fit(X_train_svd, y_train)

    xgb_val_prob = xgb_model.predict_proba(X_val_svd)[:, 1]
    
    # Validation decision threshold tuning
    best_thresh = 0.5
    best_f1 = 0.0
    for th in np.linspace(0.1, 0.9, 81):
        score = f1_score(y_val, (xgb_val_prob >= th).astype(int), zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = float(th)

    xgb_val_metrics = evaluate_predictions(y_val, xgb_val_prob, threshold=best_thresh)
    print(f"XGBoost Validation AUC: {xgb_val_metrics['roc_auc']:.4f} | Best Val F1: {best_f1:.4f} @ Threshold: {best_thresh:.4f}")

    # -------------------------------------------------------------
    # STAGE 4 — UNTOUCHED FINAL TEST EVALUATION
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STAGE 4 — UNTOUCHED FINAL TEST EVALUATION")
    print("=" * 70)

    X_test_svd = svd.transform(X_test)
    xgb_test_prob = xgb_model.predict_proba(X_test_svd)[:, 1]
    
    test_metrics = evaluate_predictions(y_test, xgb_test_prob, threshold=best_thresh)

    print(f"Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"Precision:       {test_metrics['precision']:.4f}")
    print(f"Recall:          {test_metrics['recall']:.4f}")
    print(f"F1 Score:        {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:         {test_metrics['roc_auc']:.4f}")
    print(f"MCC:             {test_metrics['mcc']:.4f}")
    print(f"Confusion Matrix: {test_metrics['confusion_matrix']}")

    # -------------------------------------------------------------
    # SAVE ARTIFACT & REPORT
    # -------------------------------------------------------------
    artifact_bundle = {
        "model": xgb_model,
        "svd": svd,
        "scaler": scaler,
        "k": 6,
        "feature_dim": 4096,
        "decision_threshold": best_thresh,
        "feature_version": "kmer_6_direct_count",
        "dataset_version": "2602_genomic_cohort",
        "validation_metrics": xgb_val_metrics,
        "test_metrics": test_metrics
    }
    joblib.dump(artifact_bundle, artifact_output_path)
    print(f"\nModel artifact saved to: {artifact_output_path}")

    report = {
        "dataset_size": len(splits_df),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "logistic_regression_val_metrics": lr_val_metrics,
        "xgboost_val_metrics": xgb_val_metrics,
        "selected_threshold": best_thresh,
        "final_untouched_test_metrics": test_metrics
    }
    with open(report_output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Evaluation report saved to: {report_output_path}")

if __name__ == "__main__":
    run_stage_3_and_4()
