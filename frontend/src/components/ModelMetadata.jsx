import React from "react";
import { IconInfo } from "./Icons.jsx";

export default function ModelMetadata() {
  const meta = [
    { label: "Model Architecture", val: "XGBoost + TruncatedSVD" },
    { label: "Target Antibiotic", val: "Ciprofloxacin (E. coli)" },
    { label: "Input Format", val: "Bacterial Genome FASTA" },
    { label: "Feature Space", val: "4,096 Normalized 6-mers (k=6)" },
    { label: "Training Cohort", val: "2,602 Genomes (1,319 R / 1,283 S)" },
    { label: "Partitioning Split", val: "70% Train / 15% Val / 15% Test" },
    { label: "Decision Threshold", val: "0.45 (Validation Selected)" },
    { label: "Model Artifact", val: "xgboost_genomic_kmer.pkl" }
  ];

  return (
    <div className="card">
      <div className="card-title">
        <IconInfo /> Model & Experiment Metadata
      </div>
      <table className="table-custom">
        <tbody>
          {meta.map((m, idx) => (
            <tr key={idx}>
              <td style={{ fontWeight: 600, width: "35%", color: "var(--text-secondary)" }}>{m.label}</td>
              <td style={{ fontWeight: 500, color: "var(--text-primary)" }}>{m.val}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
