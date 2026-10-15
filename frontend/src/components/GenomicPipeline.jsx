import React from "react";
import { IconDna } from "./Icons.jsx";

export default function GenomicPipeline() {
  const steps = [
    { title: "FASTA Sequence", desc: "Whole-genome FASTA" },
    { title: "FASTA Parser", desc: "Independent Contigs" },
    { title: "6-mer Extraction", desc: "ACGT Sliding Windows" },
    { title: "4,096 Vector", desc: "Normalized Frequencies" },
    { title: "SVD Transform", desc: "Train-Fitted Truncation" },
    { title: "XGBoost Classifier", desc: "Phase 2 Genomic Model" },
    { title: "Probability", desc: "Threshold @ 45%" },
    { title: "Logging & Drift", desc: "PSI Monitoring" }
  ];

  return (
    <div className="card">
      <div className="card-title">
        <IconDna /> End-to-End Genomic Pipeline Architecture
      </div>
      <div className="workflow-flex">
        {steps.map((step, idx) => (
          <React.Fragment key={idx}>
            <div className="workflow-step">
              <div className="step-title">{step.title}</div>
              <div className="step-desc">{step.desc}</div>
            </div>
            {idx < steps.length - 1 && <span className="workflow-arrow">→</span>}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}
