import React from "react";
import { IconBarChart } from "./Icons.jsx";

export default function ModelPerformance() {
  const metrics = [
    { label: "Accuracy", value: "85.42%" },
    { label: "Precision", value: "81.61%" },
    { label: "Recall", value: "91.92%" },
    { label: "F1 Score", value: "86.46%" },
    { label: "ROC-AUC", value: "92.19%" },
    { label: "MCC", value: "0.7138" }
  ];

  return (
    <div className="card">
      <div className="card-title">
        <IconBarChart /> Final Model Performance (Untouched Test Partition: 391 Genomes)
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: "12px", marginBottom: "20px" }}>
        {metrics.map((m, i) => (
          <div key={i} style={{ background: "#f8fafc", padding: "10px", borderRadius: "6px", border: "1px solid var(--border-color)", textAlign: "center" }}>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>{m.label}</div>
            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--text-primary)", marginTop: "2px" }}>{m.value}</div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: "16px" }}>
        <div style={{ fontSize: "0.85rem", fontWeight: 600, color: "var(--text-secondary)", marginBottom: "8px", textAlign: "center" }}>
          Test Set Confusion Matrix
        </div>
        <table className="matrix-table">
          <thead>
            <tr>
              <th></th>
              <th>Predicted Susceptible</th>
              <th>Predicted Resistant</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th style={{ textAlign: "right" }}>Actual Susceptible</th>
              <td className="matrix-cell tn">152 (TN)</td>
              <td className="matrix-cell fp">41 (FP)</td>
            </tr>
            <tr>
              <th style={{ textAlign: "right" }}>Actual Resistant</th>
              <td className="matrix-cell fn">16 (FN)</td>
              <td className="matrix-cell tp">182 (TP)</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
