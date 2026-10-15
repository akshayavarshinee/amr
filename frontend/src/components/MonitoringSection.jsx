import React from "react";
import { IconActivity } from "./Icons.jsx";

export default function MonitoringSection({ drift, predictionsHistory }) {
  const hasDriftData = drift && drift.status === "ok";
  const nSamples = drift?.n_samples || predictionsHistory.length;

  return (
    <div className="card">
      <div className="card-title">
        <IconActivity /> MLOps Drift & Activity Monitoring
      </div>

      {/* Prediction Activity Summary */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "12px", marginBottom: "20px" }}>
        <div style={{ background: "#f8fafc", padding: "12px", borderRadius: "6px", border: "1px solid var(--border-color)" }}>
          <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>TOTAL SERVED PREDICTIONS</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700 }}>{predictionsHistory.length}</div>
        </div>
        <div style={{ background: "#fff5f5", padding: "12px", borderRadius: "6px", border: "1px solid var(--status-resistant-border)" }}>
          <div style={{ fontSize: "0.75rem", color: "#dc2626", fontWeight: 600 }}>RESISTANT PREDICTIONS</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#dc2626" }}>
            {predictionsHistory.filter(p => p.prediction.toLowerCase() === "resistant").length}
          </div>
        </div>
        <div style={{ background: "#f0fdf4", padding: "12px", borderRadius: "6px", border: "1px solid var(--status-susceptible-border)" }}>
          <div style={{ fontSize: "0.75rem", color: "#16a34a", fontWeight: 600 }}>SUSCEPTIBLE PREDICTIONS</div>
          <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "#16a34a" }}>
            {predictionsHistory.filter(p => p.prediction.toLowerCase() === "susceptible").length}
          </div>
        </div>
      </div>

      {/* Prediction History Table */}
      <div style={{ marginBottom: "24px" }}>
        <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}>
          Recent Prediction Log
        </div>
        {predictionsHistory.length === 0 ? (
          <div className="empty-state">
            <p>No logged predictions yet.</p>
            <div style={{ fontSize: "0.8rem" }}>Run a genomic prediction above to begin tracking live inference requests.</div>
          </div>
        ) : (
          <table className="table-custom">
            <thead>
              <tr>
                <th>Timestamp</th>
                <th>Sample Source</th>
                <th>Prediction</th>
                <th>Probability</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {predictionsHistory.map((p, i) => (
                <tr key={i}>
                  <td>{p.timestamp}</td>
                  <td>{p.source || "FASTA sequence"}</td>
                  <td>
                    <span className={`badge ${p.prediction.toLowerCase() === "resistant" ? "resistant" : "susceptible"}`}>
                      {p.prediction}
                    </span>
                  </td>
                  <td>{(p.probability * 100).toFixed(1)}%</td>
                  <td>{((p.threshold || 0.45) * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Feature Drift Subsection */}
      <div>
        <div style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-primary)", marginBottom: "8px" }}>
          Feature Drift Status (PSI)
        </div>

        {hasDriftData ? (
          <div>
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "12px" }}>
              Population Stability Index calculated across {nSamples} served samples against reference training distribution.
            </div>
            <table className="table-custom">
              <thead>
                <tr>
                  <th>Feature / Component</th>
                  <th>PSI Value</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {drift.features.slice(0, 5).map((f, idx) => (
                  <tr key={idx}>
                    <td>{f.feature}</td>
                    <td>{f.psi.toFixed(4)}</td>
                    <td>
                      <span className={`badge ${f.drifted ? "resistant" : "ok"}`}>
                        {f.drifted ? "Drift Detected" : "Stable"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="empty-state">
            <div style={{ fontWeight: 600, color: "var(--text-primary)", marginBottom: "4px" }}>
              Data Sufficiency: Drift Monitoring Requires 10 Predictions
            </div>
            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
              Current logged predictions: {nSamples} / 10
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
