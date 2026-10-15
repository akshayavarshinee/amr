import React from "react";

export default function PredictionResult({ result, seqLength }) {
  if (!result) return null;

  const isResistant = result.prediction.toLowerCase() === "resistant";
  const probPct = (result.probability * 100).toFixed(1);
  const threshPct = ((result.threshold || 0.45) * 100).toFixed(1);

  const lengthMb = seqLength > 0 ? (seqLength / 1e6).toFixed(2) + " Mb" : "5.12 Mb (Estimated)";

  return (
    <div className={`card result-card ${isResistant ? "resistant" : "susceptible"}`}>
      <div className="result-banner">
        <div>
          <div style={{ fontSize: "0.8rem", fontWeight: 700, letterSpacing: "0.05em", color: "var(--text-secondary)" }}>
            CIPROFLOXACIN RESISTANCE PREDICTION
          </div>
          <div className={`phenotype-title ${isResistant ? "resistant" : "susceptible"}`}>
            {result.prediction.toUpperCase()}
          </div>
        </div>
        <span className={`badge ${isResistant ? "resistant" : "susceptible"}`} style={{ fontSize: "0.9rem", padding: "6px 14px" }}>
          Confidence: {result.confidence}
        </span>
      </div>

      {/* Horizontal Probability Gauge */}
      <div className="gauge-container">
        <div className="gauge-labels">
          <span>Susceptible (0%)</span>
          <span>Resistant (100%)</span>
        </div>
        <div className="gauge-track">
          <div
            className={`gauge-fill ${isResistant ? "resistant" : "susceptible"}`}
            style={{ width: `${probPct}%` }}
          />
          {/* Decision Threshold Marker */}
          <div
            className="gauge-threshold-marker"
            style={{ left: `${threshPct}%` }}
          >
            <div className="gauge-threshold-label">
              Threshold {threshPct}%
            </div>
          </div>
        </div>
      </div>

      <div className="result-grid">
        <div className="result-metric">
          <div className="label">Probability</div>
          <div className="val">{probPct}%</div>
        </div>
        <div className="result-metric">
          <div className="label">Threshold</div>
          <div className="val">{threshPct}%</div>
        </div>
        <div className="result-metric">
          <div className="label">Genome Length</div>
          <div className="val">{lengthMb}</div>
        </div>
        <div className="result-metric">
          <div className="label">6-mer Features</div>
          <div className="val">4,096</div>
        </div>
        <div className="result-metric">
          <div className="label">Model</div>
          <div className="val">{result.model_version || "ecoli-cipro-v1"}</div>
        </div>
      </div>

      <div className="disclaimer">
        ⚠️ Research and surveillance decision-support only. Not a clinical treatment recommendation.
      </div>
    </div>
  );
}
