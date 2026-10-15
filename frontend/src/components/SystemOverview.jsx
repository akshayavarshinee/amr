import React from "react";
import { IconActivity, IconCpu } from "./Icons.jsx";

export default function SystemOverview({ modelInfo, health }) {
  return (
    <div className="grid-2col">
      <div className="card">
        <div className="card-title">
          <IconActivity /> System Status
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "8px" }}>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>API SERVICE</div>
            <div style={{ fontWeight: 600, color: health?.status === "ok" ? "var(--accent-blue)" : "#dc2626" }}>
              {health?.status === "ok" ? "Operational" : "Offline"}
            </div>
          </div>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>MODEL STATE</div>
            <div style={{ fontWeight: 600 }}>{health?.model_loaded ? "Loaded & Active" : "Unloaded"}</div>
          </div>
        </div>
      </div>

      <div className="card">
        <div className="card-title">
          <IconCpu /> Production Genomic Model
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px", marginTop: "8px" }}>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>MODEL VERSION</div>
            <div style={{ fontWeight: 600 }}>{modelInfo?.model_version || "ecoli-cipro-v1"}</div>
          </div>
          <div>
            <div style={{ fontSize: "0.75rem", color: "var(--text-secondary)", fontWeight: 600 }}>DECISION THRESHOLD</div>
            <div style={{ fontWeight: 600, color: "var(--accent-blue)" }}>
              {modelInfo?.decision_threshold ? `${(modelInfo.decision_threshold * 100).toFixed(1)}%` : "45.0%"}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
