import React from "react";

export default function Header({ status }) {
  return (
    <header className="dashboard-header">
      <div className="header-title-area">
        <h1>AMR-GenomeGuard</h1>
        <p>Genomic AMR Phenotype Prediction & MLOps Monitoring</p>
        <span className="header-tag">E. coli · Ciprofloxacin · Genome-Based Model</span>
      </div>
      <div>
        <span className={`badge ${status?.status === "ok" ? "ok" : "warn"}`}>
          ● API Status: {status?.status === "ok" ? "Online" : "Connecting..."}
        </span>
      </div>
    </header>
  );
}
