import React, { useState } from "react";
import { IconMicroscope, IconUpload } from "./Icons.jsx";

export default function PredictorCard({ onPredict, loading }) {
  const [activeTab, setActiveTab] = useState("upload");
  const [fastaContent, setFastaContent] = useState("");
  const [fileName, setFileName] = useState("");
  const [seqLength, setSeqLength] = useState(0);

  const calculateLength = (str) => {
    if (!str) return 0;
    const lines = str.split("\n");
    let len = 0;
    for (let line of lines) {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith(">")) {
        len += trimmed.length;
      }
    }
    return len;
  };

  const handleTextChange = (e) => {
    const val = e.target.value;
    setFastaContent(val);
    setSeqLength(calculateLength(val));
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setFileName(file.name);
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target.result;
      setFastaContent(content);
      setSeqLength(calculateLength(content));
    };
    reader.readAsText(file);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!fastaContent.trim()) return;
    onPredict({ fasta_content: fastaContent, filename: fileName });
  };

  return (
    <div className="card">
      <div className="card-title">
        <IconMicroscope /> Predict Ciprofloxacin Resistance
      </div>
      <div className="card-subtitle">
        Provide a bacterial genome sequence in FASTA format to derive 4,096 6-mer frequency features and predict phenotype.
      </div>

      <div className="tab-buttons">
        <button
          className={`tab-btn ${activeTab === "upload" ? "active" : ""}`}
          onClick={() => setActiveTab("upload")}
        >
          Upload FASTA File
        </button>
        <button
          className={`tab-btn ${activeTab === "paste" ? "active" : ""}`}
          onClick={() => setActiveTab("paste")}
        >
          Paste FASTA Sequence
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        {activeTab === "upload" ? (
          <div>
            <label className="dropzone">
              <input
                type="file"
                accept=".fasta,.fa,.fna"
                onChange={handleFileUpload}
                style={{ display: "none" }}
              />
              <div style={{ color: "var(--accent-blue)", marginBottom: "8px" }}>
                <IconUpload />
              </div>
              <div style={{ fontWeight: 600, color: "var(--text-primary)" }}>
                {fileName ? fileName : "Click or drag & drop a .fasta / .fa / .fna file here"}
              </div>
              <div style={{ fontSize: "0.8rem", color: "var(--text-secondary)", marginTop: "4px" }}>
                Supports whole-genome bacterial FASTA contigs
              </div>
            </label>
          </div>
        ) : (
          <div>
            <textarea
              className="textarea-fasta"
              placeholder=">sample_contig_1&#10;AGCTTAGCGA...&#10;>sample_contig_2&#10;GCTAGCTAGC..."
              value={fastaContent}
              onChange={handleTextChange}
            />
          </div>
        )}

        {seqLength > 0 && (
          <div className="sequence-info">
            <span><strong>Sequence Length:</strong> {seqLength.toLocaleString()} bp</span>
            <span><strong>Features to Extract:</strong> 4,096 (k=6)</span>
            <span><strong>Format:</strong> Validated FASTA</span>
          </div>
        )}

        <button
          type="submit"
          className="btn-primary"
          disabled={loading || !fastaContent.trim()}
        >
          {loading ? "EXTRACTING K-MERS & PREDICTING..." : "PREDICT CIPROFLOXACIN RESISTANCE"}
        </button>
      </form>
    </div>
  );
}
