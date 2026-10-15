import React, { useState, useEffect } from "react";
import Header from "./components/Header.jsx";
import SystemOverview from "./components/SystemOverview.jsx";
import PredictorCard from "./components/PredictorCard.jsx";
import PredictionResult from "./components/PredictionResult.jsx";
import GenomicPipeline from "./components/GenomicPipeline.jsx";
import ModelPerformance from "./components/ModelPerformance.jsx";
import ModelMetadata from "./components/ModelMetadata.jsx";
import MonitoringSection from "./components/MonitoringSection.jsx";

export default function App() {
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [drift, setDrift] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [lastSeqLength, setLastSeqLength] = useState(0);
  const [loading, setLoading] = useState(false);
  const [predictionsHistory, setPredictionsHistory] = useState([]);

  useEffect(() => {
    fetchHealth();
    fetchModelInfo();
    fetchDrift();
  }, []);

  const fetchHealth = async () => {
    try {
      const res = await fetch("/health");
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.error("Health check failed", err);
    }
  };

  const fetchModelInfo = async () => {
    try {
      const res = await fetch("/models/current");
      const data = await res.json();
      setModelInfo(data);
    } catch (err) {
      console.error("Model info fetch failed", err);
    }
  };

  const fetchDrift = async () => {
    try {
      const res = await fetch("/monitoring/drift");
      const data = await res.json();
      setDrift(data);
    } catch (err) {
      console.error("Drift fetch failed", err);
    }
  };

  const handlePredict = async (payload) => {
    setLoading(true);
    setPredictionResult(null);

    // Calculate length for display
    if (payload.fasta_content) {
      const lines = payload.fasta_content.split("\n");
      let len = 0;
      for (let l of lines) {
        const t = l.trim();
        if (t && !t.startsWith(">")) len += t.length;
      }
      setLastSeqLength(len);
    }

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`Server returned status ${res.status}`);
      }

      const data = await res.json();
      setPredictionResult(data);

      // Add to prediction history log
      const newEntry = {
        timestamp: new Date().toLocaleTimeString(),
        source: payload.filename || "Pasted FASTA",
        prediction: data.prediction,
        probability: data.probability,
        threshold: data.threshold || 0.45
      };

      setPredictionsHistory((prev) => [newEntry, ...prev]);
      fetchDrift();
    } catch (err) {
      alert(`Prediction Failed: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="dashboard-container">
      <Header status={health} />
      <SystemOverview health={health} modelInfo={modelInfo} />

      <PredictorCard onPredict={handlePredict} loading={loading} />

      <PredictionResult result={predictionResult} seqLength={lastSeqLength} />

      <GenomicPipeline />

      <ModelPerformance />

      <MonitoringSection drift={drift} predictionsHistory={predictionsHistory} />

      <ModelMetadata />
    </div>
  );
}
