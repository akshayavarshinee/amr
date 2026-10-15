import { useApi } from "../useApi.js";

export default function HealthCard() {
  const { data, error, loading } = useApi("/health", 10000);

  return (
    <div className="card">
      <h3>Service Health</h3>
      {loading && <p>Checking…</p>}
      {error && <span className="badge error">unreachable: {error}</span>}
      {data && (
        <div className="status-row">
          <span className={`badge ${data.status === "ok" ? "ok" : "error"}`}>{data.status}</span>
          <span className={`badge ${data.model_loaded ? "ok" : "error"}`}>
            model {data.model_loaded ? "loaded" : "missing"}
          </span>
        </div>
      )}
    </div>
  );
}
