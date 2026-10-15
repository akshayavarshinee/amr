import { useApi } from "../useApi.js";

export default function ModelInfoCard() {
  const { data, error, loading } = useApi("/models/current");

  return (
    <div className="card">
      <h3>Current Model</h3>
      {loading && <p>Loading…</p>}
      {error && <span className="badge error">{error}</span>}
      {data && (
        <table>
          <tbody>
            <tr>
              <td>Version</td>
              <td>{data.model_version}</td>
            </tr>
            <tr>
              <td>Artifact path</td>
              <td>{data.model_path}</td>
            </tr>
            <tr>
              <td>Decision threshold</td>
              <td>{data.decision_threshold}</td>
            </tr>
            <tr>
              <td>Registered name</td>
              <td>{data.registered_model_name ?? "—"}</td>
            </tr>
          </tbody>
        </table>
      )}
    </div>
  );
}
