import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../useApi.js";

export default function DriftPanel() {
  const { data, error, loading } = useApi("/monitoring/drift", 15000);

  return (
    <div className="card">
      <h3>Feature Drift (PSI vs training data)</h3>
      {loading && <p>Loading…</p>}
      {error && <span className="badge error">{error}</span>}
      {data && data.status !== "ok" && <p>{data.detail}</p>}
      {data && data.status === "ok" && (
        <>
          <p>
            {data.n_samples} logged predictions · threshold {data.psi_threshold} ·{" "}
            <strong>{data.n_drifted_features}</strong> drifted feature(s)
          </p>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={data.features}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="feature" angle={-25} textAnchor="end" height={70} />
              <YAxis />
              <Tooltip />
              <Bar dataKey="psi" fill="#c0392b" />
            </BarChart>
          </ResponsiveContainer>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>PSI</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {data.features.map((f) => (
                <tr key={f.feature}>
                  <td>{f.feature}</td>
                  <td>{f.psi}</td>
                  <td>
                    <span className={`badge ${f.drifted ? "drifted" : "stable"}`}>
                      {f.drifted ? "drifted" : "stable"}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
