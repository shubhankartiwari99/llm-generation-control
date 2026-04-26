"use client";

import { RecentRun } from "@/types";

interface RecentRunsPanelProps {
  runs: RecentRun[];
}

export default function RecentRunsPanel({ runs }: RecentRunsPanelProps) {
  return (
    <div className="glass-panel panel-content" style={{ marginTop: "2rem" }}>
      <h3 style={{ marginBottom: "1rem" }}>Recent Runs</h3>
      {runs.length === 0 ? (
        <div className="text-secondary">No run history yet.</div>
      ) : (
        <div style={{ overflowX: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
            <thead>
              <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left" }}>
                <th style={{ padding: "0.6rem 0.5rem" }}>Prompt</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Mode</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Δ Confidence</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Instability Reduction</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const delta = run.summary_metrics?.compare?.delta_confidence;
                const reduced = run.summary_metrics?.compare?.instabilities_reduced_by;
                return (
                  <tr key={run.trace_id} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                    <td style={{ padding: "0.6rem 0.5rem", maxWidth: "320px" }}>{run.prompt}</td>
                    <td style={{ padding: "0.6rem 0.5rem" }}>{run.mode}</td>
                    <td style={{ padding: "0.6rem 0.5rem" }}>
                      {typeof delta === "number" ? `${(delta * 100).toFixed(1)} pts` : "-"}
                    </td>
                    <td style={{ padding: "0.6rem 0.5rem" }}>
                      {typeof reduced === "number" ? reduced : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
