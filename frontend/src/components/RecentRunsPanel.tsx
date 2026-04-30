"use client";

import { Activity } from "lucide-react";
import { RecentRun } from "@/types";

interface RecentRunsPanelProps {
  runs: RecentRun[];
  isLoading?: boolean;
}

export default function RecentRunsPanel({ runs, isLoading = false }: RecentRunsPanelProps) {
  return (
    <div className="glass-panel panel-content mt-2">
      <h3 style={{ marginBottom: "1rem" }}>Recent Runs</h3>
      {isLoading ? (
        <div className="empty-state">
          <Activity size={18} />
          <span>Loading run history...</span>
        </div>
      ) : runs.length === 0 ? (
        <div className="empty-state">
          <Activity size={18} />
          <span>No run history yet. Run inference to populate history.</span>
        </div>
      ) : (
        <div className="scroll-x">
          <table className="table-base">
            <thead>
              <tr className="table-head-row">
                <th style={{ padding: "0.6rem 0.5rem" }}>Prompt</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Mode</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Δ Reliability</th>
                <th style={{ padding: "0.6rem 0.5rem" }}>Instability Reduction</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const delta = run.summary_metrics?.compare?.delta_reliability_score;
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
