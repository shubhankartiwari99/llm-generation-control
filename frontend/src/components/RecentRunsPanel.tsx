"use client";

import { RecentRun } from "@/types";

interface RecentRunsPanelProps {
  runs: RecentRun[];
  isLoading?: boolean;
}

export default function RecentRunsPanel({ runs, isLoading = false }: RecentRunsPanelProps) {
  return (
    <section className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col gap-4">
      <h3 className="font-headline-md text-headline-md text-text-primary m-0">Recent Runs</h3>
      {isLoading ? (
        <div className="flex flex-col items-center justify-center gap-3 text-text-secondary opacity-70 p-8">
          <span className="material-symbols-outlined animate-spin" style={{fontFamily: 'Material Symbols Outlined'}}>sync</span>
          <p className="font-label-sm text-label-sm m-0">Loading run history...</p>
        </div>
      ) : runs.length === 0 ? (
        <div className="flex flex-col items-center justify-center gap-3 text-text-secondary opacity-70 p-8">
          <span className="material-symbols-outlined" style={{fontFamily: 'Material Symbols Outlined'}}>history</span>
          <p className="font-label-sm text-label-sm m-0">No run history yet. Run inference to populate history.</p>
        </div>
      ) : (
        <div className="overflow-x-auto w-full">
          <table className="w-full border-collapse font-body-md text-body-md">
            <thead>
              <tr className="border-b border-border-subtle text-left text-text-secondary">
                <th className="py-3 px-2 font-medium">Prompt</th>
                <th className="py-3 px-2 font-medium">Mode</th>
                <th className="py-3 px-2 font-medium">Δ Reliability</th>
                <th className="py-3 px-2 font-medium">Instability Reduction</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border-subtle">
              {runs.map((run) => {
                const delta = run.summary_metrics?.compare?.delta_reliability;
                const reduced = run.summary_metrics?.compare?.instabilities_reduced_by;
                return (
                  <tr key={run.trace_id} className="hover:bg-surface-elevated/50 transition-colors">
                    <td className="py-3 px-2 max-w-[320px] truncate" title={run.prompt}>{run.prompt}</td>
                    <td className="py-3 px-2 capitalize">{run.mode}</td>
                    <td className="py-3 px-2 font-medium">
                      {typeof delta === "number" ? `${(delta * 100).toFixed(1)} pts` : "-"}
                    </td>
                    <td className="py-3 px-2 font-medium">
                      {typeof reduced === "number" ? reduced : "-"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
