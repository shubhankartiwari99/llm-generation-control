"use client";

import { GenerateSummary, TokenStep } from "@/types";

interface ExplanationPanelProps {
  mode: string;
  steps: TokenStep[];
  regenerations: number;
  summary?: GenerateSummary;
}

export default function ExplanationPanel({ mode, steps, regenerations, summary }: ExplanationPanelProps) {
  if (!steps || steps.length === 0) return null;

  const instabilities = steps.filter((s) => s.instability !== null);
  
  if (instabilities.length === 0 && regenerations === 0) {
    return (
      <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 border-l-4 border-l-[#10B981]">
        <p className="m-0 font-body-md text-body-md text-on-surface">
          <strong className="font-headline-md text-headline-md">Status:</strong> Generation proceeded stably. No critical entropy collapse or repetition loops detected.
          {summary?.compare?.delta_reliability !== undefined && (
            <> Delta reliability_score vs plain: {(summary.compare.delta_reliability * 100).toFixed(1)} pts.</>
          )}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 border-l-4 border-l-[#ef4444]">
      <p className="m-0 mb-3 font-headline-md text-headline-md text-text-primary">System Explanation:</p>
      <ul className="m-0 pl-6 text-text-secondary font-body-md text-body-md leading-relaxed space-y-2 list-disc marker:text-text-secondary">
        {instabilities.slice(0, 3).map((inst, idx) => (
          <li key={idx}>
            <span className="text-[#ef4444]">⚠️ {inst.instability?.replace("_", " ")}</span> detected around token <code className="text-on-surface bg-surface-deep px-1.5 py-0.5 rounded font-label-md text-label-md">&quot;{inst.token}&quot;</code>.
          </li>
        ))}
        {instabilities.length > 3 && (
          <li>...and {instabilities.length - 3} more instability events.</li>
        )}
        
        {mode === "adaptive" && regenerations > 0 && (
          <li className="text-[#f59e0b] mt-2">
            <strong className="font-headline-sm text-headline-sm">Action Taken:</strong> Controller triggered {regenerations} regeneration(s) to escape the unstable distribution and reset temperature.
          </li>
        )}
        {summary?.compare?.instabilities_reduced_by !== undefined && (
          <li className="text-[#10B981] mt-2">
            <strong className="font-headline-sm text-headline-sm">Measured Gain:</strong> Instabilities reduced by {summary.compare.instabilities_reduced_by} and reliability_score shifted by {((summary.compare.delta_reliability ?? 0) * 100).toFixed(1)} pts versus plain generation.
          </li>
        )}
        {mode === "plain" && instabilities.length > 0 && (
          <li className="text-[#ef4444] mt-2">
            <strong className="font-headline-sm text-headline-sm">Action Taken:</strong> None (Plain mode). The model is permitted to degenerate.
          </li>
        )}
      </ul>
    </div>
  );
}
