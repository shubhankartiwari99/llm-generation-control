"use client";

import { TokenStep } from "@/types";

interface ExplanationPanelProps {
  mode: string;
  steps: TokenStep[];
  regenerations: number;
}

export default function ExplanationPanel({ mode, steps, regenerations }: ExplanationPanelProps) {
  if (!steps || steps.length === 0) return null;

  const instabilities = steps.filter((s) => s.instability !== null);
  
  if (instabilities.length === 0 && regenerations === 0) {
    return (
      <div className="glass-panel" style={{ padding: "1rem", marginTop: "1rem", borderLeft: "4px solid var(--success)" }}>
        <p style={{ margin: 0 }}><strong>Status:</strong> Generation proceeded stably. No critical entropy collapse or repetition loops detected.</p>
      </div>
    );
  }

  return (
    <div className="glass-panel" style={{ padding: "1rem", marginTop: "1rem", borderLeft: "4px solid var(--danger)" }}>
      <p style={{ margin: "0 0 0.5rem 0", fontWeight: 600 }}>System Explanation:</p>
      <ul style={{ margin: 0, paddingLeft: "1.5rem", color: "var(--text-secondary)", lineHeight: 1.6 }}>
        {instabilities.slice(0, 3).map((inst, idx) => (
          <li key={idx}>
            <span className="text-danger">⚠️ {inst.instability?.replace("_", " ")}</span> detected around token <code style={{ color: "var(--text-primary)" }}>&quot;{inst.token}&quot;</code>.
          </li>
        ))}
        {instabilities.length > 3 && (
          <li>...and {instabilities.length - 3} more instability events.</li>
        )}
        
        {mode === "adaptive" && regenerations > 0 && (
          <li style={{ color: "var(--warning)", marginTop: "0.5rem" }}>
            <strong>Action Taken:</strong> Controller triggered {regenerations} regeneration(s) to escape the unstable distribution and reset temperature.
          </li>
        )}
        {mode === "plain" && instabilities.length > 0 && (
          <li style={{ color: "var(--danger)", marginTop: "0.5rem" }}>
            <strong>Action Taken:</strong> None (Plain mode). The model is permitted to degenerate.
          </li>
        )}
      </ul>
    </div>
  );
}
