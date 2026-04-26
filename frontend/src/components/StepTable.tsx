"use client";

import { TokenStep } from "@/types";

interface StepTableProps {
  steps: TokenStep[];
}

export default function StepTable({ steps }: StepTableProps) {
  if (!steps || steps.length === 0) {
    return <div className="text-secondary text-center p-4">No steps to display</div>;
  }

  return (
    <div style={{ overflowX: "auto" }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
        <thead>
          <tr style={{ borderBottom: "1px solid var(--border)", textAlign: "left" }}>
            <th style={{ padding: "0.75rem 0.5rem", width: "15%" }}>Step</th>
            <th style={{ padding: "0.75rem 0.5rem", width: "45%" }}>Token</th>
            <th style={{ padding: "0.75rem 0.5rem", width: "10%" }}>Entropy</th>
            <th style={{ padding: "0.75rem 0.5rem", width: "20%" }}>⚠️ Alert</th>
            <th style={{ padding: "0.75rem 0.5rem", width: "15%" }}>Action</th>
          </tr>
        </thead>
        <tbody>
          {steps.map((step, idx) => {
            const hasInstability = step.instability !== null;
            const isLowEntropy = step.entropy < 1;
            const isHighEntropy = step.entropy > 5;
            const isRegeneration = step.action === "regenerate";

            const rowStyle = {
              borderBottom: "1px solid rgba(255,255,255,0.05)",
              backgroundColor: isRegeneration
                ? "rgba(34, 197, 94, 0.12)"
                : hasInstability
                  ? "rgba(239, 68, 68, 0.1)"
                  : "transparent",
            };
            
            let entropyColor = "var(--text-primary)";
            if (isLowEntropy) entropyColor = "var(--warning)";
            if (isHighEntropy) entropyColor = "var(--success)";

            return (
              <tr key={idx} style={rowStyle}>
                <td style={{ padding: "0.5rem", color: "var(--text-secondary)" }}>{idx}</td>
                <td style={{ padding: "0.5rem", fontFamily: "monospace" }}>
                  <span style={{ 
                    backgroundColor: "rgba(0,0,0,0.3)", 
                    padding: "0.1rem 0.3rem", 
                    borderRadius: "4px" 
                  }}>
                    {step.token}
                  </span>
                </td>
                <td style={{ padding: "0.5rem", color: entropyColor, fontWeight: 500 }}>
                  {step.entropy.toFixed(3)}
                </td>
                <td style={{ padding: "0.5rem", color: "var(--danger)" }}>
                  {hasInstability ? `🔴 ${step.instability}` : isLowEntropy ? "🟡 low_entropy" : ""}
                </td>
                <td style={{ padding: "0.5rem", color: step.action !== "continue" ? "var(--warning)" : "var(--text-secondary)" }}>
                  {isRegeneration ? "🟢 regenerate" : step.action || "continue"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
