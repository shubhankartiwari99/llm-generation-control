"use client";

import { TokenStep } from "@/types";

interface OutputPanelProps {
  title: string;
  output: string;
  steps?: TokenStep[];
  isLoading: boolean;
}

export default function OutputPanel({ title, output, steps, isLoading }: OutputPanelProps) {
  return (
    <div className="glass-panel panel-content">
      <h3 style={{ marginBottom: "1rem" }}>{title}</h3>
      <div className={`token-stream ${isLoading ? "loading" : ""}`} style={{ flex: 1, overflowY: "auto" }}>
        {steps ? (
          steps.map((step, idx) => {
            const isHighEntropy = step.entropy > 5;
            const isLowEntropy = step.entropy < 1;
            const hasInstability = step.instability !== null;
            
            let backgroundColor = "transparent";
            if (hasInstability) backgroundColor = "rgba(239, 68, 68, 0.2)"; // Red tint
            else if (isLowEntropy) backgroundColor = "rgba(245, 158, 11, 0.15)"; // Yellow tint
            else if (isHighEntropy) backgroundColor = "rgba(16, 185, 129, 0.1)"; // Green tint

            return (
              <span 
                key={idx} 
                style={{ backgroundColor, padding: "0 2px", borderRadius: "2px" }}
                title={`Entropy: ${step.entropy.toFixed(2)}${step.instability ? ` | ${step.instability}` : ""}`}
              >
                {step.token}
                {hasInstability && <span style={{ fontSize: "0.8em", margin: "0 2px" }}>⚠️</span>}
              </span>
            );
          })
        ) : (
          output || <span className="text-secondary">No output yet...</span>
        )}
      </div>
    </div>
  );
}
