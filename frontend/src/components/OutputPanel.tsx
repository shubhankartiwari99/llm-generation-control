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
    <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col gap-8 min-h-[240px]">
      <h3 className="font-headline-md text-headline-md text-text-primary m-0">{title}</h3>
      <div className={`font-label-md text-label-md flex-1 ${isLoading ? "opacity-50 animate-pulse" : ""} overflow-y-auto`}>
        {steps ? (
          <div className="whitespace-pre-wrap leading-relaxed">
            {steps.map((step, idx) => {
              const isHighEntropy = step.entropy > 5;
              const isLowEntropy = step.entropy < 1;
              const hasInstability = step.instability !== null;
              
              let backgroundColor = "transparent";
              if (hasInstability) backgroundColor = "rgba(255, 180, 171, 0.2)"; // On-error-container tint
              else if (isLowEntropy) backgroundColor = "rgba(255, 182, 141, 0.15)"; // Tertiary tint
              else if (isHighEntropy) backgroundColor = "rgba(185, 200, 222, 0.1)"; // Secondary tint

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
            })}
          </div>
        ) : (
          output || (
            <div className="flex flex-col items-center justify-center gap-3 text-text-secondary opacity-70 h-full">
              <span className="material-symbols-outlined" style={{fontVariationSettings: "'FILL' 0", fontSize: "24px", fontFamily: 'Material Symbols Outlined'}}>description</span>
              <p className="font-label-sm text-label-sm m-0">No output yet. Run inference to view generated tokens.</p>
            </div>
          )
        )}
      </div>
    </div>
  );
}
