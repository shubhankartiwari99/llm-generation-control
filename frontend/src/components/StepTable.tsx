"use client";

import { TokenStep } from "@/types";

interface StepTableProps {
  steps: TokenStep[];
}

export default function StepTable({ steps }: StepTableProps) {
  if (!steps || steps.length === 0) {
    return <div className="text-text-secondary text-center p-4 font-body-md text-body-md">No steps to display</div>;
  }

  return (
    <div className="w-full">
      <table className="w-full border-collapse font-body-md text-body-md">
        <thead>
          <tr className="border-b border-border-subtle text-left text-text-secondary">
            <th className="py-3 px-2 w-[15%] font-medium">Step</th>
            <th className="py-3 px-2 w-[45%] font-medium">Token</th>
            <th className="py-3 px-2 w-[10%] font-medium">Entropy</th>
            <th className="py-3 px-2 w-[20%] font-medium">⚠️ Alert</th>
            <th className="py-3 px-2 w-[15%] font-medium">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-border-subtle">
          {steps.map((step, idx) => {
            const hasInstability = step.instability !== null;
            const isLowEntropy = step.entropy < 1;
            const isHighEntropy = step.entropy > 5;
            const isRegeneration = step.action === "regenerate";

            const rowStyle = {
              backgroundColor: isRegeneration
                ? "rgba(16, 185, 129, 0.12)"
                : hasInstability
                  ? "rgba(239, 68, 68, 0.1)"
                  : "transparent",
            };
            
            let entropyColor = "text-text-primary";
            if (isLowEntropy) entropyColor = "text-[#f59e0b]";
            if (isHighEntropy) entropyColor = "text-[#10B981]";

            return (
              <tr key={idx} style={rowStyle} className="hover:bg-surface-elevated/50 transition-colors">
                <td className="py-2 px-2 text-text-secondary">{idx}</td>
                <td className="py-2 px-2">
                  <span className="bg-black/30 px-1.5 py-0.5 rounded text-on-surface font-label-md text-label-md inline-block">
                    {step.token}
                  </span>
                </td>
                <td className={`py-2 px-2 font-medium ${entropyColor}`}>
                  {step.entropy.toFixed(3)}
                </td>
                <td className="py-2 px-2 text-[#ef4444]">
                  {hasInstability ? `🔴 ${step.instability}` : isLowEntropy ? "🟡 low_entropy" : ""}
                </td>
                <td className={`py-2 px-2 ${step.action !== "continue" ? "text-[#f59e0b]" : "text-text-secondary"}`}>
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
