"use client";

interface MetricsPanelProps {
  reliability_score: number | null;
  instabilityCount: number;
  regenerations: number;
  avgEntropy: number | null;
  latencyMs?: number;
}

export default function MetricsPanel({ reliability_score, instabilityCount, regenerations, avgEntropy, latencyMs }: MetricsPanelProps) {
  const getReliabilityColor = (conf: number | null) => {
    if (conf === null) return "text-text-secondary";
    if (conf >= 0.7) return "text-[#10B981]";
    if (conf >= 0.4) return "text-[#f59e0b]";
    return "text-[#ef4444]";
  };

  return (
    <div className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="bg-surface-elevated rounded-lg p-4 flex flex-col items-center justify-center text-center">
          <div className="font-label-sm text-label-sm text-text-secondary uppercase tracking-wider">Reliability</div>
          <div className={`font-headline-lg text-headline-lg mt-2 ${getReliabilityColor(reliability_score)}`}>
            {reliability_score !== null ? (reliability_score * 100).toFixed(1) + "%" : "-"}
          </div>
        </div>

        <div className="bg-surface-elevated rounded-lg p-4 flex flex-col items-center justify-center text-center">
          <div className="font-label-sm text-label-sm text-text-secondary uppercase tracking-wider">Instabilities</div>
          <div className={`font-headline-lg text-headline-lg mt-2 ${instabilityCount > 0 ? "text-[#ef4444]" : "text-[#10B981]"}`}>
            {instabilityCount > 0 && <span className="mr-1 text-base">⚠️</span>}
            {instabilityCount}
          </div>
        </div>

        <div className="bg-surface-elevated rounded-lg p-4 flex flex-col items-center justify-center text-center">
          <div className="font-label-sm text-label-sm text-text-secondary uppercase tracking-wider">Regenerations</div>
          <div className={`font-headline-lg text-headline-lg mt-2 ${regenerations > 0 ? "text-[#f59e0b]" : "text-text-secondary"}`}>
            {regenerations > 0 && <span className="mr-1 text-base">🔁</span>}
            {regenerations}
          </div>
        </div>

        <div className="bg-surface-elevated rounded-lg p-4 flex flex-col items-center justify-center text-center">
          <div className="font-label-sm text-label-sm text-text-secondary uppercase tracking-wider">Avg Entropy</div>
          <div className="font-headline-lg text-headline-lg mt-2 text-text-secondary">
            {avgEntropy !== null ? avgEntropy.toFixed(2) : "-"}
          </div>
        </div>

        <div className="bg-surface-elevated rounded-lg p-4 flex flex-col items-center justify-center text-center">
          <div className="font-label-sm text-label-sm text-text-secondary uppercase tracking-wider">Latency</div>
          <div className="font-headline-lg text-headline-lg mt-2 text-text-secondary">
            {latencyMs ? `${(latencyMs / 1000).toFixed(1)}s` : "-"}
          </div>
        </div>
      </div>
    </div>
  );
}
