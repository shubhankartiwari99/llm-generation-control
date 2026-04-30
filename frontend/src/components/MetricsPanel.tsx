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
    if (conf === null) return "text-secondary";
    if (conf >= 0.7) return "text-success";
    if (conf >= 0.4) return "text-warning";
    return "text-danger";
  };

  return (
    <div className="glass-panel panel-content">
      <h3 style={{ marginBottom: "1rem" }}>Run Metrics</h3>
      
      <div className="metrics-grid">
        <div className="metric-card">
          <div className="metric-label">Reliability</div>
          <div className={`metric-value ${getReliabilityColor(reliability_score)}`}>
            {reliability_score !== null ? (reliability_score * 100).toFixed(1) + "%" : "-"}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Instabilities</div>
          <div className={`metric-value ${instabilityCount > 0 ? "text-danger" : "text-success"}`}>
            {instabilityCount > 0 && "⚠️ "}
            {instabilityCount}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Regenerations</div>
          <div className={`metric-value ${regenerations > 0 ? "text-warning" : "text-secondary"}`}>
            {regenerations > 0 && "🔁 "}
            {regenerations}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Avg Entropy</div>
          <div className="metric-value text-secondary">
            {avgEntropy !== null ? avgEntropy.toFixed(2) : "-"}
          </div>
        </div>

        <div className="metric-card">
          <div className="metric-label">Latency</div>
          <div className="metric-value text-secondary">
            {latencyMs ? `${(latencyMs / 1000).toFixed(1)}s` : "-"}
          </div>
        </div>
      </div>
    </div>
  );
}
