"use client";

interface MetricsPanelProps {
  confidence: number | null;
  instabilityCount: number;
  regenerations: number;
  avgEntropy: number | null;
}

export default function MetricsPanel({ confidence, instabilityCount, regenerations, avgEntropy }: MetricsPanelProps) {
  const getConfidenceColor = (conf: number | null) => {
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
          <div className="metric-label">Confidence</div>
          <div className={`metric-value ${getConfidenceColor(confidence)}`}>
            {confidence !== null ? (confidence * 100).toFixed(1) + "%" : "-"}
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
      </div>
    </div>
  );
}
