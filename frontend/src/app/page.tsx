"use client";

import { useState } from "react";
import PromptInput from "@/components/PromptInput";
import ModeToggle from "@/components/ModeToggle";
import OutputPanel from "@/components/OutputPanel";
import EntropyChart from "@/components/EntropyChart";
import MetricsPanel from "@/components/MetricsPanel";
import StepTable from "@/components/StepTable";
import ExplanationPanel from "@/components/ExplanationPanel";
import RecentRunsPanel from "@/components/RecentRunsPanel";
import SectionHeader from "@/components/ui/SectionHeader";
import StatusBanner from "@/components/ui/StatusBanner";
import { GenerateResponse, RecentRun, RecentRunsResponse, TokenStep } from "@/types";

export default function Home() {
  const [prompt, setPrompt] = useState("Write only blank lines");
  const [mode, setMode] = useState("compare");
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<GenerateResponse | null>(null);
  const [recentRuns, setRecentRuns] = useState<RecentRun[]>([]);
  const [isHistoryLoading, setIsHistoryLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fetchRecentRuns = async () => {
    setIsHistoryLoading(true);
    try {
      const res = await fetch("/api/runs/recent?limit=8");
      if (!res.ok) return;
      const result: RecentRunsResponse = await res.json();
      setRecentRuns(result.runs || []);
    } catch {
      // Keep dashboard usable even if history endpoint is unavailable.
    } finally {
      setIsHistoryLoading(false);
    }
  };

  const runInference = async () => {
    setIsLoading(true);
    setData(null);
    setErrorMessage(null);
    
    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_tokens: 40,
          mode,
        }),
      });
      
      if (!res.ok) {
        let detail = `API error: ${res.status}`;
        try {
          const errData = await res.json();
          detail = errData.detail || detail;
        } catch {
          // Use default message if JSON parsing fails.
        }
        throw new Error(detail);
      }
      
      const result = await res.json();
      setData(result);
      await fetchRecentRuns();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      console.error("Inference failed:", err);
      setErrorMessage(message);
    } finally {
      setIsLoading(false);
    }
  };

  const getInstabilityCount = (steps: TokenStep[] = []) => steps.filter(s => s.instability !== null).length;
  const getAvgEntropy = (steps: TokenStep[] = []) => steps.length > 0 
    ? steps.reduce((acc, s) => acc + s.entropy, 0) / steps.length 
    : 0;

  return (
    <main className="container">
      <div className="hero-header">
        <h1>LLM Generation Control Dashboard</h1>
        <p className="text-secondary">Interactive control layer with real-time hardware observability (Mistral 7B on MPS).</p>
      </div>

      {errorMessage && (
        <StatusBanner tone="error">
          Generation failed: {errorMessage}
        </StatusBanner>
      )}

      <PromptInput 
        prompt={prompt} 
        setPrompt={setPrompt} 
        onRun={runInference} 
        isLoading={isLoading} 
      />
      <div className="mt-1">
        <ModeToggle mode={mode} setMode={setMode} isLoading={isLoading} />
      </div>

      <div className="section mt-2">
        <SectionHeader title="Generation Output" subtitle={`Mode: ${mode}`} />
      </div>
      <div className="panels-grid">
        <OutputPanel 
          title="Plain Generation" 
          output={data?.plain?.text || ""} 
          steps={data?.plain?.steps}
          isLoading={isLoading && !data} 
        />
        <OutputPanel 
          title="Adaptive Generation (Controlled)" 
          output={data?.adaptive?.text || ""} 
          steps={data?.adaptive?.steps}
          isLoading={isLoading && !data} 
        />
      </div>

      <div className="section mt-2">
        <SectionHeader title="Trace Analysis" subtitle="Token-level entropy + interventions" />
        <EntropyChart 
          plainSteps={data?.plain?.steps}
          adaptiveSteps={data?.adaptive?.steps}
        />
      </div>

      {data && (
        <>
          {/* Step Trace Table - Showing Adaptive trace details */}
          <div className="glass-panel panel-content mt-2" style={{ maxHeight: "400px", overflowY: "auto" }}>
            <h3 style={{ margin: "0 0 1rem 0" }}>Adaptive Step Trace</h3>
            <StepTable steps={data.adaptive?.steps || []} />
          </div>

          <div className="panels-grid" style={{ marginTop: "2rem" }}>
            {/* Metrics Panel */}
            <div>
              <h3 style={{ marginBottom: "1rem" }}>Performance Metrics</h3>
              <MetricsPanel 
                reliability_score={data.adaptive?.reliability_score ?? null}
                instabilityCount={getInstabilityCount(data.adaptive?.steps)}
                regenerations={data.adaptive?.regenerations || 0}
                avgEntropy={getAvgEntropy(data.adaptive?.steps)}
                latencyMs={data.latency_ms}
              />
              {data.summary?.compare?.delta_reliability_score !== undefined && (
                <div className="glass-panel" style={{ marginTop: "1rem", padding: "0.75rem", fontSize: "0.9rem" }}>
                  Δ reliability_score: {(data.summary.compare.delta_reliability_score * 100).toFixed(1)} pts | Instabilities reduced: {data.summary.compare.instabilities_reduced_by ?? 0}
                </div>
              )}
              <div style={{ marginTop: "1rem", fontSize: "0.85rem" }} className="text-secondary">
                Hardware: {data.model.toUpperCase()} ({data.device.toUpperCase()})
              </div>
            </div>

            {/* Insight/Explanation Panel */}
            <div>
              <h3 style={{ marginBottom: "1rem" }}>System Insights</h3>
              <ExplanationPanel 
                mode={mode}
                steps={data.adaptive?.steps || []}
                regenerations={data.adaptive?.regenerations || 0}
                summary={data.summary}
              />
            </div>
          </div>
        </>
      )}

      <div className="actions-end mt-2">
        <button onClick={() => void fetchRecentRuns()} disabled={isLoading}>
          {isHistoryLoading ? "Refreshing..." : "Refresh Run History"}
        </button>
      </div>
      <RecentRunsPanel runs={recentRuns} isLoading={isHistoryLoading} />
    </main>
  );
}
