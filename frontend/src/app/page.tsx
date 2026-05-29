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
    <main className="flex-1 w-full max-w-7xl mx-auto px-gutter py-margin-desktop flex flex-col gap-10">
      <section className="text-center flex flex-col items-center gap-2">
        <h1 className="font-headline-xl text-headline-xl text-text-primary">LLM Generation Control Dashboard</h1>
        <p className="font-body-md text-body-md text-text-secondary">Interactive control layer with real-time hardware observability (Mistral 7B on MPS).</p>
      </section>

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
      
      <ModeToggle mode={mode} setMode={setMode} isLoading={isLoading} />

      <section className="flex flex-col gap-4">
        <div className="flex justify-between items-end">
          <h2 className="font-headline-md text-headline-md text-text-primary">Generation Output</h2>
          <span className="font-label-sm text-label-sm text-text-secondary">Mode: {mode}</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
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
      </section>

      <section className="flex flex-col gap-4">
        <div className="flex justify-between items-end">
          <h2 className="font-headline-md text-headline-md text-text-primary">Trace Analysis</h2>
          <span className="font-label-sm text-label-sm text-text-secondary">Token-level entropy + interventions</span>
        </div>
        <EntropyChart 
          plainSteps={data?.plain?.steps}
          adaptiveSteps={data?.adaptive?.steps}
        />
      </section>

      {data && (
        <>
          {/* Step Trace Table - Showing Adaptive trace details */}
          <section className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col gap-4 max-h-[400px] overflow-y-auto scroll-x">
            <h3 className="font-headline-md text-headline-md text-text-primary m-0 mb-2">Adaptive Step Trace</h3>
            <StepTable steps={data.adaptive?.steps || []} />
          </section>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-2">
            {/* Metrics Panel */}
            <div className="flex flex-col gap-4">
              <h3 className="font-headline-md text-headline-md text-text-primary m-0">Performance Metrics</h3>
              <MetricsPanel 
                reliability_score={data.adaptive?.reliability_score ?? null}
                instabilityCount={getInstabilityCount(data.adaptive?.steps)}
                regenerations={data.adaptive?.regenerations || 0}
                avgEntropy={getAvgEntropy(data.adaptive?.steps)}
                latencyMs={data.latency_ms}
              />
              {data.summary?.compare?.delta_reliability !== undefined && (
                <div className="bg-surface-container border border-border-subtle p-3 rounded-lg font-body-md text-body-md text-on-surface">
                  Δ reliability_score: {(data.summary.compare.delta_reliability * 100).toFixed(1)} pts | Instabilities reduced: {data.summary.compare.instabilities_reduced_by ?? 0}
                </div>
              )}
              <div className="font-label-sm text-label-sm text-text-secondary">
                Hardware: {data.model.toUpperCase()} ({data.device.toUpperCase()})
              </div>
            </div>

            {/* Insight/Explanation Panel */}
            <div className="flex flex-col gap-4">
              <h3 className="font-headline-md text-headline-md text-text-primary m-0">System Insights</h3>
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

      <section className="flex justify-end pt-4">
        <button 
          onClick={() => void fetchRecentRuns()} 
          disabled={isLoading}
          className="bg-primary-container text-on-primary-container font-label-md text-label-md px-6 py-2.5 rounded-lg hover:bg-inverse-primary transition-colors focus:ring-2 focus:ring-primary-container focus:ring-offset-2 focus:ring-offset-surface-deep disabled:opacity-50"
        >
          {isHistoryLoading ? "Refreshing..." : "Refresh Run History"}
        </button>
      </section>
      
      <RecentRunsPanel runs={recentRuns} isLoading={isHistoryLoading} />
    </main>
  );
}
