"use client";

import { useState } from "react";
import PromptInput from "@/components/PromptInput";
import OutputPanel from "@/components/OutputPanel";
import EntropyChart from "@/components/EntropyChart";
import MetricsPanel from "@/components/MetricsPanel";
import StepTable from "@/components/StepTable";
import ExplanationPanel from "@/components/ExplanationPanel";
import { GenerateResponse } from "@/types";

export default function Home() {
  const [prompt, setPrompt] = useState("Write only blank lines");
  const [isLoading, setIsLoading] = useState(false);
  const [data, setData] = useState<GenerateResponse | null>(null);

  const runInference = async () => {
    setIsLoading(true);
    setData(null);
    
    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          max_tokens: 40,
        }),
      });
      
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || `API error: ${res.status}`);
      }
      
      const result = await res.json();
      setData(result);
    } catch (err: any) {
      console.error("Inference failed:", err);
      alert(`Failed to run inference: ${err.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const getInstabilityCount = (steps: any[]) => steps.filter(s => s.instability !== null).length;
  const getAvgEntropy = (steps: any[]) => steps.length > 0 
    ? steps.reduce((acc, s) => acc + s.entropy, 0) / steps.length 
    : 0;

  return (
    <main className="container">
      <div style={{ textAlign: "center", marginBottom: "1.5rem" }}>
        <h1>LLM Generation Control Dashboard</h1>
        <p className="text-secondary">Interactive control layer with real-time hardware observability (Mistral 7B on MPS).</p>
      </div>

      <PromptInput 
        prompt={prompt} 
        setPrompt={setPrompt} 
        onRun={runInference} 
        isLoading={isLoading} 
      />

      {/* Phase 2: Side-by-Side Comparison Output */}
      <div className="panels-grid" style={{ marginTop: "2rem" }}>
        <OutputPanel 
          title="Plain Generation" 
          output={data?.plain.text || ""} 
          steps={data?.plain.steps}
          isLoading={isLoading && !data} 
        />
        <OutputPanel 
          title="Adaptive Generation (Controlled)" 
          output={data?.adaptive.text || ""} 
          steps={data?.adaptive.steps}
          isLoading={isLoading && !data} 
        />
      </div>

      {/* Entropy Comparison Chart */}
      <div style={{ marginTop: "2rem" }}>
        <h3 style={{ marginBottom: "1rem" }}>Entropy Trace Comparison</h3>
        <EntropyChart 
          plainSteps={data?.plain.steps} 
          adaptiveSteps={data?.adaptive.steps} 
        />
      </div>

      {data && (
        <>
          {/* Step Trace Table - Showing Adaptive trace details */}
          <div className="glass-panel panel-content" style={{ marginTop: "2rem", maxHeight: "400px", overflowY: "auto" }}>
            <h3 style={{ margin: "0 0 1rem 0" }}>Adaptive Step Trace</h3>
            <StepTable steps={data.adaptive.steps} />
          </div>

          <div className="panels-grid" style={{ marginTop: "2rem" }}>
            {/* Metrics Panel */}
            <div>
              <h3 style={{ marginBottom: "1rem" }}>Performance Metrics</h3>
              <MetricsPanel 
                confidence={data.adaptive.confidence}
                instabilityCount={getInstabilityCount(data.adaptive.steps)}
                regenerations={data.adaptive.regenerations}
                avgEntropy={getAvgEntropy(data.adaptive.steps)}
                latencyMs={data.latency_ms}
              />
              <div style={{ marginTop: "1rem", fontSize: "0.85rem" }} className="text-secondary">
                Hardware: {data.model.toUpperCase()} ({data.device.toUpperCase()})
              </div>
            </div>

            {/* Insight/Explanation Panel */}
            <div>
              <h3 style={{ marginBottom: "1rem" }}>System Insights</h3>
              <ExplanationPanel 
                mode="adaptive" 
                steps={data.adaptive.steps} 
                regenerations={data.adaptive.regenerations} 
              />
            </div>
          </div>
        </>
      )}
    </main>
  );
}
