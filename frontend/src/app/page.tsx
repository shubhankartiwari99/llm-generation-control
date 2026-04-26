"use client";

import { useState } from "react";
import PromptInput from "@/components/PromptInput";
import ModeToggle from "@/components/ModeToggle";
import OutputPanel from "@/components/OutputPanel";
import EntropyChart from "@/components/EntropyChart";
import MetricsPanel from "@/components/MetricsPanel";
import StepTable from "@/components/StepTable";
import ExplanationPanel from "@/components/ExplanationPanel";
import { GenerateResponse, TokenStep } from "@/types";

export default function Home() {
  const [prompt, setPrompt] = useState("Write only blank lines");
  const [mode, setMode] = useState("compare"); // "plain", "adaptive", "compare"
  const [isLoading, setIsLoading] = useState(false);
  
  // State for the two panels
  const [plainData, setPlainData] = useState<GenerateResponse | null>(null);
  const [adaptiveData, setAdaptiveData] = useState<GenerateResponse | null>(null);

  const fetchGeneration = async (runMode: string) => {
    const res = await fetch("http://127.0.0.1:8000/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        mode: runMode,
        max_tokens: 40,
      }),
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return await res.json();
  };

  const runInference = async () => {
    setIsLoading(true);
    setPlainData(null);
    setAdaptiveData(null);
    
    try {
      if (mode === "plain" || mode === "compare") {
        const pData = await fetchGeneration("plain");
        setPlainData(pData);
      }
      
      // Run sequentially to save MPS memory
      if (mode === "adaptive" || mode === "compare") {
        const aData = await fetchGeneration("adaptive");
        setAdaptiveData(aData);
      }
    } catch (err) {
      console.error("Inference failed:", err);
      alert("Failed to run inference. Make sure the FastAPI server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  const showPlain = mode === "plain" || mode === "compare";
  const showAdaptive = mode === "adaptive" || mode === "compare";

  const getMetrics = (data: GenerateResponse | null) => {
    const steps = data?.steps || [];
    const instabilityCount = steps.filter(s => s.instability !== null).length;
    const avgEntropy = steps.length > 0 
      ? steps.reduce((acc, s) => acc + s.entropy, 0) / steps.length 
      : null;
    return { instabilityCount, avgEntropy };
  };

  const plainMetrics = getMetrics(plainData);
  const adaptiveMetrics = getMetrics(adaptiveData);

  return (
    <main className="container">
      <div style={{ textAlign: "center", marginBottom: "1rem" }}>
        <h1>LLM Generation Control</h1>
        <p className="text-secondary">Interactive control layer with token-level observability and adaptive decoding.</p>
      </div>

      <div className="header-row">
        <div className="controls-row">
          <label style={{ fontWeight: 500 }}>Execution Mode:</label>
          <select 
            value={mode} 
            onChange={(e) => setMode(e.target.value)}
            disabled={isLoading}
          >
            <option value="plain">Plain (No Control)</option>
            <option value="adaptive">Adaptive (Closed-Loop)</option>
            <option value="compare">Compare Both (Sequential)</option>
          </select>
        </div>
      </div>

      <PromptInput 
        prompt={prompt} 
        setPrompt={setPrompt} 
        onRun={runInference} 
        isLoading={isLoading} 
      />

      <div className="panels-grid">
        {showPlain && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <OutputPanel 
              title="Plain Generation" 
              output={plainData?.output || ""} 
              steps={plainData?.steps}
              isLoading={isLoading && !plainData} 
            />
            {plainData && (
              <>
                <MetricsPanel 
                  confidence={plainData.confidence}
                  instabilityCount={plainMetrics.instabilityCount}
                  regenerations={plainData.regenerations}
                  avgEntropy={plainMetrics.avgEntropy}
                />
                <ExplanationPanel mode="plain" steps={plainData.steps} regenerations={plainData.regenerations} />
                <div className="glass-panel panel-content" style={{ maxHeight: "300px", overflowY: "auto" }}>
                  <h4 style={{ margin: "0 0 1rem 0" }}>Plain Trace Log</h4>
                  <StepTable steps={plainData.steps} />
                </div>
              </>
            )}
          </div>
        )}
        
        {showAdaptive && (
          <div style={{ display: "flex", flexDirection: "column", gap: "1.5rem" }}>
            <OutputPanel 
              title="Adaptive Generation" 
              output={adaptiveData?.output || ""} 
              steps={adaptiveData?.steps}
              isLoading={isLoading && !adaptiveData && (!showPlain || plainData)} 
            />
            {adaptiveData && (
              <>
                <MetricsPanel 
                  confidence={adaptiveData.confidence}
                  instabilityCount={adaptiveMetrics.instabilityCount}
                  regenerations={adaptiveData.regenerations}
                  avgEntropy={adaptiveMetrics.avgEntropy}
                />
                <ExplanationPanel mode="adaptive" steps={adaptiveData.steps} regenerations={adaptiveData.regenerations} />
                <div className="glass-panel panel-content" style={{ maxHeight: "300px", overflowY: "auto" }}>
                  <h4 style={{ margin: "0 0 1rem 0" }}>Adaptive Trace Log</h4>
                  <StepTable steps={adaptiveData.steps} />
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Main Comparison Chart */}
      <EntropyChart 
        plainSteps={plainData?.steps} 
        adaptiveSteps={adaptiveData?.steps} 
      />

    </main>
  );
}
