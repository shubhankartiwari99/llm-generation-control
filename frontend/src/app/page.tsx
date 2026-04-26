"use client";

import { useState } from "react";
import PromptInput from "@/components/PromptInput";
import ModeToggle from "@/components/ModeToggle";
import OutputPanel from "@/components/OutputPanel";
import EntropyChart from "@/components/EntropyChart";
import MetricsPanel from "@/components/MetricsPanel";
import { GenerateResponse, TokenStep } from "@/types";

export default function Home() {
  const [prompt, setPrompt] = useState("Write only blank lines");
  const [mode, setMode] = useState("adaptive");
  const [isLoading, setIsLoading] = useState(false);
  
  // State for the two panels
  const [plainData, setPlainData] = useState<GenerateResponse | null>(null);
  const [adaptiveData, setAdaptiveData] = useState<GenerateResponse | null>(null);

  // Active data for charts/metrics based on mode
  const activeData = mode === "plain" ? plainData : adaptiveData;

  const runInference = async () => {
    setIsLoading(true);
    
    try {
      const res = await fetch("http://127.0.0.1:8000/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt,
          mode,
          max_tokens: 40,
        }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const data: GenerateResponse = await res.json();
      
      if (mode === "plain") {
        setPlainData(data);
      } else {
        setAdaptiveData(data);
      }
    } catch (err) {
      console.error("Inference failed:", err);
      alert("Failed to run inference. Make sure the FastAPI server is running on port 8000.");
    } finally {
      setIsLoading(false);
    }
  };

  // Derive metrics
  const steps = activeData?.steps || [];
  const instabilityCount = steps.filter(s => s.instability !== null).length;
  const avgEntropy = steps.length > 0 
    ? steps.reduce((acc, s) => acc + s.entropy, 0) / steps.length 
    : null;

  return (
    <main className="container">
      <div style={{ textAlign: "center", marginBottom: "1rem" }}>
        <h1>LLM Generation Control</h1>
        <p className="text-secondary">Interactive control layer with token-level observability and adaptive decoding.</p>
      </div>

      <div className="header-row">
        <ModeToggle mode={mode} setMode={setMode} isLoading={isLoading} />
      </div>

      <PromptInput 
        prompt={prompt} 
        setPrompt={setPrompt} 
        onRun={runInference} 
        isLoading={isLoading} 
      />

      <div className="panels-grid">
        <OutputPanel 
          title="Plain Generation" 
          output={plainData?.output || ""} 
          steps={plainData?.steps}
          isLoading={isLoading && mode === "plain"} 
        />
        <OutputPanel 
          title="Adaptive Generation" 
          output={adaptiveData?.output || ""} 
          steps={adaptiveData?.steps}
          isLoading={isLoading && mode === "adaptive"} 
        />
      </div>

      <EntropyChart steps={steps} />

      <MetricsPanel 
        confidence={activeData?.confidence ?? null}
        instabilityCount={instabilityCount}
        regenerations={activeData?.regenerations || 0}
        avgEntropy={avgEntropy}
      />
    </main>
  );
}
