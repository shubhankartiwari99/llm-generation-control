"use client";

interface PromptInputProps {
  prompt: string;
  setPrompt: (value: string) => void;
  onRun: () => void;
  isLoading: boolean;
}

export default function PromptInput({ prompt, setPrompt, onRun, isLoading }: PromptInputProps) {
  return (
    <div className="glass-panel" style={{ padding: "1.5rem" }}>
      <h2 style={{ marginBottom: "1rem" }}>Input Prompt</h2>
      <textarea
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter prompt here..."
        disabled={isLoading}
      />
      <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end" }}>
        <button onClick={onRun} disabled={isLoading || !prompt.trim()}>
          {isLoading ? "Generating..." : "Run Inference"}
        </button>
      </div>
    </div>
  );
}
