"use client";

interface PromptInputProps {
  prompt: string;
  setPrompt: (value: string) => void;
  onRun: () => void;
  isLoading: boolean;
}

export default function PromptInput({ prompt, setPrompt, onRun, isLoading }: PromptInputProps) {
  const promptId = "prompt-input";
  const promptHintId = "prompt-input-hint";

  return (
    <div className="glass-panel panel-content">
      <h2 style={{ marginBottom: "1rem" }}>Input Prompt</h2>
      <label htmlFor={promptId} className="small-note">
        Prompt text (max 2000 chars)
      </label>
      <textarea
        id={promptId}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Enter prompt here..."
        disabled={isLoading}
        aria-describedby={promptHintId}
      />
      <div className="hint-row">
        <div id={promptHintId} className="small-note">
          Use compare mode to quantify control gains automatically.
        </div>
        <div className="small-note" aria-live="polite">
          {prompt.length}/2000
        </div>
      </div>
      <div className="actions-end">
        <button onClick={onRun} disabled={isLoading || !prompt.trim()}>
          {isLoading ? "Generating..." : "Run Inference"}
        </button>
      </div>
    </div>
  );
}
