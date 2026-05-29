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
    <section className="bg-surface-container-lowest border border-border-subtle rounded-xl p-6 flex flex-col gap-4">
      <h2 className="font-headline-md text-headline-md text-text-primary">Input Prompt</h2>
      <div className="flex flex-col gap-2 relative">
        <label htmlFor={promptId} className="font-label-sm text-label-sm text-text-secondary">
          Prompt text (max 2000 chars)
        </label>
        <textarea
          id={promptId}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Enter prompt here..."
          disabled={isLoading}
          aria-describedby={promptHintId}
          className="w-full bg-surface-deep border border-border-subtle rounded-lg p-4 font-body-md text-body-md text-on-surface focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container resize-none h-[120px]"
        />
        <div className="absolute bottom-3 right-3 opacity-30 pointer-events-none">
          <svg fill="none" height="10" viewBox="0 0 10 10" width="10" xmlns="http://www.w3.org/2000/svg">
            <path d="M10 0L0 10H10V0Z" fill="currentColor"></path>
          </svg>
        </div>
      </div>
      <div className="flex justify-between items-end mt-2">
        <div className="flex flex-col gap-1">
          <p id={promptHintId} className="font-label-sm text-label-sm text-text-secondary m-0">
            Use compare mode to quantify control gains automatically.
          </p>
        </div>
        <div className="flex items-center gap-6">
          <span className="font-label-sm text-label-sm text-text-secondary" aria-live="polite">
            {prompt.length}/2000
          </span>
          <button 
            onClick={onRun} 
            disabled={isLoading || !prompt.trim()}
            className="bg-primary-container text-on-primary-container font-label-md text-label-md px-6 py-2.5 rounded-lg hover:bg-inverse-primary transition-colors focus:ring-2 focus:ring-primary-container focus:ring-offset-2 focus:ring-offset-surface-deep disabled:opacity-50"
          >
            {isLoading ? "Generating..." : "Run Inference"}
          </button>
        </div>
      </div>
    </section>
  );
}
