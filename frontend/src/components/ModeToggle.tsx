"use client";

interface ModeToggleProps {
  mode: string;
  setMode: (mode: string) => void;
  isLoading: boolean;
}

export default function ModeToggle({ mode, setMode, isLoading }: ModeToggleProps) {
  const modeSelectId = "mode-select";

  return (
    <div className="controls-row">
      <label htmlFor={modeSelectId} style={{ fontWeight: 500 }}>Decoding Mode:</label>
      <select 
        id={modeSelectId}
        value={mode} 
        onChange={(e) => setMode(e.target.value)}
        disabled={isLoading}
      >
        <option value="compare">Compare (Plain + Adaptive)</option>
        <option value="plain">Plain (No Control)</option>
        <option value="adaptive">Adaptive (Closed-Loop)</option>
      </select>
    </div>
  );
}
