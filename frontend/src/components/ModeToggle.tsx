"use client";

interface ModeToggleProps {
  mode: string;
  setMode: (mode: string) => void;
  isLoading: boolean;
}

export default function ModeToggle({ mode, setMode, isLoading }: ModeToggleProps) {
  return (
    <div className="controls-row">
      <label style={{ fontWeight: 500 }}>Decoding Mode:</label>
      <select 
        value={mode} 
        onChange={(e) => setMode(e.target.value)}
        disabled={isLoading}
      >
        <option value="plain">Plain (No Control)</option>
        <option value="adaptive">Adaptive (Closed-Loop)</option>
      </select>
    </div>
  );
}
