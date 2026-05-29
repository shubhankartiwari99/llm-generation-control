"use client";

interface ModeToggleProps {
  mode: string;
  setMode: (mode: string) => void;
  isLoading: boolean;
}

export default function ModeToggle({ mode, setMode, isLoading }: ModeToggleProps) {
  const modeSelectId = "mode-select";

  return (
    <section className="flex items-center gap-4">
      <label htmlFor={modeSelectId} className="font-body-md text-body-md text-on-surface">Decoding Mode:</label>
      <div className="relative">
        <select 
          id={modeSelectId}
          value={mode} 
          onChange={(e) => setMode(e.target.value)}
          disabled={isLoading}
          className="appearance-none bg-surface-container-lowest border border-border-subtle text-on-surface font-body-md text-body-md rounded-lg pl-4 pr-10 py-2 focus:outline-none focus:border-primary-container focus:ring-1 focus:ring-primary-container disabled:opacity-50"
        >
          <option value="compare">Compare (Plain + Adaptive)</option>
          <option value="plain">Plain (No Control)</option>
          <option value="adaptive">Adaptive (Closed-Loop)</option>
        </select>
        <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-3 text-on-surface-variant">
          <span className="material-symbols-outlined text-sm" style={{fontFamily: 'Material Symbols Outlined'}}>expand_more</span>
        </div>
      </div>
    </section>
  );
}
