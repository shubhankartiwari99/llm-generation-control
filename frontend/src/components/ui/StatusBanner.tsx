"use client";

interface StatusBannerProps {
  tone: "error" | "info";
  children: React.ReactNode;
}

export default function StatusBanner({ tone, children }: StatusBannerProps) {
  const toneClasses = tone === "error"
    ? "bg-error-container text-on-error-container border-error"
    : "bg-surface-elevated text-on-surface border-primary";

  return (
    <div 
      className={`border-l-4 rounded-r-lg p-4 font-body-md text-body-md shadow-sm ${toneClasses}`}
      role={tone === "error" ? "alert" : "status"} 
      aria-live="polite"
    >
      <div className="flex items-center gap-2">
        {tone === "error" && <span className="material-symbols-outlined" style={{fontFamily: 'Material Symbols Outlined'}}>error</span>}
        {tone === "info" && <span className="material-symbols-outlined" style={{fontFamily: 'Material Symbols Outlined'}}>info</span>}
        {children}
      </div>
    </div>
  );
}

