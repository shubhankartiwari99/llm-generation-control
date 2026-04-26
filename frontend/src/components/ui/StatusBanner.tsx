"use client";

interface StatusBannerProps {
  tone: "error" | "info";
  children: React.ReactNode;
}

export default function StatusBanner({ tone, children }: StatusBannerProps) {
  return (
    <div className={`status-banner ${tone}`} role={tone === "error" ? "alert" : "status"} aria-live="polite">
      {children}
    </div>
  );
}

