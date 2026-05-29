import type { Metadata } from "next";
import { Hanken_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const hanken = Hanken_Grotesk({
  variable: "--font-hanken",
  subsets: ["latin"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "LLM Generation Control Dashboard",
  description:
    "Interactive LLM control system with token-level observability, instability detection, and adaptive intervention.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${hanken.variable} ${jetbrains.variable} dark`}>
      <head>
        <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet" />
      </head>
      <body className="bg-surface-deep text-on-surface font-body-md min-h-screen flex flex-col">
        {/* TopAppBar */}
        <header className="bg-surface-deep border-b border-border-subtle docked full-width top-0 z-50">
          <div className="flex justify-between items-center w-full px-gutter h-16">
            {/* Brand */}
            <div className="font-headline-lg text-headline-lg font-bold text-on-surface">
              LLM Control
            </div>
            {/* Navigation */}
            <nav className="hidden md:flex items-center gap-6">
              <a className="text-primary border-b-2 border-primary pb-1 font-label-md text-label-md hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80" href="#">Dashboard</a>
              <a className="text-on-surface-variant hover:text-on-surface font-label-md text-label-md hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80" href="#">Evaluations</a>
              <a className="text-on-surface-variant hover:text-on-surface font-label-md text-label-md hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80" href="#">Traces</a>
              <a className="text-on-surface-variant hover:text-on-surface font-label-md text-label-md hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80" href="#">Models</a>
            </nav>
            {/* Trailing Icons */}
            <div className="flex items-center gap-4 text-primary">
              <button className="hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80 p-2 rounded-full flex items-center justify-center">
                <span className="material-symbols-outlined" style={{fontFamily: 'Material Symbols Outlined'}}>settings</span>
              </button>
              <button className="hover:bg-surface-container transition-colors duration-200 scale-95 active:opacity-80 p-2 rounded-full flex items-center justify-center">
                <span className="material-symbols-outlined" style={{fontFamily: 'Material Symbols Outlined'}}>account_circle</span>
              </button>
            </div>
          </div>
        </header>

        {children}

        {/* Footer Component */}
        <footer className="bg-surface-deep border-t border-border-subtle docked full-width mt-auto">
          <div className="flex justify-between items-center w-full px-gutter py-4">
            <div className="text-text-secondary font-label-sm text-label-sm">
              v1.4.2-stable • System Latency: 12ms
            </div>
            <div className="flex gap-6">
              <a className="text-text-secondary hover:text-on-surface font-label-sm text-label-sm transition-colors" href="#">Documentation</a>
              <a className="text-text-secondary hover:text-on-surface font-label-sm text-label-sm transition-colors" href="#">API Reference</a>
              <a className="text-text-secondary hover:text-on-surface font-label-sm text-label-sm transition-colors" href="#">Support</a>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
