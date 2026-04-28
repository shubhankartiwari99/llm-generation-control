export interface TokenStep {
  token: string;
  entropy: number;
  instability: string | null;
  action: string | null;
}

export interface ModeResponse {
  text: string;
  steps: TokenStep[];
  confidence: number;
  regenerations: number;
  trace_available: boolean;
}

export interface ModeSummary {
  confidence: number;
  instabilities: number;
  regenerations: number;
  avg_entropy: number | null;
  max_entropy: number | null;
  min_entropy: number | null;
  trace_available?: boolean;
  note?: string;
}

export interface CompareSummary {
  delta_confidence: number;
  instabilities_reduced_by: number;
  regeneration_gain: number;
}

export interface GenerateSummary {
  plain: Partial<ModeSummary>;
  adaptive: Partial<ModeSummary>;
  compare: Partial<CompareSummary>;
}

export interface GenerateResponse {
  plain: ModeResponse | null;
  adaptive: ModeResponse | null;
  summary: GenerateSummary;
  latency_ms: number;
  model: string;
  device: string;
}

export interface RecentRun {
  trace_id: string;
  timestamp: string;
  prompt: string;
  mode: string;
  confidence: number | null;
  regenerations: number;
  instabilities: number;
  summary_metrics?: GenerateSummary;
}

export interface RecentRunsResponse {
  runs: RecentRun[];
}
