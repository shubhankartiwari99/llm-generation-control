export interface TokenStep {
  token: string;
  entropy: number;
  instability: string | null;
  action: string | null;
  temperature: number;
}

export interface ModeResponse {
  text: string;
  steps: TokenStep[];
  reliability_score: number | null;
  reliability_type: string;
  confidence_breakdown?: Record<string, number>;
  regenerations: number;
  control_metrics?: {
    tokens_generated: number;
    interventions: number;
    temperature_adjustments: number;
  };
  steps_available: boolean;
  trace_available: boolean;
}

export interface ModeSummary {
  reliability_score: number;
  instabilities: number;
  regenerations: number;
  avg_entropy: number | null;
  max_entropy: number | null;
  min_entropy: number | null;
  confidence_breakdown?: Record<string, number>;
  trace_available?: boolean;
  note?: string;
}

export interface CompareSummary {
  delta_reliability: number;
  instabilities_reduced_by: number;
  regeneration_gain: number;
}

export interface GenerateSummary {
  plain: Partial<ModeSummary>;
  adaptive: Partial<ModeSummary>;
  compare: Partial<CompareSummary>;
}

export interface Capabilities {
  entropy_available: boolean;
  control_enabled: boolean;
}

export interface GenerateResponse {
  plain: ModeResponse | null;
  adaptive: ModeResponse | null;
  summary: GenerateSummary;
  latency_ms: number;
  model: string;
  device: string;
  capabilities: Capabilities;
}

export interface RecentRun {
  trace_id: string;
  timestamp: string;
  prompt: string;
  mode: string;
  reliability_score: number | null;
  regenerations: number;
  instabilities: number;
  summary_metrics?: GenerateSummary;
}

export interface RecentRunsResponse {
  runs: RecentRun[];
}
