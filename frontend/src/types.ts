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
}

export interface ModeSummary {
  confidence: number;
  instabilities: number;
  regenerations: number;
  avg_entropy: number;
  max_entropy: number;
  min_entropy: number;
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
