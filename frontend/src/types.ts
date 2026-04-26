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

export interface GenerateResponse {
  plain: ModeResponse;
  adaptive: ModeResponse;
  latency_ms: number;
  model: string;
  device: string;
}
