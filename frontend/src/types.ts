export interface TokenStep {
  token: string;
  entropy: number;
  instability: string | null;
}

export interface GenerateResponse {
  output: string;
  steps: TokenStep[];
  confidence: number;
  regenerations: number;
}
