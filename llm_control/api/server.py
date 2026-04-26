"""FastAPI backend for LLM generation control."""

from __future__ import annotations

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_control.model.loader import load_mistral_7b
from llm_control.generation.base_generator import generate_stepwise
from llm_control.generation.adaptive_generator import generate_adaptive
from llm_control.metrics.confidence import compute_confidence


# Global state for model
state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model, tokenizer = load_mistral_7b()
    state["model"] = model
    state["tokenizer"] = tokenizer
    print("Model loaded successfully.")
    yield
    state.clear()


app = FastAPI(title="LLM Control API", lifespan=lifespan)

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 50
    mode: str = "adaptive"  # "plain" or "adaptive"


class TokenStepResponse(BaseModel):
    token: str
    entropy: float
    instability: Optional[str] = None


class GenerateResponse(BaseModel):
    output: str
    steps: list[TokenStepResponse]
    confidence: float
    regenerations: int


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    model = state["model"]
    tokenizer = state["tokenizer"]

    if req.mode == "plain":
        result = generate_stepwise(model, tokenizer, req.prompt, max_tokens=req.max_tokens, stop_at_eos=True)
        # Use existing compute_confidence
        conf_summary = compute_confidence(result.steps, regeneration_count=0)
    else:
        result = generate_adaptive(model, tokenizer, req.prompt, max_tokens=req.max_tokens, verbose=False)
        conf_summary = compute_confidence(result.steps, regeneration_count=result.regeneration_count)

    steps_res = [
        TokenStepResponse(
            token=s.token_text,
            entropy=s.entropy,
            instability=s.instability,
        )
        for s in result.steps
    ]

    return GenerateResponse(
        output=result.generated_text,
        steps=steps_res,
        confidence=conf_summary.confidence,
        regenerations=getattr(result, "regeneration_count", 0),
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": "model" in state}
