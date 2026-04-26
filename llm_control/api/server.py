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
from llm_control.logging.storage import RunStorage


# Global state for model
state = {}
storage = RunStorage()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    model, tokenizer = load_mistral_7b()
    state["model"] = model
    state["tokenizer"] = tokenizer
    
    print("Warming up model...")
    import torch
    try:
        model(torch.ones(1, 1, dtype=torch.long).to("mps"))
    except Exception as e:
        print(f"Warmup warning: {e}")
        
    print("Model loaded and warmed up successfully.")
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
    action: Optional[str] = "continue"


class ModeResponse(BaseModel):
    text: str
    steps: list[TokenStepResponse]
    confidence: float
    regenerations: int = 0

class GenerateResponse(BaseModel):
    plain: ModeResponse
    adaptive: ModeResponse
    latency_ms: int
    model: str = "mistral-7b"
    device: str = "mps"


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    from fastapi import HTTPException
    import time
    start_time = time.time()

    model = state["model"]
    tokenizer = state["tokenizer"]

    try:
        # Run plain first
        plain_res = generate_stepwise(model, tokenizer, req.prompt, max_tokens=req.max_tokens, stop_at_eos=True)
        plain_conf = compute_confidence(plain_res.steps, regeneration_count=0)
        plain_steps = [{"token": s.token_text, "entropy": s.entropy, "instability": s.instability, "action": s.action} for s in plain_res.steps]

        # Run adaptive second (sequential to save memory)
        adapt_res = generate_adaptive(model, tokenizer, req.prompt, max_tokens=req.max_tokens, verbose=False)
        adapt_conf = compute_confidence(adapt_res.steps, regeneration_count=adapt_res.regeneration_count)
        adapt_steps = [{"token": s.token_text, "entropy": s.entropy, "instability": s.instability, "action": s.action} for s in adapt_res.steps]

    except RuntimeError as e:
        if "MPS out of memory" in str(e):
            raise HTTPException(status_code=500, detail="Model too large for current memory")
        raise e

    latency_ms = int((time.time() - start_time) * 1000)

    plain_obj = ModeResponse(
        text=plain_res.generated_text,
        steps=plain_steps,
        confidence=plain_conf.confidence,
        regenerations=0
    )
    
    adapt_obj = ModeResponse(
        text=adapt_res.generated_text,
        steps=adapt_steps,
        confidence=adapt_conf.confidence,
        regenerations=getattr(adapt_res, "regeneration_count", 0)
    )

    response_data = {
        "plain": plain_obj.dict(),
        "adaptive": adapt_obj.dict(),
        "latency_ms": latency_ms,
        "model": "mistral-7b",
        "device": "mps"
    }

    storage.log_run(prompt=req.prompt, mode="compare", response_data=response_data)

    return GenerateResponse(
        plain=plain_obj,
        adaptive=adapt_obj,
        latency_ms=latency_ms,
        model="mistral-7b",
        device="mps",
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model": "mistral-7b" if "model" in state else None,
        "device": "mps" if "model" in state else None,
    }
