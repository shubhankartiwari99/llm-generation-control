"""FastAPI backend for LLM generation control."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from collections import defaultdict, deque
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

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
rate_limit_window_s = 60
rate_limit_max_requests = 6
request_history: dict[str, deque[float]] = defaultdict(deque)

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
    prompt: str = Field(..., max_length=2000)
    max_tokens: int = Field(default=50, le=100)
    mode: Literal["compare", "plain", "adaptive"] = "compare"


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
    plain: Optional[ModeResponse] = None
    adaptive: Optional[ModeResponse] = None
    summary: dict
    latency_ms: int
    model: str = "mistral-7b"
    device: str = "mps"


class RecentRunsResponse(BaseModel):
    runs: list[dict]


def enforce_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    history = request_history[client_ip]

    while history and now - history[0] > rate_limit_window_s:
        history.popleft()

    if len(history) >= rate_limit_max_requests:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: max {rate_limit_max_requests} requests per {rate_limit_window_s}s",
        )

    history.append(now)


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest, request: Request):
    enforce_rate_limit(request)
    start_time = time.time()

    model = state["model"]
    tokenizer = state["tokenizer"]

    prompt_length = len(req.prompt.strip())
    if prompt_length == 0:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    if prompt_length > 2000:
        raise HTTPException(status_code=400, detail="Prompt too long (max 2000 chars)")
    if req.max_tokens > 100:
        raise HTTPException(status_code=400, detail="max_tokens too large (max 100)")
    
    plain_obj = None
    adapt_obj = None
    plain_conf = None
    adapt_conf = None
    plain_steps = []
    adapt_steps = []

    try:
        if req.mode in ["compare", "plain"]:
            plain_res = generate_stepwise(model, tokenizer, req.prompt, max_tokens=req.max_tokens, stop_at_eos=True)
            plain_conf = compute_confidence(plain_res.steps, regeneration_count=0)
            plain_steps = [{"token": s.token_text, "entropy": s.entropy, "instability": s.instability, "action": s.action} for s in plain_res.steps]
            plain_obj = ModeResponse(
                text=plain_res.generated_text,
                steps=plain_steps,
                confidence=plain_conf.confidence,
                regenerations=0
            )

        if req.mode in ["compare", "adaptive"]:
            adapt_res = generate_adaptive(model, tokenizer, req.prompt, max_tokens=req.max_tokens, verbose=False)
            adapt_conf = compute_confidence(adapt_res.steps, regeneration_count=adapt_res.regeneration_count)
            adapt_steps = [{"token": s.token_text, "entropy": s.entropy, "instability": s.instability, "action": s.action} for s in adapt_res.steps]
            adapt_obj = ModeResponse(
                text=adapt_res.generated_text,
                steps=adapt_steps,
                confidence=adapt_conf.confidence,
                regenerations=getattr(adapt_res, "regeneration_count", 0)
            )

    except RuntimeError as e:
        if "MPS out of memory" in str(e) or "out of memory" in str(e).lower():
            raise HTTPException(status_code=500, detail="Model too large for current memory")
        raise e

    latency_ms = int((time.time() - start_time) * 1000)
        
    def compute_mode_summary(steps: list[dict], confidence: float | None, regenerations: int = 0) -> dict:
        if confidence is None:
            return {}
        entropies = [s["entropy"] for s in steps] if steps else []
        return {
            "confidence": confidence,
            "instabilities": sum(1 for s in steps if s["instability"]),
            "regenerations": regenerations,
            "avg_entropy": (sum(entropies) / len(entropies)) if entropies else 0.0,
            "max_entropy": max(entropies) if entropies else 0.0,
            "min_entropy": min(entropies) if entropies else 0.0,
        }

    summary = {
        "plain": compute_mode_summary(plain_steps, plain_conf.confidence if plain_conf else None, 0),
        "adaptive": compute_mode_summary(
            adapt_steps,
            adapt_conf.confidence if adapt_conf else None,
            adapt_obj.regenerations if adapt_obj else 0,
        ),
        "compare": {},
    }
    if plain_conf and adapt_conf:
        summary["compare"] = {
            "delta_confidence": adapt_conf.confidence - plain_conf.confidence,
            "instabilities_reduced_by": summary["plain"].get("instabilities", 0) - summary["adaptive"].get("instabilities", 0),
            "regeneration_gain": summary["adaptive"].get("regenerations", 0) - summary["plain"].get("regenerations", 0),
        }

    response_data = {
        "plain": plain_obj.dict() if plain_obj else None,
        "adaptive": adapt_obj.dict() if adapt_obj else None,
        "latency_ms": latency_ms,
        "model": "mistral-7b",
        "device": "mps",
        "summary": summary
    }

    storage.log_run(prompt=req.prompt, mode=req.mode, response_data=response_data)

    return GenerateResponse(
        plain=plain_obj,
        adaptive=adapt_obj,
        summary=summary,
        latency_ms=latency_ms,
        model="mistral-7b",
        device="mps",
    )


@app.get("/runs/recent", response_model=RecentRunsResponse)
def recent_runs(limit: int = 10):
    safe_limit = max(1, min(limit, 50))
    runs = storage.get_recent_runs(limit=safe_limit)
    return RecentRunsResponse(runs=runs)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": "model" in state,
        "model": "mistral-7b" if "model" in state else None,
        "device": "mps" if "model" in state else None,
    }
