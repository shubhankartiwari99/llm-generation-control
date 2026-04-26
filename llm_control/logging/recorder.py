"""Run recording and persistence for generation experiments."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_control.generation.base_generator import GenerationResult
from llm_control.metrics.confidence import compute_confidence

LOGS_ROOT = Path("logs")
RUNS_FILE = LOGS_ROOT / "runs.json"
TRACES_DIR = LOGS_ROOT / "traces"


def ensure_log_dirs() -> None:
    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    TRACES_DIR.mkdir(parents=True, exist_ok=True)


def step_to_dict(step: Any) -> dict[str, Any]:
    return {
        "index": step.index,
        "token": step.token_text,
        "token_id": step.token_id,
        "token_probability": step.token_probability,
        "entropy": step.entropy,
        "instability": step.instability,
    }


def record_run(result: GenerationResult, mode: str, max_tokens: int) -> Path:
    ensure_log_dirs()
    confidence = compute_confidence(result.steps, regeneration_count=result.regeneration_count)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    trace_file = TRACES_DIR / f"{timestamp}_{mode}.json"

    payload = {
        "prompt": result.prompt,
        "mode": mode,
        "max_tokens": max_tokens,
        "generated_text": result.generated_text,
        "full_text": result.full_text,
        "regeneration_count": result.regeneration_count,
        "confidence": confidence.confidence,
        "classification": confidence.classification,
        "avg_entropy": confidence.avg_entropy,
        "instability_count": confidence.instability_count,
        "steps": [step_to_dict(step) for step in result.steps],
    }

    trace_file.write_text(json.dumps(payload, indent=2))

    runs = []
    if RUNS_FILE.exists():
        runs = json.loads(RUNS_FILE.read_text())

    runs.append(
        {
            "timestamp": timestamp,
            "prompt": result.prompt,
            "mode": mode,
            "max_tokens": max_tokens,
            "regeneration_count": result.regeneration_count,
            "confidence": confidence.confidence,
            "classification": confidence.classification,
            "instability_count": confidence.instability_count,
            "avg_entropy": confidence.avg_entropy,
            "trace_file": str(trace_file),
        }
    )
    RUNS_FILE.write_text(json.dumps(runs, indent=2))

    return trace_file
