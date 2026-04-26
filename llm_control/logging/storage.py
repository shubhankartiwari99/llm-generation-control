"""Storage and persistence layer for generation traces."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


class RunStorage:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.traces_dir = self.log_dir / "traces"
        self.runs_file = self.log_dir / "runs.jsonl"
        
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.runs_file.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, prompt: str, mode: str, response_data: Dict[str, Any]) -> str:
        """Log a complete run to both the summary file and detailed trace file."""
        timestamp = datetime.utcnow().isoformat().replace(":", "-").replace(".", "-")
        trace_id = f"run_{timestamp}_{mode}"
        
        # 1. Save detailed trace
        trace_path = self.traces_dir / f"{trace_id}.json"
        trace_data = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "prompt": prompt,
            "mode": mode,
            **response_data
        }
        with open(trace_path, "w", encoding="utf-8") as f:
            json.dump(trace_data, f, indent=2)
            
        # 2. Append to runs summary log
        mode_data = response_data.get(mode)
        if mode == "compare":
            mode_data = response_data.get("adaptive") or response_data.get("plain")

        run_summary = response_data.get("summary", {})
        confidence = mode_data.get("confidence") if mode_data else None
        regenerations = mode_data.get("regenerations") if mode_data else 0

        instabilities = 0
        if mode == "compare":
            instabilities = run_summary.get("adaptive", {}).get("instabilities", 0)
        elif mode_data:
            instabilities = sum(1 for step in mode_data.get("steps", []) if step.get("instability"))

        summary_data = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "mode": mode,
            "confidence": confidence,
            "regenerations": regenerations,
            "instabilities": instabilities,
            "summary_metrics": run_summary
        }
        with open(self.runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_data) + "\n")
            
        return trace_id
