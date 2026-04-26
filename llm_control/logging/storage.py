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
        summary_data = {
            "trace_id": trace_id,
            "timestamp": timestamp,
            "prompt": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            "mode": mode,
            "confidence": response_data.get("confidence"),
            "regenerations": response_data.get("regenerations"),
            "instabilities": sum(1 for step in response_data.get("steps", []) if step.get("instability"))
        }
        with open(self.runs_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(summary_data) + "\n")
            
        return trace_id
