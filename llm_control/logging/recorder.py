"""Run recording scaffold."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm_control.logging.trace import TraceEvent


@dataclass
class RunRecord:
    prompt: str
    events: list[TraceEvent] = field(default_factory=list)
