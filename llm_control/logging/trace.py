"""Token-level trace structures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TraceEvent:
    token: str
    probability: float
    entropy: float
    decision: str | None = None
