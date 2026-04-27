"""Shared data types for generation results.

These dataclasses are intentionally free of heavy dependencies (no torch,
no transformers) so they can be imported by both local and remote
generation paths without pulling in large libraries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenStep:
    """Per-token observability record emitted by the generation loop."""

    index: int
    token_id: int
    token_text: str
    token_probability: float
    entropy: float
    instability: Optional[str] = None
    action: Optional[str] = "continue"


@dataclass
class GenerationResult:
    """Structured output of a single token-by-token generation run."""

    prompt: str
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    full_text: str = ""
    steps: list[TokenStep] = field(default_factory=list)
    regeneration_count: int = 0

    @property
    def entropy_trace(self) -> list[float]:
        return [step.entropy for step in self.steps]

    @property
    def instability_trace(self) -> list[str]:
        return [step.instability for step in self.steps if step.instability is not None]
