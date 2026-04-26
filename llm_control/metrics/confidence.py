"""Confidence scoring from a generation step trace."""

from __future__ import annotations

from dataclasses import dataclass
import torch


@dataclass
class ConfidenceSummary:
    confidence: float
    avg_entropy: float
    instability_count: int
    regeneration_count: int
    stable: bool
    summary: str


def compute_confidence(step_trace, regeneration_count: int = 0) -> ConfidenceSummary:
    """Derive a single confidence score from a generation trace.

    Combines average entropy, instability hits, and regeneration use.
    """

    entropies = [s.entropy for s in step_trace if s.entropy is not None]
    instabilities = [s.instability for s in step_trace if getattr(s, "instability", None)]

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    instability_count = len(instabilities)

    # Simple normalization: lower entropy and fewer events mean higher confidence.
    entropy_penalty = min(avg_entropy / 8.0, 1.0)
    instability_penalty = min(instability_count / 5.0, 1.0)
    regen_penalty = min(regeneration_count / 3.0, 1.0)

    confidence = 1.0 - (
        0.5 * entropy_penalty
        + 0.35 * instability_penalty
        + 0.15 * regen_penalty
    )
    confidence = max(0.0, min(1.0, confidence))

    stable = instability_count == 0 and regeneration_count == 0 and avg_entropy > 0.0

    summary = (
        f"confidence={confidence:.2f}, avg_entropy={avg_entropy:.2f}, "
        f"instabilities={instability_count}, regenerations={regeneration_count}"
    )

    return ConfidenceSummary(
        confidence=confidence,
        avg_entropy=avg_entropy,
        instability_count=instability_count,
        regeneration_count=regeneration_count,
        stable=stable,
        summary=summary,
    )


def max_probability_confidence(
    probabilities: torch.Tensor, *, dim: int = -1
) -> torch.Tensor:
    """Per-token confidence as max probability (kept for backward compat)."""
    return probabilities.max(dim=dim).values
