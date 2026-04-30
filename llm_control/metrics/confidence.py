"""Confidence scoring from a generation step trace."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReliabilitySummary:
    reliability_score: float
    avg_entropy: float
    instability_count: int
    regeneration_count: int
    stable: bool
    classification: str
    summary: str
    confidence_breakdown: dict[str, float]


def compute_reliability_score(step_trace, regeneration_count: int = 0) -> ReliabilitySummary:
    """Derive a single reliability score from a generation trace.

    Combines average entropy, instability hits, and regeneration use.
    """

    entropies = [s.entropy for s in step_trace if s.entropy is not None]
    instabilities = [s.instability for s in step_trace if getattr(s, "instability", None)]

    avg_entropy = sum(entropies) / len(entropies) if entropies else 0.0
    instability_count = len(instabilities)

    # Normalize signals
    entropy_score = min(avg_entropy / 5.0, 1.0)
    stability_score = 1 - min(instability_count / 10.0, 1.0)
    regen_penalty = 1 - min(regeneration_count / 2.0, 1.0)

    reliability_score = (
        0.5 * entropy_score +
        0.3 * stability_score +
        0.2 * regen_penalty
    )

    confidence_breakdown = {
        "entropy_score": round(entropy_score, 2),
        "stability_score": round(stability_score, 2),
        "regen_penalty": round(regen_penalty, 2),
    }

    stable = instability_count == 0 and regeneration_count == 0 and avg_entropy > 0.0
    if reliability_score >= 0.7:
        classification = "stable generation"
    elif reliability_score >= 0.5:
        classification = "moderate instability"
    else:
        classification = "unreliable output"

    summary = (
        f"reliability_score={reliability_score:.2f}, avg_entropy={avg_entropy:.2f}, "
        f"instabilities={instability_count}, regenerations={regeneration_count}, "
        f"classification={classification}"
    )

    return ReliabilitySummary(
        reliability_score=reliability_score,
        avg_entropy=avg_entropy,
        instability_count=instability_count,
        regeneration_count=regeneration_count,
        stable=stable,
        classification=classification,
        summary=summary,
        confidence_breakdown=confidence_breakdown,
    )


def max_probability_confidence(
    probabilities, *, dim: int = -1
):
    """Per-token confidence as max probability (kept for backward compat)."""
    return probabilities.max(dim=dim).values
