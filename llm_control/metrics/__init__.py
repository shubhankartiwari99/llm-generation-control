"""Metrics derived from token distributions.

Heavy (torch-dependent) metrics are lazy-loaded so that
importing confidence or stability alone does not pull in torch.
"""

from llm_control.metrics.confidence import ConfidenceSummary, compute_confidence, max_probability_confidence


def __getattr__(name: str):
    """Lazy-load torch-dependent entropy functions."""

    _entropy_names = {
        "compute_entropy",
        "entropy_from_logits",
        "entropy_from_probs",
        "normalized_entropy_from_logits",
    }
    if name in _entropy_names:
        from llm_control.metrics import entropy as _ent
        return getattr(_ent, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ConfidenceSummary",
    "compute_confidence",
    "compute_entropy",
    "entropy_from_logits",
    "entropy_from_probs",
    "max_probability_confidence",
    "normalized_entropy_from_logits",
]
