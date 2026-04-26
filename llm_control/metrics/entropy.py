"""Entropy helpers for logits and probability distributions."""

from __future__ import annotations

import math

import torch


def entropy_from_probs(
    probabilities: torch.Tensor,
    *,
    dim: int = -1,
    eps: float = 1e-12,
) -> torch.Tensor:
    safe_probabilities = probabilities.clamp_min(eps)
    return -(safe_probabilities * safe_probabilities.log()).sum(dim=dim)


def entropy_from_logits(logits: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=dim)
    return entropy_from_probs(probabilities, dim=dim)


def compute_entropy(logits: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    """Compatibility alias for the per-step entropy used inside the generator loop."""

    return entropy_from_logits(logits, dim=dim)


def normalized_entropy_from_logits(logits: torch.Tensor, *, dim: int = -1) -> torch.Tensor:
    entropy = entropy_from_logits(logits, dim=dim)
    vocab_size = logits.size(dim)
    if vocab_size <= 1:
        return torch.zeros_like(entropy)
    return entropy / math.log(vocab_size)
