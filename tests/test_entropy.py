from __future__ import annotations

import math

import torch

from llm_control.metrics.entropy import entropy_from_logits, entropy_from_probs, normalized_entropy_from_logits


def test_entropy_matches_uniform_distribution() -> None:
    probabilities = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
    entropy = entropy_from_probs(probabilities)
    assert torch.allclose(entropy, torch.tensor([math.log(4.0)]))


def test_entropy_from_logits_is_consistent_with_probabilities() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    probs = torch.softmax(logits, dim=-1)
    assert torch.allclose(entropy_from_logits(logits), entropy_from_probs(probs))


def test_normalized_entropy_is_bounded() -> None:
    logits = torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32)
    normalized = normalized_entropy_from_logits(logits)
    assert torch.all(normalized >= 0)
    assert torch.all(normalized <= 1)

