"""Sampling helpers for token-by-token generation."""

from __future__ import annotations

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")
    if temperature == 1.0:
        return logits
    return logits / temperature


def apply_top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1].")
    if top_p == 1.0:
        return logits

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = torch.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p
    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    mask = torch.zeros_like(sorted_mask, dtype=torch.bool)
    mask.scatter_(dim=-1, index=sorted_indices, src=sorted_mask)
    return logits.masked_fill(mask, torch.finfo(logits.dtype).min)


def select_next_token(
    logits: torch.Tensor,
    *,
    temperature: float = 1.0,
    top_p: float = 1.0,
    do_sample: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the selected token id(s) and the associated probability distribution."""

    adjusted_logits = apply_temperature(logits, temperature)
    adjusted_logits = apply_top_p_filter(adjusted_logits, top_p)
    probabilities = torch.softmax(adjusted_logits, dim=-1)

    if do_sample:
        token_ids = torch.multinomial(probabilities, num_samples=1).squeeze(-1)
    else:
        token_ids = torch.argmax(probabilities, dim=-1)

    return token_ids, probabilities

