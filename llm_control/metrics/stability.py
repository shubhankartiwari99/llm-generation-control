"""Stability detectors for generation traces.

Each detector inspects rolling windows of entropy and token histories
to flag degenerate model behavior before it compounds.
"""

from __future__ import annotations

from typing import Optional, Sequence


LOW_ENTROPY_THRESHOLD = 1.0
HIGH_ENTROPY_THRESHOLD = 4.5
REPETITION_WINDOW = 3

def detect_instability(entropy: Optional[float], recent_tokens: Sequence[int]) -> Optional[str]:
    """Return the first instability signal found, or ``None`` if stable."""

    if entropy is None:
        return None

    def is_repeating(tokens: Sequence[int], window: int) -> bool:
        if len(tokens) < window:
            return False
        return len(set(tokens[-window:])) == 1

    if is_repeating(recent_tokens, window=REPETITION_WINDOW):
        return "repetition_loop"

    if entropy < LOW_ENTROPY_THRESHOLD:
        return "entropy_collapse"

    if entropy > HIGH_ENTROPY_THRESHOLD:
        return "high_uncertainty"

    return None
