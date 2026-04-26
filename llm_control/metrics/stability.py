"""Stability detectors for generation traces.

Each detector inspects rolling windows of entropy and token histories
to flag degenerate model behavior before it compounds.
"""

from __future__ import annotations

from typing import Optional, Sequence


# ── Instability labels ──────────────────────────────────────────────
ENTROPY_COLLAPSE = "entropy_collapse"
REPETITION_LOOP = "repetition_loop"
LOW_ENTROPY_LOCK = "low_entropy_lock"


def detect_instability(
    entropy_history: Sequence[float],
    token_history: Sequence[int],
    *,
    collapse_low: float = 0.5,
    collapse_high: float = 2.0,
    repetition_window: int = 3,
    low_entropy_threshold: float = 0.5,
    low_entropy_window: int = 5,
) -> Optional[str]:
    """Return the first instability signal found, or ``None`` if stable.

    Checks are ordered by severity (most urgent first):

    1. **entropy_collapse** – entropy dropped from above *collapse_high*
       to below *collapse_low* within the last two steps.
    2. **repetition_loop** – the same token was emitted for the last
       *repetition_window* consecutive steps.
    3. **low_entropy_lock** – entropy stayed below *low_entropy_threshold*
       for the last *low_entropy_window* steps.
    """

    # 1. entropy collapse: sharp drop between consecutive steps
    if len(entropy_history) >= 2:
        prev, curr = entropy_history[-2], entropy_history[-1]
        if curr < collapse_low and prev > collapse_high:
            return ENTROPY_COLLAPSE

    # 2. repetition loop: same token N times in a row
    if len(token_history) >= repetition_window:
        tail = token_history[-repetition_window:]
        if len(set(tail)) == 1:
            return REPETITION_LOOP

    # 3. sustained low entropy
    if len(entropy_history) >= low_entropy_window:
        tail = entropy_history[-low_entropy_window:]
        if all(e < low_entropy_threshold for e in tail):
            return LOW_ENTROPY_LOCK

    return None
