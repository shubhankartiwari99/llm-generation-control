"""Central control policy: observe instability, then decide action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

Action = Literal["continue", "lower_temperature", "regenerate", "stop"]


@dataclass
class ControlDecision:
    action: Action
    new_temperature: Optional[float] = None
    reason: Optional[str] = None


def decide_control(
    instability: Optional[str],
    step: int,
    entropy: float,
    temperature: float,
) -> ControlDecision:
    """Map an instability signal to a concrete control action.

    Priority order (most urgent first):
    1. repetition_loop   -> regenerate (one-shot)
    2. entropy_collapse  -> lower temperature (early steps only)
    3. low_entropy_lock  -> stop generation
    """

    # 1. repetition -> regenerate
    if instability == "repetition_loop":
        return ControlDecision(
            action="regenerate",
            reason="repetition_loop",
        )

    # 2. early collapse -> lower temperature
    if instability == "entropy_collapse" and step < 5:
        return ControlDecision(
            action="lower_temperature",
            new_temperature=max(0.2, temperature - 0.2),
            reason="early_entropy_collapse",
        )

    # 3. locked -> stop
    if instability == "low_entropy_lock":
        return ControlDecision(
            action="stop",
            reason="low_entropy_lock",
        )

    return ControlDecision(action="continue")
