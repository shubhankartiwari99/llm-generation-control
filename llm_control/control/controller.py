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


def decide_action(instability: Optional[str], step: int, has_regenerated: bool) -> Action:
    """Map an instability signal to a concrete control action."""

    if instability == "entropy_collapse":
        if step < 5:
            return "lower_temperature"
        return "continue"

    if instability == "repetition_loop":
        if not has_regenerated:
            return "regenerate"
        return "stop"

    if instability == "high_uncertainty":
        return "lower_temperature"

    return "continue"
