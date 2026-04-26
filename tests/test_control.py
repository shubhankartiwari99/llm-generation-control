from __future__ import annotations

from llm_control.control.controller import decide_control
from llm_control.metrics.stability import (
    ENTROPY_COLLAPSE,
    LOW_ENTROPY_LOCK,
    REPETITION_LOOP,
    detect_instability,
)


def test_detect_entropy_collapse() -> None:
    assert detect_instability([2.8, 0.3], [11, 12]) == ENTROPY_COLLAPSE


def test_detect_repetition_loop() -> None:
    assert detect_instability([1.2, 1.0, 0.9], [42, 42, 42]) == REPETITION_LOOP


def test_detect_low_entropy_lock() -> None:
    assert detect_instability([0.4, 0.3, 0.2, 0.1, 0.2], [1, 2, 3, 4, 5]) == LOW_ENTROPY_LOCK


def test_stable_trace_returns_none() -> None:
    assert detect_instability([1.2, 1.4, 1.1], [1, 2, 3]) is None


def test_controller_regenerates_on_repetition_loop() -> None:
    decision = decide_control(REPETITION_LOOP, step=3, entropy=1.0, temperature=1.0)

    assert decision.action == "regenerate"
    assert decision.reason == REPETITION_LOOP


def test_controller_lowers_temperature_on_early_collapse() -> None:
    decision = decide_control(ENTROPY_COLLAPSE, step=2, entropy=0.3, temperature=1.0)

    assert decision.action == "lower_temperature"
    assert decision.new_temperature == 0.8
    assert decision.reason == "early_entropy_collapse"


def test_controller_stops_on_low_entropy_lock() -> None:
    decision = decide_control(LOW_ENTROPY_LOCK, step=8, entropy=0.2, temperature=0.7)

    assert decision.action == "stop"
    assert decision.reason == LOW_ENTROPY_LOCK
