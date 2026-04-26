from __future__ import annotations

from llm_control.metrics.stability import (
    ENTROPY_COLLAPSE,
    LOW_ENTROPY_LOCK,
    REPETITION_LOOP,
    detect_instability,
)


def test_detects_entropy_collapse() -> None:
    history = [3.0, 0.4]
    tokens = [10, 11]

    assert detect_instability(history, tokens) == ENTROPY_COLLAPSE


def test_detects_repetition_loop() -> None:
    history = [1.2, 1.1, 1.0]
    tokens = [5, 5, 5]

    assert detect_instability(history, tokens) == REPETITION_LOOP


def test_detects_low_entropy_lock() -> None:
    history = [0.4, 0.3, 0.2, 0.1, 0.45]
    tokens = [1, 2, 3, 4, 5]

    assert detect_instability(history, tokens) == LOW_ENTROPY_LOCK


def test_returns_none_for_stable_history() -> None:
    history = [1.5, 1.8, 1.3, 1.0, 0.6]
    tokens = [1, 2, 3, 4, 5]

    assert detect_instability(history, tokens) is None
