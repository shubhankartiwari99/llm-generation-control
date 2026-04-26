from __future__ import annotations

from llm_control.evaluation.analysis import compare_runs
from llm_control.generation.base_generator import GenerationResult, TokenStep
from llm_control.metrics.confidence import compute_confidence


def test_compute_confidence_summarizes_trace() -> None:
    steps = [
        TokenStep(index=0, token_id=1, token_text="a", token_probability=0.7, entropy=2.0),
        TokenStep(index=1, token_id=2, token_text="b", token_probability=0.6, entropy=4.0),
        TokenStep(
            index=2,
            token_id=3,
            token_text="c",
            token_probability=0.5,
            entropy=6.0,
            instability="repetition_loop",
        ),
    ]

    summary = compute_confidence(steps, regeneration_count=1)

    assert summary.avg_entropy == 4.0
    assert summary.instability_count == 1
    assert summary.regeneration_count == 1
    assert summary.stable is False
    assert "confidence=" in summary.summary


def test_compare_runs_stable_vs_unstable() -> None:
    """Comparison should show adaptive improvement over plain."""
    plain_steps = [
        TokenStep(index=0, token_id=1, token_text="a", token_probability=0.9, entropy=2.5),
        TokenStep(
            index=1,
            token_id=2,
            token_text="b",
            token_probability=0.8,
            entropy=0.2,
            instability="entropy_collapse",
        ),
        TokenStep(
            index=2,
            token_id=3,
            token_text="c",
            token_probability=0.7,
            entropy=0.1,
            instability="low_entropy_lock",
        ),
    ]
    adaptive_steps = [
        TokenStep(index=0, token_id=1, token_text="a", token_probability=0.9, entropy=2.5),
        TokenStep(index=1, token_id=2, token_text="b", token_probability=0.8, entropy=2.8),
        TokenStep(index=2, token_id=3, token_text="c", token_probability=0.7, entropy=2.7),
    ]

    plain_result = GenerationResult(
        prompt="test",
        generated_token_ids=[1, 2, 3],
        generated_text="abc",
        full_text="test abc",
        steps=plain_steps,
        regeneration_count=0,
    )
    adaptive_result = GenerationResult(
        prompt="test",
        generated_token_ids=[1, 2, 3],
        generated_text="abc",
        full_text="test abc",
        steps=adaptive_steps,
        regeneration_count=0,
    )

    comparison = compare_runs(plain_result, adaptive_result)

    assert "plain" in comparison
    assert "adaptive" in comparison
    assert "delta_confidence" in comparison
    assert comparison["delta_confidence"] > 0  # adaptive should be better
    assert comparison["delta_instability"] < 0  # fewer instabilities
