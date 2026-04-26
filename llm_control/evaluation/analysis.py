"""Analysis helpers for traces and experiment outputs."""

from __future__ import annotations

from llm_control.metrics.confidence import compute_confidence


def compare_runs(plain_result, adaptive_result):
    plain_conf = compute_confidence(plain_result.steps, regeneration_count=0)
    adaptive_conf = compute_confidence(
        adaptive_result.steps,
        regeneration_count=adaptive_result.regeneration_count,
    )

    return {
        "plain": plain_conf,
        "adaptive": adaptive_conf,
        "delta_confidence": adaptive_conf.confidence - plain_conf.confidence,
        "delta_instability": adaptive_conf.instability_count - plain_conf.instability_count,
    }


def run_comparative_experiment(
    prompts: list[str],
    plain_generator,
    adaptive_generator,
    max_tokens: int = 20,
) -> None:
    """Run plain vs adaptive generation on a set of prompts and print a summary table.

    Args:
        prompts: List of prompts to test.
        plain_generator: Callable that generates with plain strategy.
        adaptive_generator: Callable that generates with adaptive strategy.
        max_tokens: Maximum tokens to generate per prompt.
    """

    print("\n" + "=" * 80)
    print("COMPARATIVE GENERATION EXPERIMENT")
    print("=" * 80)

    for prompt in prompts:
        print(f"\nPrompt: {prompt}")

        plain_result = plain_generator(prompt, max_tokens=max_tokens)
        adaptive_result = adaptive_generator(prompt, max_tokens=max_tokens)

        comparison = compare_runs(plain_result, adaptive_result)
        plain_conf = comparison["plain"]
        adaptive_conf = comparison["adaptive"]
        delta_conf = comparison["delta_confidence"]

        print(
            f"Plain:    confidence={plain_conf.confidence:.2f}, "
            f"instabilities={plain_conf.instability_count}, "
            f"regenerations={plain_conf.regeneration_count}"
        )
        print(
            f"Adaptive: confidence={adaptive_conf.confidence:.2f}, "
            f"instabilities={adaptive_conf.instability_count}, "
            f"regenerations={adaptive_conf.regeneration_count}"
        )
        print(f"Delta: {delta_conf:+.2f}")
