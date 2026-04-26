"""Batch experiment entry points."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from llm_control.evaluation.analysis import compare_runs
from llm_control.generation.adaptive_generator import generate_adaptive
from llm_control.generation.base_generator import generate_stepwise

DEFAULT_PROMPTS = (
    "Explain discipline for long-term success.",
    "Write only blank lines",
    "Summarize the benefits of consistency.",
)


def run_comparison_experiment(
    model,
    tokenizer,
    prompts: Sequence[str] = DEFAULT_PROMPTS,
    *,
    max_tokens: int = 20,
    seed: int | None = None,
):
    results = []

    for prompt in prompts:
        prompt_seed = seed

        if prompt_seed is not None:
            torch.manual_seed(prompt_seed)
        plain_result = generate_stepwise(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            stop_at_eos=False,
        )

        if prompt_seed is not None:
            torch.manual_seed(prompt_seed)
        adaptive_result = generate_adaptive(
            model,
            tokenizer,
            prompt,
            max_tokens=max_tokens,
            verbose=False,
        )

        results.append(
            {
                "prompt": prompt,
                "plain_result": plain_result,
                "adaptive_result": adaptive_result,
                "comparison": compare_runs(plain_result, adaptive_result),
            }
        )

    return results


def format_comparison_rows(results) -> list[str]:
    lines: list[str] = []

    for row in results:
        comparison = row["comparison"]
        plain = comparison["plain"]
        adaptive = comparison["adaptive"]
        delta = comparison["delta_confidence"]

        lines.append(f"Prompt: {row['prompt']}")
        lines.append(f"Plain: {plain.summary}")
        lines.append(f"Adaptive: {adaptive.summary}")
        lines.append(f"Delta: {delta:+.2f}")
        lines.append("")

    return lines
