"""Remote generation via HuggingFace Inference API.

Mirrors the interface of base_generator and adaptive_generator but delegates
inference to the HF API, then runs the *same* stability-detection and
control-decision logic locally.
"""

from __future__ import annotations

from llm_control.generation.base_generator import GenerationResult, TokenStep
from llm_control.metrics.stability import detect_instability
from llm_control.control.controller import decide_control
from llm_control.model.remote_client import (
    RemoteModelClient,
    RemoteGenerationOutput,
    entropy_from_top_logprobs,
)


# ---------------------------------------------------------------------------
# Internal: convert an API response into a GenerationResult
# ---------------------------------------------------------------------------

def _api_output_to_result(
    prompt: str,
    output: RemoteGenerationOutput,
    *,
    regeneration_count: int = 0,
) -> GenerationResult:
    """Map HF API response tokens into the existing TokenStep/GenerationResult."""

    steps: list[TokenStep] = []
    token_ids: list[int] = []
    entropy_history: list[float] = []
    token_id_history: list[int] = []

    for idx, tok in enumerate(output.tokens):
        entropy = entropy_from_top_logprobs(tok.top_tokens)
        token_prob = min(1.0, max(0.0, __import__("math").exp(tok.logprob))) if tok.logprob else 0.0

        entropy_history.append(entropy)
        token_id_history.append(tok.token_id)

        instability = detect_instability(entropy_history, token_id_history)

        steps.append(
            TokenStep(
                index=idx,
                token_id=tok.token_id,
                token_text=tok.text,
                token_probability=token_prob,
                entropy=entropy,
                instability=instability,
                action="continue",
            )
        )
        token_ids.append(tok.token_id)

    return GenerationResult(
        prompt=prompt,
        generated_token_ids=token_ids,
        generated_text=output.generated_text,
        full_text=prompt + output.generated_text,
        steps=steps,
        regeneration_count=regeneration_count,
    )


# ---------------------------------------------------------------------------
# Public: plain generation
# ---------------------------------------------------------------------------

def generate_remote_plain(
    client: RemoteModelClient,
    prompt: str,
    max_tokens: int = 40,
) -> GenerationResult:
    """Run uncontrolled generation via the remote API."""

    output = client.generate(prompt, max_new_tokens=max_tokens, temperature=1.0)
    return _api_output_to_result(prompt, output)


# ---------------------------------------------------------------------------
# Public: adaptive generation (with control logic)
# ---------------------------------------------------------------------------

def generate_remote_adaptive(
    client: RemoteModelClient,
    prompt: str,
    max_tokens: int = 40,
) -> GenerationResult:
    """Simulate adaptive control by analysing a first pass, then regenerating if unstable.

    Mirrors the local adaptive generator's strategy:
      1. Generate with default temperature.
      2. Run stability analysis on the returned tokens.
      3. If instability is detected (repetition / collapse / lock),
         regenerate with lower temperature (0.7) — exactly like the
         local controller does.
    """

    # --- first pass (temperature 1.0) ---
    output_1 = client.generate(prompt, max_new_tokens=max_tokens, temperature=1.0)
    result_1 = _api_output_to_result(prompt, output_1)

    # Analyse the first pass through the control pipeline
    has_instability = False
    for step in result_1.steps:
        if step.instability is not None:
            decision = decide_control(
                step.instability, step.index, step.entropy, 1.0,
            )
            if decision.action in ("regenerate", "lower_temperature", "stop"):
                has_instability = True
                # Tag the step with the control action for the UI
                step.action = decision.action
            break  # react on first instability (mirrors local behaviour)

    if not has_instability:
        # First pass was stable — return as-is with action annotations
        return result_1

    # --- second pass (temperature 0.7 — same as local regeneration) ---
    output_2 = client.generate(
        prompt,
        max_new_tokens=max_tokens,
        temperature=0.7,
        repetition_penalty=1.2,
    )
    result_2 = _api_output_to_result(prompt, output_2, regeneration_count=1)

    # Merge step traces: keep the first-pass steps as "pre-regen" context,
    # then append second-pass steps.  This mirrors how the local adaptive
    # generator records all steps including pre-regeneration ones.
    combined_steps = list(result_1.steps) + list(result_2.steps)

    return GenerationResult(
        prompt=prompt,
        generated_token_ids=result_2.generated_token_ids,
        generated_text=result_2.generated_text,
        full_text=prompt + result_2.generated_text,
        steps=combined_steps,
        regeneration_count=1,
    )
