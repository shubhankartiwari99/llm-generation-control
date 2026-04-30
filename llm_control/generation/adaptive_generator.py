"""Adaptive manual decoding with closed-loop control actions."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizerBase, RepetitionPenaltyLogitsProcessor, TopPLogitsWarper

from llm_control.control.controller import decide_action
from llm_control.generation.base_generator import GenerationResult, TokenStep
from llm_control.metrics.entropy import compute_entropy
from llm_control.metrics.stability import detect_instability


def generate_adaptive(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    max_tokens: int = 20,
    *,
    verbose: bool = True,
) -> GenerationResult:
    """Generate tokens while reacting to detected instability signals."""

    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_input_ids = inputs["input_ids"].to(device)
    generated = prompt_input_ids.clone()

    entropy_history: list[float] = []
    token_history: list[int] = []
    all_steps: list[TokenStep] = []
    final_token_ids: list[int] = []

    temperature = 1.0
    regen_count = 0
    max_regens = 1
    total_steps = 0
    local_step = 0

    with torch.no_grad():
        while total_steps < max_tokens:
            outputs = model(input_ids=generated)
            logits = outputs.logits[:, -1, :]

            # Apply repetition penalty
            rep_processor = RepetitionPenaltyLogitsProcessor(penalty=1.2)
            logits = rep_processor(generated, logits)

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Apply top_p if regenerating or high uncertainty
            current_top_p = top_p if 'top_p' in locals() else (0.9 if regen_count > 0 else 1.0)
            if current_top_p < 1.0:
                top_p_warper = TopPLogitsWarper(top_p=current_top_p)
                logits = top_p_warper(generated, logits)

            probs = F.softmax(logits, dim=-1)
            entropy = float(compute_entropy(logits, dim=-1).item())
            next_token = torch.multinomial(probs, num_samples=1)
            token_id = int(next_token.item())

            entropy_history.append(entropy)
            token_history.append(token_id)

            instability = detect_instability(entropy, token_history)
            action = decide_action(instability, local_step, regen_count > 0)

            token_text = tokenizer.decode(
                [token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            step = TokenStep(
                index=local_step,
                token_id=token_id,
                token_text=token_text,
                token_probability=float(probs[0, token_id].item()),
                entropy=entropy,
                instability=instability,
                action=action,
                temperature=temperature,
            )
            all_steps.append(step)

            if verbose:
                token_display = token_text.encode("unicode_escape").decode("ascii")
                flag = f" | Instability: {instability}" if instability else ""
                action_info = (
                    f" | Action: {action}"
                    if action != "continue"
                    else ""
                )
                print(
                    f"Step {local_step:2d} | Token: {token_display:12s} | "
                    f"Entropy: {entropy:.3f} | Temp: {temperature:.2f}"
                    f"{flag}{action_info}"
                )

            total_steps += 1

            if action == "lower_temperature":
                temperature = max(0.4, temperature - 0.2)
                if instability == "high_uncertainty":
                    top_p = 0.85
                elif regen_count == 0:
                    top_p = 1.0

            elif action == "regenerate":
                if regen_count < max_regens:
                    if verbose:
                        print("Regenerating from prompt...")
                    generated = prompt_input_ids.clone()
                    entropy_history = []
                    token_history = []
                    final_token_ids = []
                    temperature = 0.7
                    top_p = 0.85 if instability == "repetition_loop" else 0.9
                    regen_count += 1
                    local_step = 0
                    continue

            elif action == "stop":
                if verbose:
                    print("Stopping generation")
                break

            generated = torch.cat([generated, next_token], dim=1)
            final_token_ids.append(token_id)
            local_step += 1

    generated_text = tokenizer.decode(
        final_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    full_text = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return GenerationResult(
        prompt=prompt,
        generated_token_ids=final_token_ids,
        generated_text=generated_text,
        full_text=full_text,
        steps=all_steps,
        regeneration_count=regen_count,
    )
