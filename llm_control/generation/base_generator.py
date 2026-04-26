"""Core token-level generation loop with metric hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from llm_control.generation.sampler import select_next_token
from llm_control.metrics.entropy import compute_entropy
from llm_control.metrics.stability import detect_instability


@dataclass
class TokenStep:
    """Per-token observability record emitted by the generation loop."""

    index: int
    token_id: int
    token_text: str
    token_probability: float
    entropy: float
    instability: Optional[str] = None


@dataclass
class GenerationResult:
    """Structured output of a single token-by-token generation run."""

    prompt: str
    generated_token_ids: list[int] = field(default_factory=list)
    generated_text: str = ""
    full_text: str = ""
    steps: list[TokenStep] = field(default_factory=list)
    regeneration_count: int = 0

    @property
    def entropy_trace(self) -> list[float]:
        return [step.entropy for step in self.steps]

    @property
    def instability_trace(self) -> list[str]:
        return [step.instability for step in self.steps if step.instability is not None]


class BaseGenerator:
    """Minimal generation engine that exposes logits-derived metrics each step."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        *,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = False,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def generate_stepwise(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 20,
        stop_at_eos: bool = True,
    ) -> GenerationResult:
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than 0.")

        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, device=self.device)
        else:
            attention_mask = attention_mask.to(self.device)

        eos_token_id = self.tokenizer.eos_token_id
        generated_ids: list[int] = []
        steps: list[TokenStep] = []
        entropy_history: list[float] = []
        token_id_history: list[int] = []

        with torch.no_grad():
            for step_index in range(max_new_tokens):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                next_token_logits = outputs.logits[:, -1, :]

                next_token_ids, probabilities = select_next_token(
                    next_token_logits,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )

                token_id = int(next_token_ids.item())
                token_probability = float(probabilities[0, token_id].item())
                entropy = float(compute_entropy(next_token_logits, dim=-1).item())
                token_text = self.tokenizer.decode(
                    [token_id],
                    clean_up_tokenization_spaces=False,
                )

                # ── stability tracking ──
                entropy_history.append(entropy)
                token_id_history.append(token_id)
                instability = detect_instability(entropy_history, token_id_history)

                steps.append(
                    TokenStep(
                        index=step_index,
                        token_id=token_id,
                        token_text=token_text,
                        token_probability=token_probability,
                        entropy=entropy,
                        instability=instability,
                    )
                )
                generated_ids.append(token_id)

                next_token = next_token_ids.unsqueeze(-1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=self.device)],
                    dim=-1,
                )

                if stop_at_eos and eos_token_id is not None and token_id == eos_token_id:
                    break

        generated_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        full_text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return GenerationResult(
            prompt=prompt,
            generated_token_ids=generated_ids,
            generated_text=generated_text,
            full_text=full_text,
            steps=steps,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 20,
        stop_at_eos: bool = True,
    ) -> GenerationResult:
        return self.generate_stepwise(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_at_eos=stop_at_eos,
        )


def generate_stepwise(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    *,
    max_tokens: int = 20,
    stop_at_eos: bool = True,
) -> GenerationResult:
    """Public Day 2 entry point for manual token-by-token decoding."""

    generator = BaseGenerator(
        model,
        tokenizer,
        do_sample=True,
    )
    return generator.generate_stepwise(
        prompt,
        max_new_tokens=max_tokens,
        stop_at_eos=stop_at_eos,
    )
