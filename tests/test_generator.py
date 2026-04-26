from __future__ import annotations

from dataclasses import dataclass

import torch

from llm_control.generation.base_generator import BaseGenerator, generate_stepwise
from llm_control.metrics.stability import REPETITION_LOOP


@dataclass
class FakeOutput:
    logits: torch.Tensor


class FakeTokenizer:
    eos_token_id = 2
    eos_token = "<eos>"
    pad_token = "<pad>"

    def __call__(self, text: str, return_tensors: str = "pt") -> dict[str, torch.Tensor]:
        del text, return_tensors
        return {
            "input_ids": torch.tensor([[0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1]], dtype=torch.long),
        }

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        del clean_up_tokenization_spaces
        if isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = [int(token_id) for token_id in token_ids]
        mapping = {0: "<bos>", 1: "hello", 2: "<eos>"}
        if skip_special_tokens:
            ids = [token_id for token_id in ids if token_id not in {0, 2}]
        return "".join(mapping[token_id] for token_id in ids)


class FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeOutput:
        del input_ids, attention_mask
        vocab_size = 3
        logits = torch.full((1, 1, vocab_size), -10.0, dtype=torch.float32, device=self.anchor.device)
        next_token = 1 if self.calls == 0 else 2
        logits[0, 0, next_token] = 10.0
        self.calls += 1
        return FakeOutput(logits=logits)


class RepetitionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> FakeOutput:
        del input_ids, attention_mask
        logits = torch.tensor([[[10.0, -10.0, -10.0]]], dtype=torch.float32, device=self.anchor.device)
        return FakeOutput(logits=logits)


def test_base_generator_emits_token_steps() -> None:
    generator = BaseGenerator(FakeModel(), FakeTokenizer())
    result = generator.generate_stepwise("test", max_new_tokens=4)

    assert result.generated_token_ids == [1, 2]
    assert result.generated_text == "hello"
    assert len(result.steps) == 2
    assert result.steps[0].token_text == "hello"
    assert result.steps[1].token_id == 2
    assert len(result.entropy_trace) == 2


def test_generate_stepwise_function_uses_manual_loop() -> None:
    result = generate_stepwise(FakeModel(), FakeTokenizer(), "test", max_tokens=4)

    assert result.generated_token_ids == [1, 2]
    assert result.full_text == "hello"


def test_generator_surfaces_repetition_instability() -> None:
    generator = BaseGenerator(RepetitionModel(), FakeTokenizer(), do_sample=False)
    result = generator.generate_stepwise("test", max_new_tokens=4, stop_at_eos=False)

    assert result.steps[2].instability == REPETITION_LOOP
    assert REPETITION_LOOP in result.instability_trace
