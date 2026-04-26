"""Configuration objects for model loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class QuantizationConfig:
    """Optional quantization knobs for future large-model support."""

    load_in_4bit: bool = False
    load_in_8bit: bool = False

    def as_pretrained_kwargs(self) -> dict[str, Any]:
        if self.load_in_4bit and self.load_in_8bit:
            raise ValueError("Only one quantization mode can be enabled at a time.")

        kwargs: dict[str, Any] = {}
        if self.load_in_4bit:
            kwargs["load_in_4bit"] = True
        if self.load_in_8bit:
            kwargs["load_in_8bit"] = True
        return kwargs


@dataclass
class ModelConfig:
    """Base runtime configuration for a causal language model."""

    model_name: str = "distilgpt2"
    device: str | None = None
    trust_remote_code: bool = False
    local_files_only: bool = False
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
