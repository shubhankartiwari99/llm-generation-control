"""Core package for controllable LLM generation.

Heavy dependencies (torch, transformers) are loaded lazily so that
lightweight modules (metrics, control, remote inference) can be imported
without pulling in ~500MB of torch — critical for Render free tier.
"""

from llm_control.generation.types import GenerationResult, TokenStep


def __getattr__(name: str):
    """Lazy-load torch-dependent symbols on first access."""

    if name == "BaseGenerator":
        from llm_control.generation.base_generator import BaseGenerator
        return BaseGenerator

    if name == "generate_stepwise":
        from llm_control.generation.base_generator import generate_stepwise
        return generate_stepwise

    if name == "ModelConfig":
        from llm_control.model.config import ModelConfig
        return ModelConfig

    if name == "QuantizationConfig":
        from llm_control.model.config import QuantizationConfig
        return QuantizationConfig

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "ModelConfig",
    "QuantizationConfig",
    "TokenStep",
    "generate_stepwise",
]
