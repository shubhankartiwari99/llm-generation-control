"""Model loading and runtime configuration.

Heavy loader imports are deferred so that importing model.config
(pure dataclasses) does not pull in torch.
"""

from llm_control.model.config import ModelConfig, QuantizationConfig


def __getattr__(name: str):
    """Lazy-load torch-dependent loader symbols."""

    if name in ("load_model", "load_model_and_tokenizer", "resolve_device"):
        from llm_control.model import loader
        return getattr(loader, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ModelConfig", "QuantizationConfig", "load_model", "load_model_and_tokenizer", "resolve_device"]
