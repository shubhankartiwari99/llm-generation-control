"""Generation loops and sampling utilities.

Heavy imports (base_generator, adaptive_generator) are lazy-loaded to
avoid pulling in torch when only the type definitions are needed.
"""

from llm_control.generation.types import GenerationResult, TokenStep


def __getattr__(name: str):
    """Lazy-load torch-dependent generators on first access."""

    if name == "BaseGenerator":
        from llm_control.generation.base_generator import BaseGenerator
        return BaseGenerator

    if name == "generate_stepwise":
        from llm_control.generation.base_generator import generate_stepwise
        return generate_stepwise

    if name == "generate_adaptive":
        from llm_control.generation.adaptive_generator import generate_adaptive
        return generate_adaptive

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseGenerator",
    "GenerationResult",
    "TokenStep",
    "generate_adaptive",
    "generate_stepwise",
]
