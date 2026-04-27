"""HuggingFace Inference API client for remote model inference.

Wraps huggingface_hub.InferenceClient to provide token-level details
(logprobs, top-N alternatives) needed by the entropy / stability pipeline.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional

from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Data containers for the raw API response
# ---------------------------------------------------------------------------

@dataclass
class RemoteTokenInfo:
    """Single generated token with logprob and top-N alternatives."""

    text: str
    logprob: float
    token_id: int
    top_tokens: List[TopTokenInfo] = field(default_factory=list)


@dataclass
class TopTokenInfo:
    """One candidate from the top-N distribution at a generation step."""

    text: str
    token_id: int
    logprob: float


@dataclass
class RemoteGenerationOutput:
    """Full output from a single HF Inference API call."""

    generated_text: str
    tokens: List[RemoteTokenInfo] = field(default_factory=list)
    finish_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.1"
TOP_N_TOKENS = 20  # enough to approximate entropy


class RemoteModelClient:
    """Thin wrapper around HF InferenceClient with entropy-friendly defaults."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv("HF_MODEL_ID", DEFAULT_MODEL)
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for remote inference. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.client = InferenceClient(model=self.model_id, token=self.token)

    # ----- public API -----

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> RemoteGenerationOutput:
        """Call HF Inference API and return token-level details."""

        response = self.client.text_generation(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),  # API rejects exactly 0
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            details=True,
            top_n_tokens=TOP_N_TOKENS,
            return_full_text=False,
        )

        # Parse the response into our dataclasses
        tokens: list[RemoteTokenInfo] = []
        if hasattr(response, "details") and response.details:
            for tok in response.details.tokens:
                top_list: list[TopTokenInfo] = []
                if tok.top_tokens:
                    for alt in tok.top_tokens:
                        top_list.append(
                            TopTokenInfo(
                                text=alt.text if hasattr(alt, "text") else str(alt.get("text", "")),
                                token_id=alt.id if hasattr(alt, "id") else int(alt.get("id", 0)),
                                logprob=alt.logprob if hasattr(alt, "logprob") else float(alt.get("logprob", 0.0)),
                            )
                        )
                tokens.append(
                    RemoteTokenInfo(
                        text=tok.text if hasattr(tok, "text") else str(tok),
                        logprob=tok.logprob if hasattr(tok, "logprob") else 0.0,
                        token_id=tok.id if hasattr(tok, "id") else 0,
                        top_tokens=top_list,
                    )
                )

        generated_text = response.generated_text if hasattr(response, "generated_text") else str(response)
        finish_reason = None
        if hasattr(response, "details") and response.details:
            finish_reason = getattr(response.details, "finish_reason", None)

        return RemoteGenerationOutput(
            generated_text=generated_text,
            tokens=tokens,
            finish_reason=str(finish_reason) if finish_reason else None,
        )


# ---------------------------------------------------------------------------
# Entropy helpers for logprob distributions
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(top_tokens: List[TopTokenInfo]) -> float:
    """Approximate Shannon entropy from top-N token log-probabilities.

    We convert logprobs → probs, normalise over the observed set, then
    compute -Σ p·log(p).  This is an approximation because we only see
    top-N of the full vocabulary, but with N=20 it captures the
    high-probability mass that dominates the entropy value.
    """

    if not top_tokens:
        return 0.0

    # logprobs → probabilities
    probs = [math.exp(t.logprob) for t in top_tokens]
    total = sum(probs)
    if total <= 0:
        return 0.0

    # normalise so they sum to 1 over the observed set
    probs = [p / total for p in probs]

    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy
