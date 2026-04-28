"""HuggingFace Inference API client for remote model inference.

Uses the chat_completion API (the modern, well-supported path) with
optional logprobs for entropy computation.  Falls back to heuristic
entropy estimation when logprobs are unavailable.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

from huggingface_hub import InferenceClient


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class RemoteTokenInfo:
    """Single generated token with probability information."""

    text: str
    logprob: float
    token_id: int
    top_logprobs: List[TopLogprob] = field(default_factory=list)


@dataclass
class TopLogprob:
    """One candidate from the top-N distribution at a generation step."""

    token: str
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

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_PROVIDER = "together"


class RemoteModelClient:
    """Wrapper around HF InferenceClient using chat_completion API."""

    def __init__(
        self,
        model_id: Optional[str] = None,
        token: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        self.model_id = model_id or os.getenv("HF_MODEL_ID", DEFAULT_MODEL)
        self.provider = provider or os.getenv("HF_PROVIDER", DEFAULT_PROVIDER)
        self.token = token or os.getenv("HF_TOKEN")
        if not self.token:
            raise RuntimeError(
                "HF_TOKEN environment variable is required for remote inference. "
                "Get a free token at https://huggingface.co/settings/tokens"
            )
        self.client = InferenceClient(
            model=self.model_id,
            provider=self.provider,
            token=self.token,
        )

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 40,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
    ) -> RemoteGenerationOutput:
        """Call HF chat_completion API and return token-level details."""

        response = self.client.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_new_tokens,
            temperature=max(temperature, 0.01),
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        generated_text = choice.message.content or ""
        finish_reason = str(choice.finish_reason) if choice.finish_reason else None

        # Parse token-level logprobs if available
        tokens: list[RemoteTokenInfo] = []

        if choice.logprobs and choice.logprobs.content:
            for lp_item in choice.logprobs.content:
                top_list: list[TopLogprob] = []
                if lp_item.top_logprobs:
                    for alt in lp_item.top_logprobs:
                        top_list.append(TopLogprob(
                            token=alt.token,
                            logprob=alt.logprob,
                        ))

                tokens.append(RemoteTokenInfo(
                    text=lp_item.token,
                    logprob=lp_item.logprob,
                    token_id=hash(lp_item.token) & 0xFFFFFFFF,  # synthetic ID
                    top_logprobs=top_list,
                ))
        else:
            # No logprobs available — synthesize a trace from the text
            token_strings = re.findall(r'\s+|\w+|[^\s\w]', generated_text)
            
            for i, token_text in enumerate(token_strings):
                # Detect repetition (checking i-2 because words and spaces interleave)
                is_repeat = i >= 2 and token_text.strip() and token_text == token_strings[i-2]
                
                # Synthetic proxy logprob: 
                # - If repeating: logprob near 0 (yielding low entropy)
                # - If normal: moderate logprob (yielding moderate entropy ~1.6)
                synthetic_logprob = -0.1 if is_repeat else -2.0
                
                tokens.append(RemoteTokenInfo(
                    text=token_text,
                    logprob=synthetic_logprob,
                    token_id=hash(token_text) & 0xFFFFFFFF,
                    top_logprobs=[],
                ))

        return RemoteGenerationOutput(
            generated_text=generated_text,
            tokens=tokens,
            finish_reason=finish_reason,
        )


# ---------------------------------------------------------------------------
# Entropy helpers
# ---------------------------------------------------------------------------

def entropy_from_top_logprobs(top_logprobs: List[TopLogprob]) -> float:
    """Approximate Shannon entropy from top-N token log-probabilities.

    Converts logprobs to probs, normalises over the observed set, then
    computes -sum(p * log(p)).
    """

    if not top_logprobs:
        return 0.0

    probs = [math.exp(t.logprob) for t in top_logprobs]
    total = sum(probs)
    if total <= 0:
        return 0.0

    probs = [p / total for p in probs]
    entropy = 0.0
    for p in probs:
        if p > 0:
            entropy -= p * math.log(p)

    return entropy


def entropy_from_single_logprob(logprob: float) -> float:
    """Estimate entropy when only the chosen token's logprob is known.

    Uses the heuristic: high probability (logprob near 0) = low entropy,
    low probability (large negative logprob) = high entropy.
    Maps -logprob linearly into a reasonable entropy range [0, ~4].
    """

    return min(abs(logprob) * 0.8, 6.0)
