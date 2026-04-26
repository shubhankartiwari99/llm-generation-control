"""Utilities for loading a causal LM and tokenizer."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from llm_control.model.config import ModelConfig


@contextmanager
def _offline_hf_mode(enabled: bool):
    if not enabled:
        yield
        return

    previous_values = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def resolve_device(preferred: str | None = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_pretrained_source(model_name: str, *, local_files_only: bool) -> str:
    if not local_files_only:
        return model_name

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return str(model_path)

    cache_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    repo_cache_dir = cache_home / "hub" / f"models--{model_name.replace('/', '--')}"
    refs_main = repo_cache_dir / "refs" / "main"
    snapshots_dir = repo_cache_dir / "snapshots"

    if refs_main.exists():
        revision = refs_main.read_text().strip()
        snapshot_path = snapshots_dir / revision
        if snapshot_path.exists():
            return str(snapshot_path)

    if snapshots_dir.exists():
        snapshots = sorted(
            (path for path in snapshots_dir.iterdir() if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if snapshots:
            return str(snapshots[0])

    return model_name


def load_model_and_tokenizer(
    config: ModelConfig,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a tokenizer and causal LM using the repo's shared config object."""

    device = resolve_device(config.device)
    pretrained_source = resolve_pretrained_source(
        config.model_name,
        local_files_only=config.local_files_only,
    )

    try:
        with _offline_hf_mode(config.local_files_only):
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_source,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
            )
            model = AutoModelForCausalLM.from_pretrained(
                pretrained_source,
                local_files_only=config.local_files_only,
                trust_remote_code=config.trust_remote_code,
                **config.quantization.as_pretrained_kwargs(),
            )
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load model '{config.model_name}'. "
            "If you are offline, point MODEL_NAME to a local model path or a cached Hugging Face model."
        ) from exc

    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model.to(device)
    model.eval()
    return model, tokenizer


def load_mistral_7b(device: str | None = None) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load the Mistral 7B instruct model with the best available local device support."""

    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    device = resolve_device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    if device == "cpu":
        raise RuntimeError(
            "Mistral 7B on CPU is not supported. Please use MPS or CUDA, or fall back to a smaller model."
        )

    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if device == "cuda":
        model_kwargs["load_in_4bit"] = False
    elif device == "mps":
        # Use native float16 on Apple Silicon for compatibility.
        model_kwargs["load_in_4bit"] = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=False,
        **model_kwargs,
    )
    model.eval()
    return model, tokenizer


def load_model(
    model_name: str = "distilgpt2",
    *,
    device: str | None = None,
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Convenience wrapper for the first-generation scripts."""

    config = ModelConfig(
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
        trust_remote_code=trust_remote_code,
    )
    return load_model_and_tokenizer(config)
