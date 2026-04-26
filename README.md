# llm-generation-control

Controllable generation engine with token-level observability and adaptive decoding.

## Design

This repo is structured as a system, not a single demo script:

- `llm_control/model/` owns model loading, quantization flags, and device selection.
- `llm_control/generation/` owns the token loop and sampling decisions.
- `llm_control/metrics/` owns entropy and other logits-derived signals.
- `llm_control/control/` is reserved for adaptive policies.
- `llm_control/logging/` is reserved for token-level traces and persisted runs.
- `llm_control/evaluation/` is reserved for experiments and analysis.

## Day 1 Scope

The initial vertical slice is:

`prompt -> logits -> entropy -> print`

Implemented modules:

- `llm_control/model/loader.py`
- `llm_control/generation/base_generator.py`
- `llm_control/metrics/entropy.py`
- `scripts/run_basic.py`

## Quickstart

Use a cached Hugging Face model or a local model path:

```bash
./venv/bin/python scripts/run_basic.py --model distilgpt2 --prompt "Explain entropy in one sentence."
```

If you are offline and the model is not cached yet, point `--model` to a local path or rerun once with network access to cache it.

## Next Milestone

Add adaptive decoding policies that react to entropy and instability inside `llm_control/control/` and `llm_control/generation/adaptive_generator.py`.

