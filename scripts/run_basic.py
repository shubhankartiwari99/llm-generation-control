"""Manual stepwise generation entry point with entropy tracing."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_control.generation.base_generator import generate_stepwise
from llm_control.logging.storage import RunStorage
from llm_control.metrics.confidence import compute_confidence
from llm_control.model.loader import load_model


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=os.getenv("PROMPT", "Explain entropy in one sentence."),
        help="Prompt to feed into the language model.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "mistral"),
        help="Model alias: 'mistral' for Mistral-7B or 'small' for distilgpt2.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("MAX_NEW_TOKENS", "8")),
        help="Number of tokens to decode step-by-step.",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE"),
        help="Optional explicit device override such as cpu, mps, or cuda.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        default=os.getenv("LOCAL_FILES_ONLY", "").lower() in {"1", "true", "yes"},
        help="Require models to be loaded from local cache or a local path.",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Persist generation run metadata and detailed trace to logs/.",
    )
    return parser


def main() -> int:
    load_env_file(ROOT / ".env")
    args = build_parser().parse_args()

    model, tokenizer = load_model(
        model_name=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    result = generate_stepwise(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_new_tokens,
    )

    if args.log:
        storage = RunStorage()
        confidence = compute_confidence(result.steps)
        response_data = {
            "output": result.generated_text,
            "full_text": result.full_text,
            "confidence": confidence.confidence,
            "regenerations": 0,
            "steps": [
                {
                    "token": step.token_text,
                    "entropy": step.entropy,
                    "instability": step.instability,
                }
                for step in result.steps
            ],
        }
        trace_id = storage.log_run(prompt=args.prompt, mode="plain", response_data=response_data)
        print(f"Logged run to {trace_id}")

    print(f"model: {args.model}")
    print(f"prompt: {args.prompt}")
    print()
    for step in result.steps:
        token_display = step.token_text.encode("unicode_escape").decode("ascii")
        print(f"Step {step.index} | Token: {token_display} | Entropy: {step.entropy:.4f}")
        if step.instability:
            print(f"WARNING: Instability detected: {step.instability}")

    print("\nFinal Output:\n")
    print(result.full_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
