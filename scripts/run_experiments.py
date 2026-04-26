"""Run plain-vs-adaptive confidence comparison experiments."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_control.evaluation.experiments import DEFAULT_PROMPTS, format_comparison_rows, run_comparison_experiment
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
        "--model",
        default=os.getenv("MODEL_NAME", "distilgpt2"),
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("EXPERIMENT_MAX_NEW_TOKENS", "20")),
        help="Maximum number of generated tokens per run.",
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
        "--seed",
        type=int,
        default=None if os.getenv("EXPERIMENT_SEED") is None else int(os.getenv("EXPERIMENT_SEED", "0")),
        help="Optional base torch seed for reproducible sampling.",
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
    results = run_comparison_experiment(
        model,
        tokenizer,
        DEFAULT_PROMPTS,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    for line in format_comparison_rows(results):
        print(line)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
