"""Run adaptive decoding with closed-loop control actions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_control.generation.adaptive_generator import generate_adaptive
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
    seed_value = os.getenv("ADAPTIVE_SEED", os.getenv("SEED"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompt",
        default=os.getenv("ADAPTIVE_PROMPT", "Write only blank lines"),
        help="Prompt to feed into the adaptive decoder.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "distilgpt2"),
        help="Hugging Face model id or local model path.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.getenv("ADAPTIVE_MAX_NEW_TOKENS", "20")),
        help="Maximum number of adaptive decoding steps.",
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
        default=None if seed_value is None else int(seed_value),
        help="Optional torch seed for reproducible sampling.",
    )
    return parser


def main() -> int:
    load_env_file(ROOT / ".env")
    args = build_parser().parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, tokenizer = load_model(
        model_name=args.model,
        device=args.device,
        local_files_only=args.local_files_only,
    )

    output = generate_adaptive(
        model,
        tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_new_tokens,
    )

    print("\nFinal Output:\n")
    print(output.full_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
