"""Reproducible evaluation pipeline for controlled generation.

Compares plain vs adaptive generation modes using multi-signal metrics:
- Repetition: proportion of consecutive duplicate tokens (captures repetition loops)
- Length Variance: consistency of output lengths across runs

Computes delta metrics to show adaptive improvements over plain baseline.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_control.generation.base_generator import BaseGenerator
from llm_control.generation.adaptive_generator import generate_adaptive


def repetition_score(text: str) -> float:
    """Measure proportion of consecutive duplicate tokens.
    
    Captures repetition loops — a key failure mode in language models
    that the control system is designed to prevent.
    
    Range: [0, 1]
    - 0 = no repetition
    - 1 = all tokens are repetitions
    """
    tokens = text.split()
    if len(tokens) < 2:
        return 0.0
    
    repeats = sum(1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1])
    return repeats / len(tokens)


def length_variance(outputs: list[str]) -> float:
    """Measure variance in output lengths across runs.
    
    Captures whether the model produces consistent-length outputs,
    indicating stable generation behavior.
    """
    lengths = np.array([len(o.split()) for o in outputs])
    if len(lengths) < 2:
        return 0.0
    return float(np.std(lengths))


def compute_instability(outputs: list[str]) -> dict[str, float]:
    """Compute multi-signal instability metrics.
    
    Returns:
        Dict with:
        - repetition: avg proportion of consecutive duplicate tokens
        - length_variance: std deviation of output lengths
    """
    rep_scores = [repetition_score(o) for o in outputs]
    length_var = length_variance(outputs)
    
    return {
        "repetition": float(np.mean(rep_scores)),
        "length_variance": float(length_var),
    }


def compute_delta(plain_metrics: dict[str, float], adaptive_metrics: dict[str, float]) -> dict[str, float]:
    """Compute improvement from adaptive over plain.
    
    Positive values = improvement (lower is better)
    - repetition_reduction: how much adaptive reduces repetition (plain - adaptive)
    - length_variance_change: how much adaptive stabilizes length (adaptive - plain, inverted)
    
    Returns:
        Dict with delta metrics
    """
    return {
        "repetition_reduction": float(plain_metrics["repetition"] - adaptive_metrics["repetition"]),
        "length_variance_change": float(adaptive_metrics["length_variance"] - plain_metrics["length_variance"]),
    }


def run_evaluation(
    model_name: str,
    prompts: list[str],
    runs: int = 3,
    seed: int = 42,
    max_tokens: int = 50,
    verbose: bool = True,
) -> dict[str, list[dict]]:
    """Evaluate model stability comparing plain vs adaptive generation.
    
    Args:
        model_name: HuggingFace model ID (e.g., "distilgpt2")
        prompts: List of prompts to evaluate
        runs: Number of generation runs per prompt per mode
        seed: Random seed for reproducibility
        max_tokens: Max tokens to generate per prompt
        verbose: Whether to print per-prompt progress
        
    Returns:
        Dict with results per prompt and aggregate metrics
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    if verbose:
        print(f"Loading model: {model_name} on device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    plain_generator = BaseGenerator(model, tokenizer, temperature=1.0, do_sample=True)
    
    results = []
    
    for prompt_idx, prompt in enumerate(prompts, 1):
        if verbose:
            print(f"\n[{prompt_idx}/{len(prompts)}] Evaluating: {prompt[:60]}")
        
        plain_outputs = []
        adaptive_outputs = []
        
        for run_idx in range(runs):
            # Plain generation
            try:
                result = plain_generator.generate_stepwise(prompt, max_new_tokens=max_tokens)
                output_text = result.generated_text.strip()
                plain_outputs.append(output_text)
            except Exception as e:
                if verbose:
                    print(f"  Plain run {run_idx + 1}/{runs}: ERROR - {str(e)}")
                plain_outputs.append("")
            
            # Adaptive generation
            try:
                result = generate_adaptive(
                    model, tokenizer, prompt, max_tokens=max_tokens, verbose=False
                )
                output_text = result.generated_text.strip()
                adaptive_outputs.append(output_text)
            except Exception as e:
                if verbose:
                    print(f"  Adaptive run {run_idx + 1}/{runs}: ERROR - {str(e)}")
                adaptive_outputs.append("")
        
        # Compute metrics for both modes
        plain_metrics = compute_instability(plain_outputs)
        adaptive_metrics = compute_instability(adaptive_outputs)
        delta = compute_delta(plain_metrics, adaptive_metrics)
        
        prompt_result = {
            "prompt": prompt,
            "plain": {
                "repetition": round(plain_metrics["repetition"], 4),
                "length_variance": round(plain_metrics["length_variance"], 4),
            },
            "adaptive": {
                "repetition": round(adaptive_metrics["repetition"], 4),
                "length_variance": round(adaptive_metrics["length_variance"], 4),
            },
            "delta": {
                "repetition_reduction": round(delta["repetition_reduction"], 4),
                "length_variance_change": round(delta["length_variance_change"], 4),
            },
            "total_runs": runs,
        }
        
        results.append(prompt_result)
        
        # Print per-prompt summary
        if verbose:
            print(f"  Plain:    Rep={plain_metrics['repetition']:.4f} LenVar={plain_metrics['length_variance']:.4f}")
            print(f"  Adaptive: Rep={adaptive_metrics['repetition']:.4f} LenVar={adaptive_metrics['length_variance']:.4f}")
            print(f"  Delta:    Rep↓={delta['repetition_reduction']:+.4f} LenVar={delta['length_variance_change']:+.4f}")
    
    return {"prompts": results}


def summarize(results: dict) -> dict:
    """Compute aggregate statistics across all prompts."""
    plain_repetitions = [p["plain"]["repetition"] for p in results["prompts"]]
    plain_length_vars = [p["plain"]["length_variance"] for p in results["prompts"]]
    
    adaptive_repetitions = [p["adaptive"]["repetition"] for p in results["prompts"]]
    adaptive_length_vars = [p["adaptive"]["length_variance"] for p in results["prompts"]]
    
    rep_reductions = [p["delta"]["repetition_reduction"] for p in results["prompts"]]
    length_var_changes = [p["delta"]["length_variance_change"] for p in results["prompts"]]
    
    return {
        "plain": {
            "avg_repetition": round(float(np.mean(plain_repetitions)), 4),
            "std_repetition": round(float(np.std(plain_repetitions)), 4),
            "avg_length_variance": round(float(np.mean(plain_length_vars)), 4),
            "std_length_variance": round(float(np.std(plain_length_vars)), 4),
        },
        "adaptive": {
            "avg_repetition": round(float(np.mean(adaptive_repetitions)), 4),
            "std_repetition": round(float(np.std(adaptive_repetitions)), 4),
            "avg_length_variance": round(float(np.mean(adaptive_length_vars)), 4),
            "std_length_variance": round(float(np.std(adaptive_length_vars)), 4),
        },
        "improvement": {
            "avg_repetition_reduction": round(float(np.mean(rep_reductions)), 4),
            "std_repetition_reduction": round(float(np.std(rep_reductions)), 4),
            "avg_length_variance_change": round(float(np.mean(length_var_changes)), 4),
            "std_length_variance_change": round(float(np.std(length_var_changes)), 4),
        },
        "num_prompts": len(results["prompts"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model stability across prompts."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="HuggingFace model ID (e.g., distilgpt2, gpt2)",
    )
    parser.add_argument(
        "--prompts_file",
        required=True,
        help="Path to JSON file with list of prompts",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of generation runs per prompt (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50,
        help="Max tokens to generate per prompt (default: 50)",
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON file for results",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Output only summary JSON (minimal, shareable format)",
    )
    
    args = parser.parse_args()
    
    # Load prompts
    prompts_path = Path(args.prompts_file)
    if not prompts_path.exists():
        print(f"ERROR: Prompts file not found: {prompts_path}")
        sys.exit(1)
    
    with open(prompts_path) as f:
        prompts = json.load(f)
    
    if not isinstance(prompts, list):
        print("ERROR: Prompts file must contain a JSON array")
        sys.exit(1)
    
    if not args.summary_only:
        print(f"Loaded {len(prompts)} prompts from {prompts_path}")
    
    # Run evaluation
    results = run_evaluation(
        model_name=args.model,
        prompts=prompts,
        runs=args.runs,
        seed=args.seed,
        max_tokens=args.max_tokens,
        verbose=not args.summary_only,
    )
    
    # Compute summary
    summary = summarize(results)
    
    # Handle summary-only output
    if args.summary_only:
        clean_output = {
            "model": args.model,
            "runs": args.runs,
            "seed": args.seed,
            "num_prompts": summary["num_prompts"],
            "avg_repetition_reduction": summary["improvement"]["avg_repetition_reduction"],
            "std_repetition_reduction": summary["improvement"]["std_repetition_reduction"],
            "avg_length_variance_change": summary["improvement"]["avg_length_variance_change"],
            "std_length_variance_change": summary["improvement"]["std_length_variance_change"],
        }
        print(json.dumps(clean_output, indent=2))
        return
    
    # Print results
    print("\n" + "=" * 70)
    print("AGGREGATE SUMMARY")
    print("=" * 70)
    print("\nPlain Generation:")
    print(f"  Avg Repetition:      {summary['plain']['avg_repetition']:.4f} ± {summary['plain']['std_repetition']:.4f}")
    print(f"  Avg Length Variance: {summary['plain']['avg_length_variance']:.4f} ± {summary['plain']['std_length_variance']:.4f}")
    
    print("\nAdaptive Generation:")
    print(f"  Avg Repetition:      {summary['adaptive']['avg_repetition']:.4f} ± {summary['adaptive']['std_repetition']:.4f}")
    print(f"  Avg Length Variance: {summary['adaptive']['avg_length_variance']:.4f} ± {summary['adaptive']['std_length_variance']:.4f}")
    
    print("\nImprovement (Adaptive vs Plain):")
    print(f"  Repetition Reduction: {summary['improvement']['avg_repetition_reduction']:+.4f} ± {summary['improvement']['std_repetition_reduction']:.4f}")
    print(f"  Length Variance Change: {summary['improvement']['avg_length_variance_change']:+.4f} ± {summary['improvement']['std_length_variance_change']:.4f}")
    print(f"  Prompts Evaluated: {summary['num_prompts']}")
    
    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)
    for prompt_result in results["prompts"]:
        print(f"\nPrompt: {prompt_result['prompt'][:70]}")
        print(f"  Plain    → Rep: {prompt_result['plain']['repetition']:.4f}, LenVar: {prompt_result['plain']['length_variance']:.4f}")
        print(f"  Adaptive → Rep: {prompt_result['adaptive']['repetition']:.4f}, LenVar: {prompt_result['adaptive']['length_variance']:.4f}")
        print(f"  Delta    → Rep↓: {prompt_result['delta']['repetition_reduction']:+.4f}, LenVar: {prompt_result['delta']['length_variance_change']:+.4f}")
    
    # Save output if requested
    if args.output:
        output_path = Path(args.output)
        output_data = {
            "metadata": {
                "model": args.model,
                "runs": args.runs,
                "seed": args.seed,
                "max_tokens": args.max_tokens,
            },
            "summary": summary,
            "results": results,
        }
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
