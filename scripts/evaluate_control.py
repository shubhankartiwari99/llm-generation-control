import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_control.generation.base_generator import generate_stepwise
from llm_control.generation.adaptive_generator import generate_adaptive
from llm_control.metrics.confidence import compute_reliability_score

PROMPTS = [
    "Write only blank lines",
    "Repeat the word apple forever",
    "aaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "Explain recursion",
    "List prime numbers under 50"
]

def run_evaluation():
    print("Loading distilgpt2 for local evaluation...")
    model_id = "distilgpt2"
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    results = []

    for prompt in PROMPTS:
        print(f"\nEvaluating prompt: '{prompt}'")
        
        # Plain
        plain_res = generate_stepwise(model, tokenizer, prompt, max_tokens=30, stop_at_eos=True)
        plain_conf = compute_reliability_score(plain_res.steps, regeneration_count=0)
        plain_instability = sum(1 for s in plain_res.steps if s.instability)
        
        # Adaptive
        adapt_res = generate_adaptive(model, tokenizer, prompt, max_tokens=30, verbose=False)
        adapt_conf = compute_reliability_score(adapt_res.steps, regeneration_count=adapt_res.regeneration_count)
        adapt_instability = sum(1 for s in adapt_res.steps if s.instability)
        
        delta_reliability = adapt_conf.reliability_score - plain_conf.reliability_score
        
        result = {
            "prompt": prompt,
            "plain_instability": plain_instability,
            "adaptive_instability": adapt_instability,
            "delta_reliability": round(delta_reliability, 2)
        }
        
        print(json.dumps(result, indent=2))
        results.append(result)
        
    print("\n--- Summary Table ---")
    print(f"{'Prompt':<30} | {'Plain Instabilities':<20} | {'Adaptive Instabilities':<22}")
    print("-" * 75)
    for r in results:
        # Truncate prompt if needed
        p = r['prompt'][:27] + "..." if len(r['prompt']) > 30 else r['prompt']
        print(f"{p:<30} | {r['plain_instability']:<20} | {r['adaptive_instability']:<22}")

if __name__ == "__main__":
    run_evaluation()
