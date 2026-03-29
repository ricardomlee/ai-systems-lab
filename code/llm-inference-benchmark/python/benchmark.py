"""
Benchmark script for Python LLM inference.
Runs multiple iterations and reports statistics.
"""

import argparse
import json
import time
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def run_benchmark(model_name: str, max_tokens: int, num_runs: int = 5):
    """Run benchmark and return statistics."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    print(f"Running on device: {device}")
    print(f"Warming up...")

    # Warmup
    prompt = "The quick brown fox"
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        for _ in range(10):
            outputs = model(tokens)
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)

    print(f"Running {num_runs} benchmark iterations...")

    results = []
    test_prompt = "Explain what makes code fast"
    input_tokens = len(tokenizer.encode(test_prompt))

    for i in range(num_runs):
        print(f"  Run {i + 1}/{num_runs}...", end=" ", flush=True)

        tokens = tokenizer.encode(test_prompt, return_tensors="pt").to(device)
        generated_tokens = 0

        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(max_tokens):
                outputs = model(tokens)
                logits = outputs.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)
                generated_tokens += 1
        elapsed = time.perf_counter() - start

        stats = {
            "run": i + 1,
            "generated_tokens": generated_tokens,
            "elapsed_sec": elapsed,
            "tokens_per_sec": generated_tokens / elapsed,
            "ms_per_token": (elapsed * 1000) / generated_tokens,
        }
        results.append(stats)

        print(f"{stats['tokens_per_sec']:.2f} tokens/sec")

    # Aggregate statistics
    avg_tokens_per_sec = sum(r["tokens_per_sec"] for r in results) / len(results)
    avg_ms_per_token = sum(r["ms_per_token"] for r in results) / len(results)
    min_tokens_per_sec = min(r["tokens_per_sec"] for r in results)
    max_tokens_per_sec = max(r["tokens_per_sec"] for r in results)

    summary = {
        "model": model_name,
        "device": str(device),
        "max_tokens": max_tokens,
        "num_runs": num_runs,
        "avg_tokens_per_sec": avg_tokens_per_sec,
        "avg_ms_per_token": avg_ms_per_token,
        "min_tokens_per_sec": min_tokens_per_sec,
        "max_tokens_per_sec": max_tokens_per_sec,
    }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Benchmark LLM inference")
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Model name or path",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=5,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for results (JSON)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("LLM Inference Benchmark - Python")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Num runs: {args.num_runs}")
    print("=" * 60)

    results = run_benchmark(args.model, args.max_tokens, args.num_runs)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Device: {results['device']}")
    print(f"Avg tokens/sec: {results['avg_tokens_per_sec']:.2f}")
    print(f"Avg ms/token: {results['avg_ms_per_token']:.2f}")
    print(f"Range: {results['min_tokens_per_sec']:.2f} - {results['max_tokens_per_sec']:.2f} tokens/sec")
    print("=" * 60)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
