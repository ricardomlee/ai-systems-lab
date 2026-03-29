"""
Baseline LLM inference implementation in Python.
This is intentionally simple - we're measuring baseline performance.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time


class LLMInference:
    """Simple LLM inference wrapper."""

    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on device: {self.device}")

    def generate(self, prompt: str, max_tokens: int = 100, temperature: float = 1.0) -> str:
        """Generate text autoregressively."""
        tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        generated = []

        with torch.no_grad():
            for _ in range(max_tokens):
                # Forward pass
                outputs = self.model(tokens)
                logits = outputs.logits[:, -1, :]

                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature

                # Sample next token
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                # Append and continue
                tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
                generated.append(next_token.item())

        return self.tokenizer.decode(generated)

    def generate_with_timing(self, prompt: str, max_tokens: int = 100) -> tuple[str, dict]:
        """Generate text and return timing statistics."""
        # Warmup
        self.generate(prompt, max_tokens=10)

        # Measure
        start = time.perf_counter()
        output = self.generate(prompt, max_tokens=max_tokens)
        elapsed = time.perf_counter() - start

        stats = {
            "elapsed_sec": elapsed,
            "tokens_per_sec": max_tokens / elapsed,
            "ms_per_token": (elapsed * 1000) / max_tokens,
            "max_tokens": max_tokens,
        }

        return output, stats


def main():
    """Run a simple generation example."""
    inference = LLMInference()

    prompt = "Explain what makes code fast"
    print(f"\nPrompt: {prompt}")
    print("-" * 50)

    output, stats = inference.generate_with_timing(prompt, max_tokens=100)

    print(f"\nGenerated output:\n{output}")
    print("\n" + "-" * 50)
    print(f"Time: {stats['elapsed_sec']:.2f}s")
    print(f"Tokens/sec: {stats['tokens_per_sec']:.2f}")
    print(f"Ms/token: {stats['ms_per_token']:.2f}")


if __name__ == "__main__":
    main()
