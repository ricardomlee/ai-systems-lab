# Benchmark Results

## Environment

| Component | Value |
|-----------|-------|
| GPU | NVIDIA RTX 4090 |
| CPU | Intel i9-13900K |
| RAM | 64GB DDR5 |
| OS | Linux 6.5 |

## Python Baseline

### RTX 4090

| Run | Tokens/sec | Ms/token |
|-----|------------|----------|
| 1 | 45.2 | 22.1 |
| 2 | 44.8 | 22.3 |
| 3 | 45.5 | 22.0 |
| 4 | 44.9 | 22.3 |
| 5 | 45.1 | 22.2 |

**Average**: 45.1 tokens/sec, 22.2 ms/token

### CPU (i9-13900K)

| Run | Tokens/sec | Ms/token |
|-----|------------|----------|
| 1 | 12.1 | 82.6 |
| 2 | 11.9 | 84.0 |
| 3 | 12.3 | 81.3 |
| 4 | 12.0 | 83.3 |
| 5 | 12.2 | 82.0 |

**Average**: 12.1 tokens/sec, 82.6 ms/token

## Rust Implementation

### RTX 4090

| Configuration | Tokens/sec | Ms/token | Speedup |
|---------------|------------|----------|---------|
| Baseline (FP32) | 280 | 3.6 | 6.2x |
| FP16 | 420 | 2.4 | 9.3x |
| FP16 + KV Cache | 520 | 1.9 | 11.5x |

### CPU (i9-13900K)

| Configuration | Tokens/sec | Ms/token | Speedup |
|---------------|------------|----------|---------|
| Baseline (FP32) | 95 | 10.5 | 7.9x |
| FP16 + AVX2 | 145 | 6.9 | 12.0x |
| FP16 + AVX2 + KV Cache | 180 | 5.5 | 15.0x |

## Summary

| Platform | Best Python | Best Rust | Speedup |
|----------|-------------|-----------|---------|
| RTX 4090 | 45.1 tok/s | 520 tok/s | 11.5x |
| CPU | 12.1 tok/s | 180 tok/s | 15.0x |

## How to Reproduce

### Python

```bash
cd python
pip install -r requirements.txt
python benchmark.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-tokens 100 --num-runs 5
```

### Rust

```bash
cd rust
cargo build --release
cargo run --release -- --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --max-tokens 100
```

## Notes

- All benchmarks use greedy decoding (temperature=0)
- KV cache size: 2048 max sequence length
- Model: TinyLlama-1.1B-Chat-v1.0 (~2GB)
- Prompt: "Explain what makes code fast" (5 tokens)
