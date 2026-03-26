# Content Ideas

## P0 - Write First (Foundation)

### 1. Why Your LLM Inference is Slow
**Target**: AI engineers struggling with deployment
**Angle**: Python → C++ optimization journey
**Key Points**:
- Profile a typical Python inference loop
- Identify bottlenecks (GIL, memory, etc.)
- Rewrite hot path in C++
- Show 5-10x speedup

### 2. Running Llama-3-8B on Raspberry Pi
**Target**: Edge AI developers
**Angle**: Practical edge deployment guide
**Key Points**:
- Quantization strategies (FP16 → INT4)
- Memory optimization techniques
- Real-world benchmarks
- Full code walkthrough

---

## P1 - Write Second (Depth)

### 3. Vector DB from Scratch in Rust
**Target**: RAG/search infrastructure teams
**Key Points**:
- HNSW algorithm explained
- Memory layout decisions
- Rust implementation
- Benchmark vs FAISS

### 4. Continuous Batching Implementation
**Target**: LLM service providers
**Key Points**:
- Why batching improves throughput
- Continuous batching algorithm
- C++ scheduler implementation
- 3-5x throughput gains

### 5. Rust DataLoader for PyTorch
**Target**: ML training teams
**Key Points**:
- Python DataLoader bottlenecks
- Rust implementation with PyO3
- 5-10x data loading speedup
- Integration guide

### 6. LLM Quantization Guide
**Target**: Model deployment engineers
**Key Points**:
- GGUF vs AWQ vs GPTQ
- Implementation details
- Accuracy vs speed tradeoffs
- Code examples

---

## P2 - Write Third (Case Studies)

### 7. End-to-End Rust LLM Service
### 8. Client Optimization Case Study
### 9. Mini ONNX Compiler in Zig
### 10. Multimodal Edge Deployment

---

## Content Calendar

| Week | Article | Status |
|------|---------|--------|
| 1-2 | #1 LLM Inference Slow | Planned |
| 3-4 | #2 Raspberry Pi Llama | Planned |
| 5-6 | #3 Vector DB Rust | Planned |
| 7-8 | #4 Continuous Batching | Planned |
| 9-10 | #5 Rust DataLoader | Planned |
| 11-12 | #6 Quantization Guide | Planned |

---

## Distribution

| Platform | Format | Frequency |
|----------|--------|-----------|
| Blog | Full article | Weekly |
| Dev.to | Cross-post | Weekly |
| Twitter | Thread summary | Per article |
| LinkedIn | Professional summary | Per article |
| GitHub | Code examples | Per article |
