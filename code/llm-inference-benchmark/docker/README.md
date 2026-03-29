# LLM Inference Benchmark - Docker Environment

Docker setup for running LLM inference benchmarks with CUDA support.

## Prerequisites

### Windows 11 + WSL2 Setup

1. **Install WSL2** (if not already done):
   ```powershell
   wsl --install
   ```

2. **Install NVIDIA Container Toolkit** on Windows:
   - Install CUDA Toolkit for WSL: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html
   - Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

3. **Verify GPU access in WSL**:
   ```bash
   nvidia-smi
   ```

## Quick Start

### Option 1: Using docker-compose (Recommended)

```bash
cd docker/

# Build and run both benchmarks
./run.sh both

# Or run individually
./run.sh python    # Python only
./run.sh rust      # Rust only
./run.sh bash      # Interactive shell
```

### Option 2: Manual docker-compose

```bash
cd docker/

# Build the container
docker compose build

# Run Python benchmark
docker compose run --rm llm-benchmark python python/benchmark.py --max-tokens 100

# Run Rust benchmark
docker compose run --rm llm-benchmark bash -c "cd /app/rust && cargo build --release && cargo run --release -- --max-tokens 100"

# Interactive shell
docker compose run --rm llm-benchmark bash
```

### Option 3: Plain Docker (without compose)

```bash
# Build
docker build -t llm-benchmark .

# Run Python
docker run --gpus all -it --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/../python:/app/python \
    -v $(pwd)/../rust:/app/rust \
    llm-benchmark python python/benchmark.py --max-tokens 100

# Run Rust
docker run --gpus all -it --rm \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd)/../python:/app/python \
    -v $(pwd)/../rust:/app/rust \
    llm-benchmark bash -c "cd /app/rust && cargo build --release && cargo run --release -- --max-tokens 100"
```

### Run without Docker (WSL2 Native)

```bash
# Python
cd code/llm-inference-benchmark/python
pip install -r requirements.txt
python benchmark.py --max-tokens 100

# Rust
cd code/llm-inference-benchmark/rust
cargo build --release
cargo run --release -- --max-tokens 100
```

## Container Options

| Command | Description |
|---------|-------------|
| `python` | Run Python benchmark only |
| `rust` | Run Rust benchmark only |
| `both` | Run both benchmarks sequentially |
| `bash` | Interactive shell for manual testing |

## Model Cache

Models are cached in `~/.cache/huggingface` to avoid re-downloading. The first run will download the model (~2GB for TinyLlama), subsequent runs use the cache.

## Troubleshooting

### "docker: command not found"
Install Docker Desktop for Windows: https://www.docker.com/products/docker-desktop/

### "nvidia-smi failed" or GPU not detected
1. Install NVIDIA drivers for Windows
2. Install CUDA Toolkit for WSL: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
3. Restart WSL: `wsl --shutdown` then reopen

### "container runtime not found"
Ensure NVIDIA Container Toolkit is installed in your WSL distro:
```bash
# In WSL
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Out of memory errors
Reduce batch size or use a smaller model. Edit `benchmark.py` and change:
```python
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Use a smaller model
```

### Permission denied on volume mount
Run Docker as administrator or add your user to the docker group.
