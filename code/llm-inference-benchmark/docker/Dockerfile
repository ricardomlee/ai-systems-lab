FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHON_VERSION=3.11

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3=${PYTHON_VERSION}.* \
    python3-pip \
    python3-venv \
    curl \
    git \
    build-essential \
    libssl-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY python/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Rust (for the Rust benchmark)
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:$PATH"

# Pre-download TinyLlama model (optional, speeds up first run)
# RUN python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0')"

# Copy benchmark code
COPY python/ ./python/
COPY rust/ ./rust/

# Set working directory
WORKDIR /app

# Default command
CMD ["bash"]
