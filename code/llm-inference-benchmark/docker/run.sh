#!/bin/bash
# Run LLM inference benchmarks with Docker
# Usage: ./run.sh [python|rust|both|clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

case "${1:-both}" in
    python)
        echo "Running Python benchmark..."
        docker compose run --rm llm-benchmark python python/benchmark.py --max-tokens 100
        ;;
    rust)
        echo "Running Rust benchmark..."
        docker compose run --rm llm-benchmark bash -c "
            cd /app/rust && \
            cargo build --release && \
            cargo run --release -- --max-tokens 100
        "
        ;;
    both)
        echo "Running both benchmarks..."
        echo ""
        echo "=== Python Benchmark ==="
        docker compose run --rm llm-benchmark python python/benchmark.py --max-tokens 100
        echo ""
        echo "=== Rust Benchmark ==="
        docker compose run --rm llm-benchmark bash -c "
            cd /app/rust && \
            cargo build --release && \
            cargo run --release -- --max-tokens 100
        "
        ;;
    clean)
        echo "Cleaning up Docker resources..."
        docker compose down
        docker rmi llm-benchmark-llm-benchmark 2>/dev/null || true
        echo "Done."
        ;;
    bash)
        echo "Starting interactive shell..."
        docker compose run --rm llm-benchmark bash
        ;;
    *)
        echo "Usage: $0 [python|rust|both|clean|bash]"
        echo ""
        echo "Commands:"
        echo "  python  - Run Python benchmark only"
        echo "  rust    - Run Rust benchmark only"
        echo "  both    - Run both benchmarks (default)"
        echo "  clean   - Remove Docker containers and images"
        echo "  bash    - Start interactive shell inside container"
        exit 1
        ;;
esac
