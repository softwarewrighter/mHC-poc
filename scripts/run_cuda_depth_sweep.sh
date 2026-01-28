#!/usr/bin/env bash
# Run full CUDA depth sweep: 3 depths x 3 variants = 9 training jobs
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Ensure we're in the cuda venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating cuda/.venv..."
    source cuda/.venv/bin/activate
fi

# Generate variant configs
echo "=== Generating variant configs ==="
python scripts/gen_cuda_variant_configs.py

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_BASE="runs/cuda_sweep_${TIMESTAMP}"
mkdir -p "$RUN_BASE"

echo "=== Starting CUDA depth sweep ==="
echo "Output: $RUN_BASE"
echo ""

# Run all 9 combinations
for depth in 12 24 48; do
    for variant in baseline hc mhc; do
        config="cuda/configs/tiny_${depth}l_${variant}.yaml"
        out_dir="${RUN_BASE}/${depth}l_${variant}"

        echo "--- Running: ${depth}L ${variant} ---"
        python -m cuda.src.train --config "$config" --out "$out_dir"
        echo ""
    done
done

echo "=== Depth sweep complete ==="
echo "Results in: $RUN_BASE"
echo ""
echo "To compare results, check metrics.jsonl in each subdirectory."
