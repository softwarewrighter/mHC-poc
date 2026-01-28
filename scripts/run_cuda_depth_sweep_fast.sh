#!/usr/bin/env bash
# Fast CUDA depth sweep: 3 depths x 3 variants = 9 training jobs (200 steps each)
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Ensure we're in the cuda venv
if [[ -z "$VIRTUAL_ENV" ]]; then
    echo "Activating cuda/.venv..."
    source cuda/.venv/bin/activate
fi

# Create timestamped run directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_BASE="runs/cuda_sweep_${TIMESTAMP}"
mkdir -p "$RUN_BASE"

echo "=== Starting FAST CUDA depth sweep (200 steps each) ==="
echo "Output: $RUN_BASE"
echo ""

# Run all 9 combinations with reduced steps
for depth in 12 24 48; do
    for variant in baseline hc mhc; do
        config="cuda/configs/tiny_${depth}l_${variant}.yaml"
        out_dir="${RUN_BASE}/${depth}l_${variant}"

        echo "--- Running: ${depth}L ${variant} ---"
        # Override steps to 200 for faster results
        python -c "
import yaml
import sys
with open('$config', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['steps'] = 200
cfg['log_every'] = 20
with open('/tmp/fast_config.yaml', 'w') as f:
    yaml.safe_dump(cfg, f)
"
        python -m cuda.src.train --config /tmp/fast_config.yaml --out "$out_dir"
        echo ""
    done
done

echo "=== Fast depth sweep complete ==="
echo "Results in: $RUN_BASE"
