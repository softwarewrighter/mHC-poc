#!/usr/bin/env bash
set -euo pipefail

# Generate per-variant configs (baseline/hc/mhc) from the baseline templates
python scripts/gen_variant_configs.py >/dev/null

RUNS_DIR="runs"
mkdir -p "$RUNS_DIR"

timestamp="$(date +"%Y%m%d_%H%M%S")"

depths=("12" "24" "48")
variants=("baseline" "hc" "mhc")

for d in "${depths[@]}"; do
  for v in "${variants[@]}"; do
    cfg="mlx/configs/tiny_${d}l_${v}.yaml"
    out="${RUNS_DIR}/${timestamp}_${v}_${d}l"
    echo "==> running ${v} depth=${d}  out=${out}"
    python -m mlx.src.train --config "${cfg}" --out "${out}"
  done
done

echo
echo "Done. See ${RUNS_DIR}/${timestamp}_*/plots/*.png"
