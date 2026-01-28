"""Generate variant configs for CUDA depth sweep."""
from __future__ import annotations
import os
import yaml
import copy

BASES = ["tiny_12l.yaml", "tiny_24l.yaml", "tiny_48l.yaml"]
VARIANTS = ["baseline", "hc", "mhc"]


def main():
    config_dir = os.path.join("cuda", "configs")

    for base in BASES:
        base_path = os.path.join(config_dir, base)
        if not os.path.exists(base_path):
            print(f"Skipping {base} (not found)")
            continue

        with open(base_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        for v in VARIANTS:
            c = copy.deepcopy(cfg)
            c["variant"] = v
            out = base.replace(".yaml", f"_{v}.yaml")
            out_path = os.path.join(config_dir, out)
            with open(out_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
