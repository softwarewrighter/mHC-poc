from __future__ import annotations
import os, yaml, copy

BASES = ["tiny_12l.yaml", "tiny_24l.yaml", "tiny_48l.yaml"]
VARIANTS = ["baseline", "hc", "mhc"]

def main():
    for base in BASES:
        with open(os.path.join("mlx", "configs", base), "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for v in VARIANTS:
            c = copy.deepcopy(cfg)
            c["variant"] = v
            out = base.replace(".yaml", f"_{v}.yaml")
            with open(os.path.join("mlx", "configs", out), "w", encoding="utf-8") as f:
                yaml.safe_dump(c, f, sort_keys=False)
            print("wrote", out)

if __name__ == "__main__":
    main()
