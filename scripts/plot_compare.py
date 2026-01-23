from __future__ import annotations
import os, glob, json
import matplotlib.pyplot as plt

def read(path):
    rows=[]
    with open(path,"r",encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return rows

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default="runs/*_baseline_48l")
    ap.add_argument("--out", default="runs/compare.png")
    args=ap.parse_args()

    run_dirs=sorted(glob.glob(args.runs_glob))
    if not run_dirs:
        raise SystemExit("No runs matched glob")

    plt.figure()
    for rd in run_dirs:
        mpath=os.path.join(rd,"metrics.jsonl")
        rows=read(mpath)
        steps=[r["step"] for r in rows]
        loss=[r["loss"] for r in rows]
        plt.plot(steps, loss, label=os.path.basename(rd))
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Compare loss curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(args.out)
    print("wrote", args.out)

if __name__=="__main__":
    main()
