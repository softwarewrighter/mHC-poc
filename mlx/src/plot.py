from __future__ import annotations
import os, json
import matplotlib.pyplot as plt

def read_metrics(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def plot_run(out_dir: str):
    mpath = os.path.join(out_dir, "metrics.jsonl")
    rows = read_metrics(mpath)
    if not rows:
        return

    steps = [r["step"] for r in rows]
    loss  = [r["loss"] for r in rows]
    gnorm = [r["grad_norm"] for r in rows]
    nans  = [r["nan_or_inf_total"] for r in rows]
    gain  = [r["gain_proxy_log10_max_entry"] for r in rows]

    pdir = os.path.join(out_dir, "plots")
    os.makedirs(pdir, exist_ok=True)

    plt.figure()
    plt.plot(steps, loss)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(os.path.join(pdir, "loss.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, gnorm)
    plt.xlabel("step")
    plt.ylabel("global grad norm")
    plt.title("Gradient norm")
    plt.tight_layout()
    plt.savefig(os.path.join(pdir, "grad_norm.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, nans)
    plt.xlabel("step")
    plt.ylabel("NaN/Inf events (cumulative)")
    plt.title("NaN/Inf events")
    plt.tight_layout()
    plt.savefig(os.path.join(pdir, "nan_inf_events.png"))
    plt.close()

    plt.figure()
    plt.plot(steps, gain)
    plt.xlabel("step")
    plt.ylabel("log10(max |prod(H_res)| entry))")
    plt.title("Gain proxy (residual mixing)")
    plt.tight_layout()
    plt.savefig(os.path.join(pdir, "gain_proxy.png"))
    plt.close()
