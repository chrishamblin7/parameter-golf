"""Post-run script: generates plots and config.json for an experiment subfolder."""
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_log(log_path):
    steps_train, train_losses = [], []
    steps_val, val_losses, val_bpbs = [], [], []
    final_lines = {}

    with open(log_path) as f:
        for line in f:
            m = re.match(r"step:(\d+)/\d+ train_loss:([\d.]+)", line)
            if m and "val_loss" not in line:
                steps_train.append(int(m.group(1)))
                train_losses.append(float(m.group(2)))

            m = re.match(r"step:(\d+)/\d+ val_loss:([\d.]+) val_bpb:([\d.]+)", line)
            if m:
                steps_val.append(int(m.group(1)))
                val_losses.append(float(m.group(2)))
                val_bpbs.append(float(m.group(3)))

            for key in (
                "stopping_early",
                "peak memory",
                "Serialized model int8",
                "Total submission size int8",
                "final_int8_zlib_roundtrip_exact",
                "final_int8_zlib_roundtrip ",
                "Serialized model int6",
                "Total submission size int6",
                "final_int6_roundtrip_exact",
                "final_int6_sliding_window_exact",
                "model_params",
                "world_size",
                "train_batch_tokens",
                "seed",
            ):
                if line.startswith(key) or key in line:
                    final_lines[key.strip()] = line.strip()

    return steps_train, train_losses, steps_val, val_losses, val_bpbs, final_lines


def parse_step_timing(log_path):
    """Extract (step, train_time_ms) pairs from log lines."""
    steps, times = [], []
    with open(log_path) as f:
        for line in f:
            m = re.match(r"step:(\d+)/\d+ .*train_time:(\d+)ms", line)
            if m:
                step, t = int(m.group(1)), int(m.group(2))
                if step > 0:
                    steps.append(step)
                    times.append(t)
    return steps, times


def make_lr_plot(log_path, out_path, warmdown_iters, max_wallclock_ms, title_suffix=""):
    """Reconstruct and plot the LR multiplier schedule from log timing data."""
    steps, times = parse_step_timing(log_path)
    if not steps:
        print(f"No step timing data found in {log_path}, skipping LR plot")
        return

    lr_muls = []
    for step, elapsed_ms in zip(steps, times):
        if warmdown_iters <= 0:
            lr_muls.append(1.0)
            continue
        step_ms = elapsed_ms / max(step, 1)
        warmdown_ms = warmdown_iters * step_ms
        remaining_ms = max(max_wallclock_ms - elapsed_ms, 0.0)
        if remaining_ms <= warmdown_ms:
            lr_muls.append(remaining_ms / max(warmdown_ms, 1e-9))
        else:
            lr_muls.append(1.0)

    warmdown_start = None
    for i, m in enumerate(lr_muls):
        if m < 1.0:
            warmdown_start = steps[i]
            break

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, lr_muls, color="#2ecc71", linewidth=1.5)
    if warmdown_start is not None:
        ax.axvline(x=warmdown_start, color="gray", linestyle="--", alpha=0.7,
                   label=f"warmdown start (step {warmdown_start})")
        ax.legend()
    ax.set_xlabel("Step")
    ax.set_ylabel("LR Multiplier")
    ax.set_title(f"LR Schedule{title_suffix}")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved LR plot: {out_path}")


def make_plots(log_path, out_path, title_suffix=""):
    steps_train, train_losses, steps_val, _, val_bpbs, _ = parse_log(log_path)

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 5))

    ax1.plot(steps_train, train_losses, alpha=0.5, linewidth=0.8, color="#4a90d9")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Train Loss")
    ax1.set_title(f"Train Loss{title_suffix}")
    ax1.grid(True, alpha=0.3)

    ax2.plot(steps_train, train_losses, alpha=0.5, linewidth=0.8, color="#4a90d9")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Train Loss (log scale)")
    ax2.set_title(f"Train Loss — Log Scale{title_suffix}")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3, which="both")

    ax3.plot(steps_val, val_bpbs, "o-", markersize=3, color="#e74c3c", linewidth=1.5)
    ax3.axhline(y=1.2244, color="gray", linestyle="--", alpha=0.7, label="H100 baseline (1.2244)")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Val BPB")
    ax3.set_title(f"Val BPB — Linear{title_suffix}")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(steps_val, val_bpbs, "o-", markersize=3, color="#e74c3c", linewidth=1.5)
    ax4.axhline(y=1.2244, color="gray", linestyle="--", alpha=0.7, label="H100 baseline (1.2244)")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Val BPB (log scale)")
    ax4.set_title(f"Val BPB — Log Scale{title_suffix}")
    ax4.set_yscale("log")
    ax4.legend()
    ax4.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def make_config(log_path, out_path, command, env_overrides):
    _, _, steps_val, val_losses, val_bpbs, final_lines = parse_log(log_path)

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()

    gpu_name = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().split("\n")

    import torch
    import sentencepiece

    pre_quant_bpb = val_bpbs[-1] if val_bpbs else None
    pre_quant_loss = val_losses[-1] if val_losses else None

    post_quant_bpb, post_quant_loss = None, None
    for rt_key in ("final_int8_zlib_roundtrip_exact", "final_int6_sliding_window_exact",
                    "final_int6_roundtrip_exact"):
        rt = final_lines.get(rt_key, "")
        m = re.search(r"val_loss:([\d.]+) val_bpb:([\d.]+)", rt)
        if m:
            post_quant_loss = float(m.group(1))
            post_quant_bpb = float(m.group(2))
            break

    total_steps = steps_val[-1] if steps_val else 0
    m = re.search(r"train_time:(\d+)ms", final_lines.get("stopping_early", ""))
    total_time_ms = int(m.group(1)) if m else None

    model_size_bytes = None
    quant_format = None
    for k, v in final_lines.items():
        for fmt, pattern in (("int8+zlib", r"int8\+zlib: (\d+) bytes"),
                             ("int6+lzma", r"int6\+lzma: (\d+) bytes")):
            if fmt.split("+")[0] in k:
                m2 = re.search(pattern, v)
                if m2:
                    model_size_bytes = int(m2.group(1))
                    quant_format = fmt

    params = None
    for k, v in final_lines.items():
        if "model_params" in k:
            m2 = re.search(r"model_params:(\d+)", v)
            if m2:
                params = int(m2.group(1))

    config = {
        "git_sha": git_sha,
        "command": command,
        "env_overrides": env_overrides,
        "hardware": {
            "gpus": list(set(gpu_name)),
            "gpu_count": len(gpu_name),
            "driver_version": subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=False,
            ).stdout.strip().split("\n")[0],
        },
        "software": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "sentencepiece_version": sentencepiece.__version__,
        },
        "data": {
            "data_path": env_overrides.get("DATA_PATH", "./data/datasets/fineweb10B_sp1024"),
            "tokenizer_path": env_overrides.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model"),
        },
        "model": {
            "param_count": params,
        },
        "results": {
            "total_steps": total_steps,
            "total_train_time_ms": total_time_ms,
            "pre_quant_val_loss": pre_quant_loss,
            "pre_quant_val_bpb": pre_quant_bpb,
            "post_quant_val_loss": post_quant_loss,
            "post_quant_val_bpb": post_quant_bpb,
            "quant_format": quant_format,
            "model_artifact_bytes": model_size_bytes,
        },
    }

    with open(out_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {out_path}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--log", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--command", required=True)
    p.add_argument("--env-json", required=True, help="JSON string of env overrides")
    p.add_argument("--title", default="")
    p.add_argument("--warmdown-iters", type=int, default=0,
                   help="WARMDOWN_ITERS value for LR plot reconstruction")
    p.add_argument("--max-wallclock-ms", type=float, default=0,
                   help="MAX_WALLCLOCK_SECONDS*1000 for LR plot reconstruction")
    args = p.parse_args()

    env_overrides = json.loads(args.env_json)
    make_plots(args.log, os.path.join(args.out_dir, "curves.png"), title_suffix=args.title)
    make_config(args.log, os.path.join(args.out_dir, "config.json"), args.command, env_overrides)
    if args.warmdown_iters > 0 and args.max_wallclock_ms > 0:
        make_lr_plot(args.log, os.path.join(args.out_dir, "lr_schedule.png"),
                     args.warmdown_iters, args.max_wallclock_ms, title_suffix=args.title)
