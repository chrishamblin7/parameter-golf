"""Sweep random sparsity levels and measure int8+zlib compressed model size.

Loads a trained model checkpoint, progressively zeros out weights at increasing
sparsity fractions (0% to 90% in 5% steps), runs the same int8 quantization +
zlib compression pipeline used in train_gpt.py, and plots compressed size vs
sparsity.
"""

import io
import os
import sys
import zlib
from pathlib import Path

import torch
from torch import Tensor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Quantization helpers (copied from train_gpt.py to avoid CUDA/DDP deps)
# ---------------------------------------------------------------------------

CONTROL_TENSOR_NAME_PATTERNS = (
    "attn_scale", "attn_scales", "mlp_scale", "mlp_scales",
    "resid_mix", "resid_mixes", "q_gain", "skip_weight", "skip_weights",
)
INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = CONTROL_TENSOR_NAME_PATTERNS
INT8_KEEP_FLOAT_MAX_NUMEL = 65_536
INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16
INT8_PER_ROW_SCALE_DTYPE = torch.float16
INT8_CLIP_PERCENTILE = 99.99984
INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0


def tensor_nbytes(t: Tensor) -> int:
    return int(t.numel()) * int(t.element_size())


def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:
    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):
        return t.float().contiguous()
    if t.dtype in {torch.float32, torch.bfloat16}:
        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix("torch.")
        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()
    return t


def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:
    t32 = t.float()
    if t32.ndim == 2:
        clip_abs = (
            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)
            if t32.numel()
            else torch.empty((t32.shape[0],), dtype=torch.float32)
        )
        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])
        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)
        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()
        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()

    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0
    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)
    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()
    return q, scale


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    quantized: dict[str, Tensor] = {}
    scales: dict[str, Tensor] = {}
    dtypes: dict[str, str] = {}
    passthrough: dict[str, Tensor] = {}
    passthrough_orig_dtypes: dict[str, str] = {}
    qmeta: dict[str, dict[str, object]] = {}
    stats = dict.fromkeys(
        ("param_count", "num_tensors", "num_float_tensors", "num_nonfloat_tensors",
         "baseline_tensor_bytes", "int8_payload_bytes"),
        0,
    )

    for name, tensor in state_dict.items():
        t = tensor.detach().to("cpu").contiguous()
        stats["param_count"] += int(t.numel())
        stats["num_tensors"] += 1
        stats["baseline_tensor_bytes"] += tensor_nbytes(t)

        if not t.is_floating_point():
            stats["num_nonfloat_tensors"] += 1
            passthrough[name] = t
            stats["int8_payload_bytes"] += tensor_nbytes(t)
            continue

        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:
            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)
            passthrough[name] = kept
            stats["int8_payload_bytes"] += tensor_nbytes(kept)
            continue

        stats["num_float_tensors"] += 1
        q, s = quantize_float_tensor(t)
        if s.ndim > 0:
            qmeta[name] = {"scheme": "per_row", "axis": 0}
        quantized[name] = q
        scales[name] = s
        dtypes[name] = str(t.dtype).removeprefix("torch.")
        stats["int8_payload_bytes"] += tensor_nbytes(q) + tensor_nbytes(s)

    obj: dict[str, object] = {
        "__quant_format__": "int8_clean_per_row_v1",
        "quantized": quantized,
        "scales": scales,
        "dtypes": dtypes,
        "passthrough": passthrough,
    }
    if qmeta:
        obj["qmeta"] = qmeta
    if passthrough_orig_dtypes:
        obj["passthrough_orig_dtypes"] = passthrough_orig_dtypes
    return obj, stats


# ---------------------------------------------------------------------------
# Sparsification
# ---------------------------------------------------------------------------

def sparsify_state_dict(state_dict: dict[str, Tensor], frac_zeroed: float, seed: int = 42) -> dict[str, Tensor]:
    """Return a copy of *state_dict* with *frac_zeroed* of eligible weights set to zero.

    Only large float tensors (numel > INT8_KEEP_FLOAT_MAX_NUMEL) are sparsified,
    matching the set of tensors that would be int8-quantized.
    """
    rng = torch.Generator().manual_seed(seed)
    out = {}
    for name, tensor in state_dict.items():
        t = tensor.detach().clone()
        if t.is_floating_point() and t.numel() > INT8_KEEP_FLOAT_MAX_NUMEL:
            mask = torch.rand(t.shape, generator=rng) >= frac_zeroed
            t.mul_(mask)
        out[name] = t
    return out


# ---------------------------------------------------------------------------
# Compression (identical to train_gpt.py pipeline)
# ---------------------------------------------------------------------------

def compress_state_dict(state_dict: dict[str, Tensor]) -> tuple[int, int]:
    """Quantize int8 -> torch.save -> zlib compress. Returns (raw_bytes, compressed_bytes)."""
    quant_obj, _ = quantize_state_dict_int8(state_dict)
    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    return len(raw), len(compressed)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    model_path = Path(__file__).resolve().parent.parent / "experiments" / "balanced_2x_1260s" / "final_model.pt"
    if not model_path.exists():
        print(f"Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(__file__).resolve().parent
    print(f"Loading model from {model_path} ...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    sparsity_levels = [round(x * 0.05, 2) for x in range(19)]  # 0.00 .. 0.90
    results = []

    for frac in sparsity_levels:
        print(f"  sparsity={frac:.0%} ... ", end="", flush=True)
        sd = sparsify_state_dict(state_dict, frac) if frac > 0 else state_dict
        raw_bytes, compressed_bytes = compress_state_dict(sd)
        results.append((frac, raw_bytes, compressed_bytes))
        print(f"raw={raw_bytes / 1e6:.2f} MB  compressed={compressed_bytes / 1e6:.2f} MB")

    # Summary table
    print()
    print(f"{'Sparsity':>10}  {'Raw (MB)':>10}  {'Compressed (MB)':>16}")
    print("-" * 42)
    for frac, raw_b, comp_b in results:
        print(f"{frac:>9.0%}  {raw_b / 1e6:>10.2f}  {comp_b / 1e6:>16.2f}")

    # Plot
    fracs = [r[0] for r in results]
    compressed_mb = [r[2] / 1e6 for r in results]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(fracs, compressed_mb, "o-", color="#2c7fb8", linewidth=2, markersize=6)
    ax.set_xlabel("Fraction of weights zeroed", fontsize=12)
    ax.set_ylabel("Compressed model size (MB)", fontsize=12)
    ax.set_title("Sparsity vs int8+zlib Compressed Size", fontsize=14)
    ax.set_xlim(-0.02, 0.92)
    ax.grid(True, alpha=0.3)

    for frac, _, comp_b in [results[0], results[-1]]:
        ax.annotate(
            f"{comp_b / 1e6:.1f} MB",
            xy=(frac, comp_b / 1e6),
            textcoords="offset points",
            xytext=(0, 12),
            ha="center",
            fontsize=9,
        )

    fig.tight_layout()
    plot_path = out_dir / "sparsity_vs_size.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {plot_path}")


if __name__ == "__main__":
    main()
