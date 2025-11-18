#!/usr/bin/env python

"""
bf16_gemm_bench.py

Standalone BF16 GEMM microbenchmark for NVIDIA DGX Spark (GB10).

This script measures approximate BF16 TFLOPs for large square GEMMs
using PyTorch's matmul / @ operator. It is intended as a "max effort"
user-space BF16 test, not a theoretical peak measurement.

Usage (inside nvcr.io/nvidia/pytorch:25.01-py3):

    python bf16_gemm_bench.py
    python bf16_gemm_bench.py --size 8192 --iters 50
    python bf16_gemm_bench.py --sizes 4096 8192 12288 --iters 30
"""

import argparse
import time

import torch


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def run_bf16_gemm(size: int, iters: int) -> float:
    """
    Run a BF16 GEMM benchmark for a square GEMM of shape:

        C = A @ B
        A: [size, size] (BF16)
        B: [size, size] (BF16)

    Returns approximate TFLOPs.
    """
    print(f"\n=== BF16 GEMM: size={size} x {size}, iters={iters} ===")

    if not torch.cuda.is_available():
        print("CUDA is not available; aborting.")
        return 0.0

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    # Allocate BF16 matrices
    x = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    x = x.contiguous()

    # Warmup (helps JIT / kernel selection / caches)
    warmup_iters = min(20, max(5, iters // 5))
    print(f"Warmup: {warmup_iters} iters...")
    for _ in range(warmup_iters):
        y = x @ x
    torch.cuda.synchronize()

    print("Benchmarking...")
    t0 = time.time()
    for _ in range(iters):
        y = x @ x
    torch.cuda.synchronize()
    dt = time.time() - t0

    # GEMM FLOPs: 2 * M * N * K * iters (here all equal: size)
    flops = 2.0 * (size**3) * iters
    tflops = flops / dt / 1e12

    print(f"Total time: {human_time(dt)} for {iters} iters")
    print(f"Approx BF16 TFLOPs: {tflops:.2f}")

    return tflops


def print_env_info():
    print("=== ENVIRONMENT / GPU INFO (BF16 GEMM BENCH) ===")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device detected.")
        return

    device = torch.device("cuda")
    print(f"CUDA device    : {torch.cuda.get_device_name(device)}")
    cap = torch.cuda.get_device_capability(device)
    print(f"Compute cap    : sm_{cap[0]}{cap[1]}")

    props = torch.cuda.get_device_properties(device)
    total_gb = props.total_memory / (1024**3)
    print(f"Total VRAM     : {total_gb:.2f} GB")

    try:
        free, total = torch.cuda.mem_get_info()
        print(f"mem_get_info   : free={free / (1024**3):.2f} GB, total={total / (1024**3):.2f} GB")
    except Exception:
        pass

    print()


def main():
    parser = argparse.ArgumentParser(description="BF16 GEMM benchmark for NVIDIA DGX Spark (GB10).")
    parser.add_argument(
        "--size",
        type=int,
        default=8192,
        help="Single square GEMM size to benchmark (M=N=K=size). Ignored if --sizes is provided.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="*",
        default=None,
        help="Optional list of sizes to benchmark. Example: --sizes 4096 8192 12288",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=50,
        help="Number of timed iterations per size.",
    )

    args = parser.parse_args()

    print_env_info()

    sizes = args.sizes if args.sizes else [args.size]
    results = {}

    for s in sizes:
        try:
            tflops = run_bf16_gemm(size=s, iters=args.iters)
            results[s] = tflops
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at size {s}; skipping.")
                results[s] = 0.0
            else:
                print(f"RuntimeError at size {s}: {e}")
                results[s] = 0.0

    print("\n=== SUMMARY (BF16 GEMM) ===")
    print("Size\tTFLOPs")
    for s in sizes:
        val = results.get(s, 0.0)
        print(f"{s}\t{val:.2f}")


if __name__ == "__main__":
    main()
