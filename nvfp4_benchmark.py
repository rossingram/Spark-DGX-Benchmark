#!/usr/bin/env python

"""
nvfp4_gemm_bench.py

Experimental "NVFP4-style" GEMM benchmark for NVIDIA DGX Spark (GB10).

IMPORTANT:
    - This script does NOT use real NVFP4 Tensor Core kernels yet.
    - Today, PyTorch does not expose native FP4 (NVFP4) GEMM for Blackwell.
    - Instead, this benchmark:
        * Quantizes matrices to a symmetric 4-bit representation
        * Dequantizes back to BF16
        * Runs a BF16 GEMM via PyTorch
    - The purpose is to:
        * Provide a future-proof harness for FP4 benchmarking
        * Let you compare "FP4-style" pipelines vs pure BF16
        * Make it easy to plug in real FP4 kernels later (cuBLASLt / TensorRT-LLM / CUTLASS)

Usage (inside nvcr.io/nvidia/pytorch:25.09-py3):

    python nvfp4_gemm_bench.py
    python nvfp4_gemm_bench.py --size 8192 --iters 50
    python nvfp4_gemm_bench.py --sizes 4096 8192 12288 --iters 30
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


def quantize_fp4_symmetric(x: torch.Tensor):
    """
    Simple symmetric 4-bit quantization:

        - Target range: [-8, 7] (signed 4-bit)
        - One global scale per tensor: scale = max(|x|) / 7
        - Stored as int8 (we conceptually use only lower 4 bits)

    This is NOT the exact NVFP4 hardware encoding, but it's good enough
    for a "NVFP4-style" weight quantization pipeline benchmark.
    """
    if x.numel() == 0:
        return torch.zeros_like(x, dtype=torch.int8), torch.tensor(1.0, device=x.device, dtype=torch.float32)

    x_absmax = x.abs().max()
    if x_absmax == 0:
        scale = torch.tensor(1.0, device=x.device, dtype=torch.float32)
        q = torch.zeros_like(x, dtype=torch.int8)
        return q, scale

    scale = x_absmax / 7.0  # 4-bit signed: -8..7 -> 7 is max magnitude
    x_scaled = x / scale
    q = torch.round(x_scaled).clamp(-8, 7).to(torch.int8)
    return q, scale


def dequantize_fp4_symmetric(q: torch.Tensor, scale: torch.Tensor, dtype=torch.bfloat16):
    """
    Dequantize int8 back to float using the symmetric scale.
    """
    x = q.to(torch.float32) * scale
    return x.to(dtype)


def run_bf16_gemm(size: int, iters: int) -> float:
    """
    Baseline BF16 GEMM:

        C = A @ B
        A: [size, size], BF16
        B: [size, size], BF16

    Returns approximate TFLOPs.
    """
    print(f"\n=== BASELINE BF16 GEMM: size={size} x {size}, iters={iters} ===")

    if not torch.cuda.is_available():
        print("CUDA is not available; aborting baseline BF16 GEMM.")
        return 0.0

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    A = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    B = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    A = A.contiguous()
    B = B.contiguous()

    warmup_iters = min(20, max(5, iters // 5))
    print(f"Warmup (BF16): {warmup_iters} iters...")
    for _ in range(warmup_iters):
        _ = A @ B
    torch.cuda.synchronize()

    print("Benchmarking BF16 GEMM...")
    t0 = time.time()
    for _ in range(iters):
        _ = A @ B
    torch.cuda.synchronize()
    dt = time.time() - t0

    flops = 2.0 * (size**3) * iters
    tflops = flops / dt / 1e12

    print(f"BF16 total time: {human_time(dt)} for {iters} iters")
    print(f"BF16 approx TFLOPs: {tflops:.2f}")
    return tflops


def run_nvfp4_style_gemm(size: int, iters: int) -> float:
    """
    "NVFP4-style" GEMM:

        - Start from FP32 weights
        - Quantize both A and B to 4-bit (symmetric)
        - Dequantize back to BF16 for GEMM
        - Run GEMM in BF16 using PyTorch

    This simulates a pipeline where weights are stored in FP4
    but matmul is implemented using BF16 GEMM (until native FP4
    kernels are available in user space).

    Returns approximate "effective" TFLOPs (still measured as 2*N^3 / time).
    """
    print(f"\n=== NVFP4-STYLE GEMM (SIMULATED): size={size} x {size}, iters={iters} ===")

    if not torch.cuda.is_available():
        print("CUDA is not available; aborting NVFP4-style GEMM.")
        return 0.0

    device = torch.device("cuda")

    # Start from higher-precision weights (FP32)
    A_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)
    B_fp32 = torch.randn(size, size, device=device, dtype=torch.float32)

    # Quantize once (as if we stored weights in FP4)
    print("Quantizing A/B to 4-bit (symmetric)...")
    A_q, A_scale = quantize_fp4_symmetric(A_fp32)
    B_q, B_scale = quantize_fp4_symmetric(B_fp32)

    # Dequantize to BF16 for matmul
    A_bf16 = dequantize_fp4_symmetric(A_q, A_scale, dtype=torch.bfloat16).contiguous()
    B_bf16 = dequantize_fp4_symmetric(B_q, B_scale, dtype=torch.bfloat16).contiguous()

    # Warmup
    warmup_iters = min(20, max(5, iters // 5))
    print(f"Warmup (NVFP4-style BF16 GEMM): {warmup_iters} iters...")
    for _ in range(warmup_iters):
        _ = A_bf16 @ B_bf16
    torch.cuda.synchronize()

    print("Benchmarking NVFP4-style GEMM (BF16 matmul on dequantized FP4 weights)...")
    t0 = time.time()
    for _ in range(iters):
        _ = A_bf16 @ B_bf16
    torch.cuda.synchronize()
    dt = time.time() - t0

    flops = 2.0 * (size**3) * iters
    tflops = flops / dt / 1e12

    print(f"NVFP4-style total time: {human_time(dt)} for {iters} iters")
    print(f"NVFP4-style approx TFLOPs (BF16 matmul): {tflops:.2f}")

    return tflops


def print_env_info():
    print("=== ENVIRONMENT / GPU INFO (NVFP4-STYLE GEMM BENCH) ===")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available : {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("No CUDA device detected.")
        print()
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

    print("\nNOTE:")
    print("  - This script does NOT use real NVFP4 Tensor Core kernels yet.")
    print("  - It simulates a 4-bit weight pipeline and uses BF16 GEMM for matmul.")
    print("  - Once PyTorch / TensorRT-LLM exposes FP4 GEMM for Blackwell,")
    print("    this harness can be updated to call the true FP4 path.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Experimental NVFP4-style GEMM benchmark for NVIDIA DGX Spark (GB10)."
    )
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
        default=30,
        help="Number of timed iterations per size.",
    )
    args = parser.parse_args()

    print_env_info()

    sizes = args.sizes if args.sizes else [args.size]
    results = {}

    for s in sizes:
        print(f"\n================ SIZE {s} ================\n")
        try:
            bf16_tflops = run_bf16_gemm(size=s, iters=args.iters)
            fp4_tflops = run_nvfp4_style_gemm(size=s, iters=args.iters)
            results[s] = (bf16_tflops, fp4_tflops)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"OOM at size {s}; skipping.")
                results[s] = (0.0, 0.0)
            else:
                print(f"RuntimeError at size {s}: {e}")
                results[s] = (0.0, 0.0)

    print("\n=== SUMMARY (NVFP4-STYLE vs BF16 GEMM) ===")
    print("Size\tBF16 TFLOPs\tNVFP4-style TFLOPs")
    for s in sizes:
        bf16, fp4 = results.get(s, (0.0, 0.0))
        print(f"{s}\t{bf16:.2f}\t\t{fp4:.2f}")

    print("\nNOTE:")
    print("  - NVFP4-style TFLOPs here are still from BF16 matmul,")
    print("    just with 4-bit weight quantization and dequantization.")
    print("  - This is a pipeline / harness benchmark, NOT true FP4 Tensor Core math.")
    print("  - Once NVIDIA exposes NVFP4 GEMM kernels in user space,")
    print("    this script can be upgraded to measure real FP4 Tensor Core throughput.\n")


if __name__ == "__main__":
    main()
