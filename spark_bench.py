#!/usr/bin/env python

import math
import os
import subprocess
import sys
import time

results = {}


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

def has_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False


def log_section(title: str):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)
    print()


def log_sub(title: str):
    print()
    print(f"--- {title} ---")


def human_time(seconds: float) -> str:
    if seconds < 1e-3:
        return f"{seconds * 1e6:.1f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.1f} ms"
    return f"{seconds:.2f} s"


def record(key: str, value):
    results[key] = value


def fmt(x, default="—"):
    if x is None:
        return default
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if abs(x) >= 100:
            return f"{x:.0f}"
        elif abs(x) >= 10:
            return f"{x:.1f}"
        else:
            return f"{x:.2f}"
    return str(x)


# ------------------------------------------------------------------------
# 0. ENVIRONMENT / GPU INFO
# ------------------------------------------------------------------------

def bench_env():
    log_section("0. ENVIRONMENT / GPU INFO")

    if not has_module("torch"):
        print("PyTorch not installed; cannot run GPU benchmarks.")
        return

    import torch

    # Be a bit aggressive about matmul kernels
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    print("Importing torch...")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        dev = torch.device("cuda")
        print("CUDA device name:", torch.cuda.get_device_name(dev))
        cap = torch.cuda.get_device_capability(dev)
        print("Device capability:", cap)
        props = torch.cuda.get_device_properties(dev)
        total_vram_gb = props.total_memory / (1024**3)
        print(f"Total VRAM (props): {total_vram_gb:.2f} GB")

        try:
            free, total = torch.cuda.mem_get_info()
            print(f"Total VRAM (mem_get_info): {total / (1024**3):.2f} GB")
            print(f"Free VRAM  (mem_get_info): {free / (1024**3):.2f} GB")
        except Exception as e:
            print("mem_get_info not available:", e)

    print()
    print("[nvidia-smi]")
    try:
        out = subprocess.run(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        print(out.stdout)
    except FileNotFoundError:
        print("nvidia-smi not found in this environment.")


# ------------------------------------------------------------------------
# 1. GEMM / TENSOR CORE TFLOPs
# ------------------------------------------------------------------------

def bench_matmul(size=4096, iters=40, key=None):
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available; skipping GEMM.")
        return

    log_sub(f"GEMM throughput at {size}x{size}, {iters} iters")
    x = torch.randn(size, size, device="cuda", dtype=torch.float16)
    x = x.contiguous()

    # Warmup
    for _ in range(10):
        y = x @ x
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        y = x @ x
    torch.cuda.synchronize()
    dt = time.time() - t0
    tflops = (2 * size**3 * iters) / dt / 1e12
    print(f"Time: {human_time(dt)}  |  Approx TFLOPs: {tflops:.2f}")
    if key:
        record(key, tflops)


def bench_gemm():
    log_section("1. GEMM / TENSOR CORE TFLOPs")

    if not has_module("torch"):
        print("PyTorch not installed; skipping GEMM.")
        return

    bench_matmul(size=4096, iters=40, key="gemm_4096_tflops")
    bench_matmul(size=6144, iters=20, key="gemm_6144_tflops")
    bench_matmul(size=8192, iters=10, key="gemm_8192_tflops")


# ------------------------------------------------------------------------
# 2. MEMORY BANDWIDTH
# ------------------------------------------------------------------------

def memory_bw(size_gb=2, key=None):
    import torch

    if not torch.cuda.is_available():
        print("CUDA not available; skipping memory bandwidth test.")
        return

    log_sub(f"Memory bandwidth with ~{size_gb} GB tensor copy")
    num_elems = int(size_gb * 1e9 / 2)  # FP16 = 2 bytes

    a = torch.empty(num_elems, device="cuda", dtype=torch.float16)
    b = torch.empty_like(a)
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        b.copy_(a, non_blocking=True)
    torch.cuda.synchronize()

    iters = 10
    t0 = time.time()
    for _ in range(iters):
        b.copy_(a, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.time() - t0

    total_gb = size_gb * iters
    bw = total_gb / dt
    print(f"Time: {human_time(dt)}  |  Approx BW: {bw:.2f} GB/s")
    if key:
        record(key, bw)


def bench_mem():
    log_section("2. MEMORY BANDWIDTH")

    if not has_module("torch"):
        print("PyTorch not installed; skipping.")
        return

    memory_bw(size_gb=2, key="mem_bw_2gb")
    memory_bw(size_gb=4, key="mem_bw_4gb")


# ------------------------------------------------------------------------
# 3. CUDA KERNEL LAUNCH LATENCY
# ------------------------------------------------------------------------

def bench_kernel_latency():
    log_section("3. CUDA KERNEL LAUNCH LATENCY")

    if not has_module("torch"):
        print("PyTorch not installed; skipping kernel latency.")
        return

    import torch

    if not torch.cuda.is_available():
        print("CUDA not available; skipping kernel latency.")
        return

    x = torch.randn(1, device="cuda")
    iters = 20000

    log_sub("Kernel launch latency (tiny add kernel)")
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        y = x + 1
    torch.cuda.synchronize()
    dt = time.time() - t0
    avg_us = dt / iters * 1e6
    print(f"Average launch latency: {avg_us:.2f} µs over {iters} iters")
    record("kernel_latency_us", avg_us)


# ------------------------------------------------------------------------
# 4. STABLE DIFFUSION 1.5
# ------------------------------------------------------------------------

def run_sd15_bench():
    if not has_module("diffusers"):
        log_section("4. STABLE DIFFUSION 1.5")
        print("diffusers not installed; skipping SD 1.5.")
        return

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        log_section("4. STABLE DIFFUSION 1.5")
        print("CUDA not available; skipping SD 1.5.")
        return

    log_section("4. STABLE DIFFUSION 1.5")
    model_id = "runwayml/stable-diffusion-v1-5"
    prompt = "a photo of a small futuristic robot on a desk"
    steps = 30

    print(f"Loading pipeline: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Running SD 1.5 benchmark...")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = pipe(prompt, num_inference_steps=steps).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = steps / dt
    print(f"Total time: {dt:.2f} s  |  Steps/sec: {sps:.2f}")
    record("sd15_steps_sec", sps)


# ------------------------------------------------------------------------
# 5. SDXL
# ------------------------------------------------------------------------

def run_sdxl_bench():
    if not has_module("diffusers"):
        log_section("5. SDXL")
        print("diffusers not installed; skipping SDXL.")
        return

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        log_section("5. SDXL")
        print("CUDA not available; skipping SDXL.")
        return

    log_section("5. SDXL")
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    prompt = "a highly detailed photograph of a tiny robot on a wooden desk"
    steps = 30

    print(f"Loading pipeline: {model_id}")
    pipe = DiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
    ).to("cuda")

    print("Running SDXL benchmark...")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = pipe(prompt, num_inference_steps=steps).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0
    sps = steps / dt
    print(f"Total time: {dt:.2f} s  |  Steps/sec: {sps:.2f}")
    record("sdxl_steps_sec", sps)


# ------------------------------------------------------------------------
# 6. SDXL TURBO (fast image benchmark, optional)
# ------------------------------------------------------------------------

def run_sdxl_turbo_bench():
    if not has_module("diffusers"):
        log_section("6. SDXL TURBO")
        print("diffusers not installed; skipping SDXL Turbo.")
        return

    import torch
    from diffusers import DiffusionPipeline

    if not torch.cuda.is_available():
        log_section("6. SDXL TURBO")
        print("CUDA not available; skipping SDXL Turbo.")
        return

    log_section("6. SDXL TURBO")
    model_id = "stabilityai/sdxl-turbo"
    prompt = "hello world on the NVIDIA DGX Spark"

    print(f"Loading pipeline: {model_id}")
    try:
        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
        ).to("cuda")
    except Exception as e:
        print("Failed to load SDXL Turbo; skipping.")
        print(e)
        return

    print("Running SDXL Turbo benchmark...")
    torch.cuda.synchronize()
    t0 = time.time()
    _ = pipe(prompt, num_inference_steps=4).images[0]
    torch.cuda.synchronize()
    dt = time.time() - t0

    print(f"SDXL Turbo time: {dt:.2f} s")
    record("sdxl_turbo_seconds", dt)


# ------------------------------------------------------------------------
# 7. LLM THROUGHPUT (PyTorch, open model)
# ------------------------------------------------------------------------

def run_llm_bench():
    if not has_module("transformers"):
        log_section("7. LLM THROUGHPUT (PyTorch)")
        print("transformers not installed; skipping LLM bench.")
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    if not torch.cuda.is_available():
        log_section("7. LLM THROUGHPUT (PyTorch)")
        print("CUDA not available; skipping LLM bench.")
        return

    log_section("7. LLM THROUGHPUT (PyTorch)")

    # Use an open model so no HF token is required.
    # You can change this to Llama 3 etc. if you want, but then you'll need HF auth.
    model_id = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    prompt = "Hello, this is a speed benchmark on the NVIDIA DGX Spark."

    print(f"Loading model: {model_id}")
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cuda",
        )
    except Exception as e:
        print("Failed to load LLM model; skipping LLM benchmark.")
        print(e)
        return

    inputs = tok(prompt, return_tensors="pt").to("cuda")

    # Warmup
    _ = model.generate(**inputs, max_new_tokens=10)

    new_tokens = 200
    torch.cuda.synchronize()
    t0 = time.time()
    _ = model.generate(**inputs, max_new_tokens=new_tokens)
    torch.cuda.synchronize()
    dt = time.time() - t0

    tps = new_tokens / dt
    print(f"Generated {new_tokens} tokens in {dt:.2f} s  |  {tps:.1f} tokens/sec")
    record("llm_tokens_sec", tps)


# ------------------------------------------------------------------------
# 8. VRAM / UNIFIED MEMORY PROBE
# ------------------------------------------------------------------------

def run_vram_bench():
    if not has_module("torch"):
        log_section("8. VRAM / UNIFIED MEMORY PROBE")
        print("PyTorch not installed; skipping.")
        return

    import torch

    if not torch.cuda.is_available():
        log_section("8. VRAM / UNIFIED MEMORY PROBE")
        print("CUDA not available; skipping.")
        return

    log_section("8. VRAM / UNIFIED MEMORY PROBE")

    free, total = torch.cuda.mem_get_info()
    total_gb = total / (1024**3)
    print(f"Reported total CUDA-visible memory: {total_gb:.2f} GB")

    # We'll binary-search the max single allocation in GB,
    # but cap at something reasonable (e.g., 96 GB) for Spark.
    low = 1.0
    high = min(96.0, total_gb * 0.9)
    best = 0.0

    print("Probing max safe single allocation size...")
    while high - low > 0.5:
        mid = (low + high) / 2.0
        num_elems = int(mid * 1e9 / 2)  # FP16

        try:
            x = torch.empty(num_elems, device="cuda", dtype=torch.float16)
            torch.cuda.synchronize()
            del x
            torch.cuda.empty_cache()
            best = mid
            low = mid
            print(f"  OK at ~{mid:.1f} GB")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at ~{mid:.1f} GB")
                high = mid
                torch.cuda.empty_cache()
            else:
                print("  RuntimeError during allocation; stopping probe.")
                print(e)
                break
        except Exception as e:
            print("  Unexpected error during allocation; stopping probe.")
            print(e)
            break

    print(f"Max tested safe single allocation: {best:.1f} GB")
    record("vram_max_gb", best)


# ------------------------------------------------------------------------
# FINAL SUMMARY
# ------------------------------------------------------------------------

def print_final_summary():
    log_section("FINAL SUMMARY — SPARK vs OTHER GPUS")

    # Pull values with defaults
    gemm_4096 = results.get("gemm_4096_tflops")
    gemm_6144 = results.get("gemm_6144_tflops")
    gemm_8192 = results.get("gemm_8192_tflops")
    bw_2g = results.get("mem_bw_2gb")
    bw_4g = results.get("mem_bw_4gb")
    lat_us = results.get("kernel_latency_us")
    sd15 = results.get("sd15_steps_sec")
    sdxl = results.get("sdxl_steps_sec")
    sdxl_turbo = results.get("sdxl_turbo_seconds")
    llm_tps = results.get("llm_tokens_sec")
    vram_max = results.get("vram_max_gb")

    # ---------------------------------------------------------------------
    # 1) Microbenchmarks vs 4090 / H100 (same style as before)
    # ---------------------------------------------------------------------
    print("""
RAW COMPUTE (TFLOPs, FP16/BF16 — Measured vs Rough 4090/H100 Targets)
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    GEMM 4096 size       {fmt(gemm_4096).rjust(8)}     330.00        1000.00")
    print(f"    GEMM 6144 size       {fmt(gemm_6144).rjust(8)}     330.00        1000.00")
    print(f"    GEMM 8192 size       {fmt(gemm_8192).rjust(8)}     330.00        1000.00")

    print("""
MEMORY BANDWIDTH (GB/s — Measured vs Approx)
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    2GB copy              {fmt(bw_2g).rjust(8)}     ~700          ~2000")
    print(f"    4GB copy              {fmt(bw_4g).rjust(8)}     ~700          ~2000")

    print("""
KERNEL LAUNCH LATENCY (Microseconds)
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    Tiny kernel           {fmt(lat_us).rjust(8)}       ~15            ~5")

    print("""
IMAGE MODELS
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    SD1.5 steps/sec       {fmt(sd15).rjust(8)}       8–12         15–20")
    print(f"    SDXL steps/sec        {fmt(sdxl).rjust(8)}       2–3           6–8")
    print(f"    SDXL Turbo seconds    {fmt(sdxl_turbo).rjust(8)}     ~0.3s         ~0.15s")

    print("""
LLM THROUGHPUT (tokens/sec — Small Open Model)
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    SmolLM2 1.7B          {fmt(llm_tps).rjust(8)}     150–300       800–1200")

    print("""
VRAM USABLE RANGE (GB — Single Allocation Probe)
    ------------------------------------------------------------------------
    Test                       Spark        RTX 4090       H100
    ------------------------------------------------------------------------""")
    print(f"    Max safe VRAM         {fmt(vram_max).rjust(8)}        18–20         70–80")

    # ---------------------------------------------------------------------
    # 2) Reference GPU Landscape — spec-level, approximate
    # ---------------------------------------------------------------------
    print("""
REFERENCE GPU LANDSCAPE (Approx FP16 Tensor TFLOPs / Memory BW / VRAM)
    ------------------------------------------------------------------------
    GPU               FP16 Tensor TFLOPs    Mem BW (GB/s)      VRAM / Memory
    ------------------------------------------------------------------------
    Spark GB10        ~   12                ~    30            128 GB unified
    L40S              ~  730                ~   864             48 GB GDDR6
    H200              ~ 1980                ~  4800            141 GB HBM3e
    GH200 (NVL2)      ~ 2000+               up to 10000      up to 288 GB HBM
    RTX 4090          ~  330                ~  1000             24 GB GDDR6X
    H100              ~ 1000                ~  3300              80 GB HBM3
    ------------------------------------------------------------------------

Notes:
- Spark GB10 is a **memory-centric**, low-power Blackwell GPU designed for large
  models and unified memory, not raw training throughput.
- L40S is an inference-optimized Ada Lovelace data center GPU (~733 FP16 TFLOPs,
  ~864 GB/s bandwidth, 48 GB GDDR6).
- H200 is Hopper with 141 GB HBM3e at ~4.8 TB/s, tuned for huge LLMs.
- GH200 NVL2 pairs Grace + Hopper with up to 288 GB HBM and ~10 TB/s memory
  bandwidth between GPUs.
- 4090 / H100 are kept here as consumer / training upper bounds, but they live
  in a different design space than Spark DGX.
""")

# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------

def main():
    bench_env()
    bench_gemm()
    bench_mem()
    bench_kernel_latency()
    run_sd15_bench()
    run_sdxl_bench()
    run_sdxl_turbo_bench()
    run_llm_bench()
    run_vram_bench()
    print_final_summary()


if __name__ == "__main__":
    main()
