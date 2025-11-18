# Spark-DGX-Benchmark

A practical benchmark suite for the [NVIDIA DGX Spark](https://docs.nvidia.com/dgx/dgx-spark/index.html)

---

## 📌 Overview & Purpose

The Spark DGX is a strange and fascinating machine — **128 GB of unified Blackwell memory** in a low-power box that doesn’t behave like a 4090 and doesn’t pretend to be an H100. When I first unboxed mine, I realized very quickly that all the usual GPU intuition breaks down: the theoretical numbers don’t map to real workloads, and there wasn’t a clean, honest way to understand what this thing is actually good at.

So ChatGPT and I created this benchmark.

The goal is to provide a **practical, real-world performance snapshot** across the kinds of workloads people actually run today: [GEMM throughput](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html), memory bandwidth, kernel latency, SD1.5, SDXL, SDXL Turbo, LLM tokens/sec, and usable unified memory. The script also includes a reference comparison across **Spark, L40S, H200, GH200, 4090, and H100** so you can see where Spark fits in the broader GPU landscape.

This isn’t meant to replicate NVIDIA’s theoretical peak numbers.
It’s meant to show **what the Spark DGX actually delivers in user-space**, using standard PyTorch and diffusers pipelines. It gives owners a consistent baseline, helps set proper expectations, and highlights Spark’s real advantage:

> **Large-memory, inference-first workloads that don’t fit well on traditional GPUs.**

---

## 🚀 Getting Started

This assumes:

* Your DGX Spark is already set up and updated
* You can SSH into the machine or access it through NVIDIA Sync
* Docker is already working (it is by default on Spark)

---

### 1. Start the recommended NVIDIA PyTorch container

This container contains:

* PyTorch **2.6+ with official Blackwell (sm_121) support**
* CUDA **13.0**
* cuDNN, NCCL, Apex, and NVIDIA optimizations
* A clean, known-good environment for reproducible benchmarks

```bash
docker run --gpus all -it nvcr.io/nvidia/pytorch:25.01-py3
```

You should now be inside:

```
root@<container>:/workspace#
```

---

### 2. Authenticate with Hugging Face (recommended)

Some models (SDXL Turbo, LLMs, etc.) require HF login.

```bash
hf auth login
```

Paste your token when prompted.

---

### 3. Install dependencies inside the container

We explicitly pin NumPy < 2.0 because PyTorch extensions in these containers are still built against NumPy 1.x.

```bash
pip install --upgrade pip \
  diffusers transformers accelerate sentencepiece safetensors huggingface_hub opencv-python "numpy<2"
```

---

### 4. Clone the benchmark repo

```bash
git clone https://github.com/rossingram/Spark-DGX-Benchmark.git
cd Spark-DGX-Benchmark
chmod +x spark_bench.py
```

---

### 5. Run the benchmark

```bash
python spark_bench.py
```

You should now see a full benchmark report including:

* Environment info
* GEMM throughput
* Memory bandwidth
* Kernel latency
* SD1.5 / SDXL / SDXL Turbo performance
* LLM tokens/sec
* Unified memory slewing limits
* Comparison table vs 4090, L40S, H200, GH200, and H100

---

## 🧠 Commentary: What Spark Is (and Isn’t)

The DGX Spark’s architecture is optimized for **inference, memory, and efficiency**, not brute-force GPU throughput. A few high-level notes:

### ⭐ What Spark *excels at*

* **Huge models** that don’t fit on consumer GPUs
* **Unified memory workloads** where CPU+GPU share 128 GB seamlessly
* **Batch-1 inference** (e.g., agents, RAG, copilots, local LLMs)
* **Long-context or MoE models**
* **Large diffusion models** that would OOM a 4090 instantly
* Running multiple concurrent models without sharding

### ⚠️ What Spark does *not* do well

* High-throughput FP16/BF16 training
* Multi-teraflop tensor-core GEMM workloads
* Anything that assumes H100-class tensor cores
* PCIe bandwidth dependent multi-GPU setups (Spark is single-GPU)

### 💡 Think of Spark as:

> **A 128 GB Blackwell inference appliance — more like a GH200 cousin than a gaming/workstation GPU.**

It trades raw FLOPs for:

* massive memory,
* low power draw,
* unified architecture,
* and ease of running huge models locally.

This benchmark helps quantify that tradeoff in real numbers.

---

## 📊 Output Example (abbreviated)

```
================================================================================
FINAL SUMMARY — SPARK vs 4090 vs H100 vs L40S vs H200 vs GH200
================================================================================
RAW COMPUTE (TFLOPs, FP16/BF16 — Measured)
    Spark: ~11–12 TFLOPs
    4090:  ~330 TFLOPs
    H100: ~1000 TFLOPs
...
```

---

## 📬 Feedback / Issues

If you have improvements, discoveries, or want to contribute additional tests:

PRs welcome.
Issues welcome.
Benchmark screenshots welcome.
