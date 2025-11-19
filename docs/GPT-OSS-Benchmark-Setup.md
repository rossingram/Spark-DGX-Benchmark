# 🚀 GPT-OSS-120B Benchmark Setup on DGX Spark

This guide explains how to benchmark **GPT-OSS-120B** on a **DGX Spark** using a clean two-container architecture:

* **Container A — vLLM Server:**
  Hosts the model and exposes an OpenAI-compatible API.

* **Container B — Benchmark Environment:**
  Runs the `Spark-DGX-Benchmark` scripts and talks to Container A over HTTP.

This method avoids CUDA compatibility issues and uses NVIDIA’s officially supported container images.

---

## 🧱 Prerequisites

* DGX Spark running DGX OS 2.6+
* Docker + NVIDIA Container Runtime
* Enough free NVMe storage (≈400 GB recommended for model caching)
* Internet access for initial model download

---

# 🟦 Container A — vLLM Server (GPT-OSS-120B)

This container serves `openai/gpt-oss-120b` using NVIDIA’s optimized vLLM image.

---

## 1. Launch the NVIDIA vLLM container

```bash
docker pull nvcr.io/nvidia/vllm:25.09-py3

docker run --gpus all -it --rm \
  --network host \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/vllm:25.09-py3
```

---

## 2. Install required tiktoken vocabs for GPT-OSS

Inside the vLLM container shell:

```bash
mkdir -p /workspace/tiktoken_encodings
cd /workspace/tiktoken_encodings

wget -O o200k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken"
wget -O cl100k_base.tiktoken "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"

export TIKTOKEN_ENCODINGS_BASE=/workspace/tiktoken_encodings
```

Optional (recommended):

```bash
export VLLM_USE_FLASHINFER_MXFP4_MOE=1
```

---

## 3. Serve GPT-OSS-120B using vLLM

Inside the same container:

```bash
vllm serve "openai/gpt-oss-120b" \
  --host 0.0.0.0 \
  --port 8000 \
  --async-scheduling \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.7 \
  --swap-space 16 \
  --max-model-len 32000 \
  --max-num-seqs 1024
```

This exposes:

```
http://localhost:8000/v1
```

Leave this container running.

---

# 🟩 Container B — Benchmark Environment

This is your working environment for running scripts inside **Spark-DGX-Benchmark**.

---

## 1. Launch the NVIDIA PyTorch container

```bash
docker run --gpus all -it \
  --network host \
  --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:25.09-py3
```

---

## 2. Clone the benchmark repository

Inside this container:

```bash
cd /workspace
git clone https://github.com/rossingram/Spark-DGX-Benchmark
cd Spark-DGX-Benchmark
```

---

## 3. Install dependencies

```bash
pip install --upgrade pip \
  diffusers transformers accelerate sentencepiece safetensors huggingface_hub opencv-python "numpy<2"

pip install "openai>=1.55.0"
```

Only the **OpenAI client** is required here.
Do **not** install vLLM in this container.

---

# 🔬 Running the GPT-OSS Benchmark

With Container A running vLLM at `localhost:8000` and Container B sitting in the repo directory:

```bash
python gpt_oss_benchmark.py \
  --server-url http://localhost:8000 \
  --model openai/gpt-oss-120b
```

This will execute:

* **Test 1:** Interactive copilot latency
* **Test 2:** Batch scaling (1 → N parallel requests)
* **Test 3:** Long-context throughput (prefill + decode)

---
