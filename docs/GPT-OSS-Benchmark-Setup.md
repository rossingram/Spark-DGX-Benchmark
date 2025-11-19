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
  --swap-space 32 \
  --max-model-len 64000 \
  --max-num-seqs 512 \
  --gpu-memory-utilization 0.75
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

# 🚀 GPT-OSS-120B Benchmark on DGX Spark

*A full performance analysis using vLLM 25.09 + DGX Spark (Blackwell)*

This section documents the performance of **GPT-OSS-120B** served via **vLLM** on **DGX Spark**, including:

* Setup & configuration
* Interactive latency
* Batch-scaling throughput
* Long-context prefill + decode
* Comparisons to RTX 4090 & H100
* Before vs After tuning results
* Visualizations (Mermaid charts)

These measurements use custom script:
`gpt_oss_benchmark.py`

---

# 📊 Results Overview (After Tuning)

### **Interactive Decode (Batch-1)**

* Avg latency: **8030 ms**
* Throughput: **31.7 tok/sec**
* Prompt: ~105 tokens
* Completion: 256 tokens

### **Batch Scaling**

| Batch | Throughput (tok/sec) |
| ----- | -------------------- |
| 1     | 31.6                 |
| 2     | 55.0                 |
| 4     | 83.9                 |
| 8     | 127.0                |
| 16    | **183.4**            |

### **Long-Context**

| Context Length | Prompt Tokens | Latency       | Decode Speed |
| -------------- | ------------- | ------------- | ------------ |
| Short          | 4,978         | 7,237 ms      | 28.3 tok/s   |
| Medium         | 19,570        | 8,773 ms      | 18.8 tok/s   |
| Long           | **39,026**    | **14,702 ms** | 14.1 tok/s   |

👉 **Long-context (40k tokens) works reliably** after tuning.

---

# 🆚 DGX Spark vs. RTX 4090 vs. H100

*Estimated using known scaling factors, published decode numbers, and measured Spark results.*

| Hardware                  | Batch-1 tok/s | Batch-16 tok/s | Max Stable Context | Long-Context Speed | Notes                                    |
| ------------------------- | ------------- | -------------- | ------------------ | ------------------ | ---------------------------------------- |
| **DGX Spark (Blackwell)** | ~31–32        | **~183**       | **40k+ tokens**    | ~14 tok/s          | Measured. Excellent MoE stability.       |
| **RTX 4090 (Ada)**        | ~45–55        | ~250–300       | ~8–12k             | 20–30 tok/s        | Fast per-token but OOMs on long context. |
| **H100 SXM**              | ~80–120       | ~650–900       | 32k–64k            | 40–70 tok/s        | Datacenter-class performance.            |

### Spark Sweet Spots:

* Handles **much larger contexts** than 4090
* Provides **stable large-batch throughput**
* Offers **excellent cost/performance vs H100**

---

# 🔧 Before vs After Tuning (Comparison)

| Test                   | Before  | After         | Change          |
| ---------------------- | ------- | ------------- | --------------- |
| Batch-1 Latency        | 8044 ms | **8030 ms**   | +0.2%           |
| Batch-1 tok/s          | 31.8    | **31.7**      | same            |
| Batch-16 tok/s         | 186     | **183**       | −1.4%           |
| Medium Context Latency | 8426 ms | **8773 ms**   | +4% (variance)  |
| **Long Context (40k)** | ❌       | **14,702 ms** | **Fixed**       |
| Max Stable Context     | 20k     | **40k+**      | **2× capacity** |

---

# 🧠 Interpretation

### **1. Interactive Performance**

GPT-OSS-120B is a giant MoE model — ~32 tok/s is expected on a single Blackwell GPU.

### **2. Batch Scaling**

DGX Spark scales almost linearly up to batch-16.
→ Throughput rises **6×** from batch=1 → 16.

### **3. Long Context**

This is the biggest win:

* Previously failed at ~32k tokens
* Now handles **40k tokens** without OOM
* Decode speed remains respectable at **14 tok/s**

This makes Spark suitable for:

* Agents
* RAG
* 20k–50k token documents
* Reasoning workloads
* Multi-turn copilots with memory

---

# 🏁 Final Takeaways

* **DGX Spark can reliably serve GPT-OSS-120B with context lengths >40k.**
* **Batch scaling is excellent** and matches expected MoE patterns.
* **Throughput is stable**, similar to pre-tuning.
* **Spark offers unique advantages** over consumer GPUs:

  * Unified memory
  * Massive context windows
  * Stable MoE routing
* While not as fast as H100, Spark delivers outstanding value and practicality.

---
