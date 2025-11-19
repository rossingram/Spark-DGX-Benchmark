#!/usr/bin/env python
"""
DGX SPARK — GPT-OSS-120B LLM BENCHMARK

This script benchmarks a locally-served `openai/gpt-oss-120b` model running
behind a vLLM OpenAI-compatible server.

It assumes:
  - vLLM is running, e.g.:

        vllm serve openai/gpt-oss-120b \
          --tensor-parallel-size 1 \
          --trust-remote-code

  - The server is reachable at http://localhost:8000/v1
  - You are using the NVIDIA PyTorch 25.09 container from the README.

Usage (inside the container):

    cd /workspace/Spark-DGX-Benchmark
    python gpt_oss_benchmark.py

You can customize:

    python gpt_oss_benchmark.py \
      --server-url http://localhost:8000 \
      --model openai/gpt-oss-120b \
      --runs 16 \
      --warmup 4 \
      --max-batch 16
"""

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

from openai import OpenAI


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def percentile(values: List[float], pct: float) -> float:
    """Compute a simple percentile (e.g., 50, 95)."""
    if not values:
        return 0.0
    values_sorted = sorted(values)
    k = (len(values_sorted) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values_sorted) - 1)
    if f == c:
        return values_sorted[f]
    d0 = values_sorted[f] * (c - k)
    d1 = values_sorted[c] * (k - f)
    return d0 + d1


def format_seconds(s: float) -> str:
    return f"{s*1000:.1f} ms"


def safe_usage_tokens(resp) -> Tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) if present, else (0, 0)."""
    try:
        usage = resp.usage
        pt = getattr(usage, "prompt_tokens", 0) or 0
        ct = getattr(usage, "completion_tokens", 0) or 0
        return pt, ct
    except Exception:
        return 0, 0


# ---------------------------------------------------------------------------
# Core request primitive
# ---------------------------------------------------------------------------

def run_chat_once(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> Dict[str, float]:
    """
    Send a single non-streaming chat completion and measure latency & token usage.
    Returns:
        {
          "latency": float (seconds),
          "prompt_tokens": int,
          "completion_tokens": int,
        }
    """
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    t1 = time.time()

    prompt_tokens, completion_tokens = safe_usage_tokens(resp)

    return {
        "latency": t1 - t0,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


# ---------------------------------------------------------------------------
# Test 1 — Interactive batch-1 latency
# ---------------------------------------------------------------------------

def bench_interactive_latency(
    client: OpenAI,
    model: str,
    warmup: int,
    runs: int,
) -> None:
    print("\n[TEST 1] Interactive Copilot Latency (Batch=1)")
    print("-------------------------------------------------")

    messages = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {
            "role": "user",
            "content": (
                "Write a short Python function that computes the nth Fibonacci "
                "number iteratively, and explain it in 3 bullet points."
            ),
        },
    ]

    # Warmup
    for i in range(warmup):
        try:
            run_chat_once(client, model, messages)
            print(f"  Warmup {i+1}/{warmup} ... ok")
        except Exception as e:
            print(f"  Warmup {i+1}/{warmup} ... ERROR: {e}")

    # Measured runs
    latencies = []
    prompt_tokens = 0
    completion_tokens = 0

    for i in range(runs):
        try:
            res = run_chat_once(client, model, messages)
            latencies.append(res["latency"])
            prompt_tokens += res["prompt_tokens"]
            completion_tokens += res["completion_tokens"]
            print(
                f"  Run {i+1}/{runs}: "
                f"latency={format_seconds(res['latency'])}, "
                f"prompt_tokens={res['prompt_tokens']}, "
                f"completion_tokens={res['completion_tokens']}"
            )
        except Exception as e:
            print(f"  Run {i+1}/{runs}: ERROR: {e}")

    if not latencies:
        print("  No successful runs; skipping stats.")
        return

    total_time = sum(latencies)
    avg_latency = statistics.mean(latencies)
    p50 = percentile(latencies, 50)
    p95 = percentile(latencies, 95)
    tok_per_sec = (
        completion_tokens / total_time if total_time > 0 and completion_tokens > 0 else 0.0
    )

    print("\n  Results:")
    print(f"    Successful runs:     {len(latencies)}/{runs}")
    print(f"    Prompt tokens total: {prompt_tokens}")
    print(f"    Completion tokens:   {completion_tokens}")
    print(f"    Avg latency:         {format_seconds(avg_latency)}")
    print(f"    P50 latency:         {format_seconds(p50)}")
    print(f"    P95 latency:         {format_seconds(p95)}")
    print(f"    Decode throughput:   {tok_per_sec:.1f} tokens/sec (approx)")


# ---------------------------------------------------------------------------
# Test 2 — Batch scaling (concurrent requests)
# ---------------------------------------------------------------------------

def _chat_worker(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
) -> Dict[str, float]:
    return run_chat_once(client, model, messages, max_tokens=max_tokens, temperature=temperature)


def bench_batch_scaling(
    client: OpenAI,
    model: str,
    max_batch: int,
    runs_per_batch: int,
) -> None:
    print("\n[TEST 2] Batch Scaling (Parallel Requests)")
    print("-------------------------------------------------")
    print("  NOTE: This simulates multiple independent chats in parallel using threads.")

    base_messages = [
        {"role": "system", "content": "You are a concise assistant."},
        {
            "role": "user",
            "content": (
                "Summarize the advantages and tradeoffs of the NVIDIA DGX Spark "
                "for running large MoE models like gpt-oss-120b in 3-4 sentences."
            ),
        },
    ]

    batch_sizes = [1, 2, 4, 8, 16, 32]
    batch_sizes = [b for b in batch_sizes if b <= max_batch]

    for batch_size in batch_sizes:
        print(f"\n  Batch size: {batch_size}")
        latencies: List[float] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # We'll run "runs_per_batch" *batches*, each of size `batch_size`
        for r in range(runs_per_batch):
            # create independent messages for each request (identical content is fine)
            messages_list = [list(base_messages) for _ in range(batch_size)]

            start = time.time()
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = [
                    executor.submit(
                        _chat_worker,
                        client,
                        model,
                        messages_list[i],
                        256,
                        0.7,
                    )
                    for i in range(batch_size)
                ]

                for fut in as_completed(futures):
                    try:
                        res = fut.result()
                        latencies.append(res["latency"])
                        total_prompt_tokens += res["prompt_tokens"]
                        total_completion_tokens += res["completion_tokens"]
                    except Exception as e:
                        print(f"    Error in worker: {e}")

            end = time.time()
            wall = end - start
            # Report per-batch wall time
            print(
                f"    Batch {r+1}/{runs_per_batch}: wall={format_seconds(wall)} "
                f"(approx {batch_size} requests)"
            )

        if not latencies:
            print("    No successful requests; skipping stats.")
            continue

        total_wall = sum(latencies)  # sum of per-request times (not same as wall clock)
        avg_latency = statistics.mean(latencies)
        p50 = percentile(latencies, 50)
        p95 = percentile(latencies, 95)

        # Throughput using total completion tokens divided by *approximate* wall time.
        # Here we approximate wall time as (max lat in each batch) summed across batches.
        # Simpler: use total_completion_tokens / (sum(latencies) / batch_size_avg)
        batches = runs_per_batch
        approx_wall = (sum(latencies) / max(1, len(latencies))) * batches
        tok_per_sec = (
            total_completion_tokens / approx_wall
            if approx_wall > 0 and total_completion_tokens > 0
            else 0.0
        )

        print("  Summary for batch size", batch_size)
        print(f"    Requests:            ~{len(latencies)}")
        print(f"    Prompt tokens total: {total_prompt_tokens}")
        print(f"    Completion tokens:   {total_completion_tokens}")
        print(f"    Avg latency:         {format_seconds(avg_latency)}")
        print(f"    P50 latency:         {format_seconds(p50)}")
        print(f"    P95 latency:         {format_seconds(p95)}")
        print(f"    Approx throughput:   {tok_per_sec:.1f} tokens/sec")


# ---------------------------------------------------------------------------
# Test 3 — Long-context throughput
# ---------------------------------------------------------------------------

def make_long_context_text(scale: int) -> str:
    """
    Create a synthetic "document" that gets longer with `scale`.
    We'll rely on the server's usage.prompt_tokens to tell us actual token counts.
    """
    base_chunk = (
        "The DGX Spark is a compact Blackwell-based system designed for large "
        "inference workloads, unified memory experiments, and long-context models. "
        "This paragraph is repeated to build up a synthetic long context. "
    )
    # scale controls approximate size; tweak as needed
    return base_chunk * (128 * scale)


def bench_long_context(
    client: OpenAI,
    model: str,
    runs_per_size: int,
) -> None:
    print("\n[TEST 3] Long-Context Prefill + Decode Throughput")
    print("-------------------------------------------------")

    scales = [
        ("short", 1),
        ("medium", 4),
        ("long", 8),
    ]

    for label, scale in scales:
        print(f"\n  Context: {label} (scale={scale})")

        context_text = make_long_context_text(scale)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert systems engineer. Answer the user's "
                    "question using ONLY the context provided."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Context:\n\n"
                    + context_text
                    + "\n\nQuestion: Summarize the key properties of this system "
                      "in 4-5 bullet points."
                ),
            },
        ]

        latencies: List[float] = []
        prompt_tokens_list: List[int] = []
        completion_tokens_list: List[int] = []

        for i in range(runs_per_size):
            try:
                res = run_chat_once(
                    client,
                    model,
                    messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                latencies.append(res["latency"])
                prompt_tokens_list.append(res["prompt_tokens"])
                completion_tokens_list.append(res["completion_tokens"])
                print(
                    f"    Run {i+1}/{runs_per_size}: "
                    f"latency={format_seconds(res['latency'])}, "
                    f"prompt_tokens={res['prompt_tokens']}, "
                    f"completion_tokens={res['completion_tokens']}"
                )
            except Exception as e:
                print(f"    Run {i+1}/{runs_per_size}: ERROR: {e}")

        if not latencies:
            print("    No successful runs; skipping stats.")
            continue

        avg_latency = statistics.mean(latencies)
        avg_prompt_tokens = statistics.mean(prompt_tokens_list) if prompt_tokens_list else 0.0
        avg_completion_tokens = (
            statistics.mean(completion_tokens_list) if completion_tokens_list else 0.0
        )

        total_completion_tokens = sum(completion_tokens_list)
        total_time = sum(latencies)
        tok_per_sec = (
            total_completion_tokens / total_time
            if total_time > 0 and total_completion_tokens > 0
            else 0.0
        )

        print("  Summary for context:", label)
        print(f"    Avg prompt tokens:     {avg_prompt_tokens:.1f}")
        print(f"    Avg completion tokens: {avg_completion_tokens:.1f}")
        print(f"    Avg latency:           {format_seconds(avg_latency)}")
        print(f"    Decode throughput:     {tok_per_sec:.1f} tokens/sec")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DGX Spark — GPT-OSS-120B LLM benchmark (vLLM / OpenAI API compatible)."
    )
    parser.add_argument(
        "--server-url",
        type=str,
        default="http://localhost:8000",
        help="Base URL for the vLLM server (without /v1). Default: http://localhost:8000",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-oss-120b",
        help="Model name as exposed by vLLM. Default: openai/gpt-oss-120b",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="Number of warmup runs for the interactive test. Default: 4",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=16,
        help="Number of measured runs for the interactive test. Default: 16",
    )
    parser.add_argument(
        "--max-batch",
        type=int,
        default=16,
        help="Maximum batch size to test in the batch scaling test (capped at 32). Default: 16",
    )
    parser.add_argument(
        "--batch-runs",
        type=int,
        default=4,
        help="Number of batches to run per batch size. Default: 4",
    )
    parser.add_argument(
        "--long-context-runs",
        type=int,
        default=4,
        help="Number of runs per context size in the long-context test. Default: 4",
    )

    args = parser.parse_args()

    print("=" * 79)
    print("DGX SPARK — GPT-OSS-120B LLM BENCHMARK")
    print("=" * 79)
    print("Server configuration:")
    print(f"  URL:   {args.server_url}/v1")
    print(f"  Model: {args.model}")
    print(f"  Note:  Expect first run to download weights if not cached.\n")

    client = OpenAI(
        base_url=f"{args.server_url}/v1",
        api_key="EMPTY",  # vLLM doesn't require auth by default
    )

    # Run tests
    bench_interactive_latency(client, args.model, args.warmup, args.runs)
    bench_batch_scaling(client, args.model, args.max_batch, args.batch_runs)
    bench_long_context(client, args.model, args.long_context_runs)

    print("\n" + "=" * 79)
    print("GPT-OSS-120B BENCHMARK COMPLETE")
    print("=" * 79)


if __name__ == "__main__":
    main()
