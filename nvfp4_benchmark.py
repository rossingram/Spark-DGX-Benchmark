#!/usr/bin/env python3
"""
nvfp4_benchmark.py

Simple "before vs after" benchmark for NVFP4-quantized models on DGX Spark.

This script assumes:
  * You have a baseline model served behind an OpenAI-compatible API.
  * You have an NVFP4-quantized model served behind an OpenAI-compatible API.
  * Both servers implement POST /v1/chat/completions with payload:
        {
          "model": "<name>",
          "prompt": "<prompt>",
          "max_tokens": <int>,
          "temperature": <float>,
          "stream": false
        }

Default ports:
  * Baseline: http://localhost:8001/v1/chat/completions
  * NVFP4:    http://localhost:8000/v1/chat/completions  (matches NVIDIA playbook)

Example:

  # Baseline container (unquantized model) on port 8001
  # docker run ... trtllm-serve /workspace/model --backend pytorch --max_batch_size 4 --port 8001

  # NVFP4 container (quantized model from nvFP4 playbook) on port 8000
  # docker run ... trtllm-serve /workspace/model --backend pytorch --max_batch_size 4 --port 8000

  python nvfp4_benchmark.py \
    --baseline-url http://localhost:8001/v1/chat/completions \
    --nvfp4-url    http://localhost:8000/v1/chat/completions \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --prompt "Paris is great because" \
    --requests 16 \
    --warmup 4
"""

import argparse
import json
import statistics
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
from urllib.error import URLError, HTTPError


BANNER_WIDTH = 79


def banner(title: str) -> None:
    print("=" * BANNER_WIDTH)
    print(title)
    print("=" * BANNER_WIDTH)


def section(title: str) -> None:
    print("\n" + title)
    print("-" * len(title))


def post_chat_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Dict[str, Any]:
    """
    Minimal HTTP client using stdlib only, to match the rest of this repo.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib_request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                raise RuntimeError(f"Non-JSON response from {url}: {raw[:500]}")
    except HTTPError as e:
        raise RuntimeError(f"HTTP error from {url}: {e.code} {e.reason}") from e
    except URLError as e:
        raise RuntimeError(f"Connection error to {url}: {e.reason}") from e


def extract_completion_tokens(response: Dict[str, Any]) -> int:
    """
    Try to get completion_tokens from OpenAI-style 'usage'.
    Fallback: approximate via word count on first choice.
    """
    usage = response.get("usage") or {}
    if "completion_tokens" in usage:
        return int(usage["completion_tokens"])

    choices = response.get("choices") or []
    if not choices:
        return 0

    # Handle both chat-style and text-style responses.
    first = choices[0]
    content = ""

    if "message" in first and isinstance(first["message"], dict):
        content = first["message"].get("content") or ""
    elif "text" in first:
        content = first.get("text") or ""

    if not isinstance(content, str):
        return 0

    # Very rough approximation if the server did not provide usage.
    return max(1, len(content.split()))


def run_benchmark_for_server(
    name: str,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    requests_n: int,
    warmup: int,
    timeout: int,
) -> Dict[str, Any]:
    """
    Run warmup + measured requests for a single server and collect stats.
    """
    section(f"Benchmarking {name}")
    print(f"  URL:   {url}")
    print(f"  Model: {model}")
    print(f"  Prompt: {prompt!r}")
    print(f"  Requests: {requests_n} (warmup: {warmup})")

    latencies: List[float] = []
    tokens_per_s: List[float] = []
    total_tokens = 0

    total_iters = warmup + requests_n

    for i in range(total_iters):
        is_warmup = i < warmup
        label = "warmup" if is_warmup else "run"
        idx = i + 1

        t0 = time.time()
        try:
            resp = post_chat_completion(
                url=url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            elapsed = time.time() - t0
            print(
                f"  [{name} {label} {idx}/{total_iters}] ERROR after {elapsed:.3f}s: {e}",
                file=sys.stderr,
            )
            if not is_warmup:
                # Count as failed; we just skip stats for this iteration.
                continue
            else:
                continue

        elapsed = time.time() - t0
        completion_tokens = extract_completion_tokens(resp)

        print(
            f"  [{name} {label} {idx}/{total_iters}] "
            f"latency={elapsed:.3f}s, completion_tokens={completion_tokens}"
        )

        if not is_warmup:
            latencies.append(elapsed)
            total_tokens += completion_tokens
            if elapsed > 0 and completion_tokens > 0:
                tokens_per_s.append(completion_tokens / elapsed)

    if not latencies:
        raise RuntimeError(f"No successful benchmark runs for {name}")

    summary = {
        "name": name,
        "url": url,
        "model": model,
        "requests": len(latencies),
        "total_tokens": total_tokens,
        "latency_mean": statistics.fmean(latencies),
        "latency_p50": statistics.median(latencies),
        "latency_p90": percentile(latencies, 90),
        "latency_p99": percentile(latencies, 99),
        "throughput_tok_per_s_mean": statistics.fmean(tokens_per_s)
        if tokens_per_s
        else 0.0,
        "throughput_tok_per_s_p50": statistics.median(tokens_per_s)
        if tokens_per_s
        else 0.0,
        "throughput_tok_per_s_p90": percentile(tokens_per_s, 90)
        if tokens_per_s
        else 0.0,
        "throughput_tok_per_s_p99": percentile(tokens_per_s, 99)
        if tokens_per_s
        else 0.0,
    }

    return summary


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    k = (len(values) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    d0 = values[f] * (c - k)
    d1 = values[c] * (k - f)
    return d0 + d1


def ratio(baseline: float, nvfp4: float) -> Optional[float]:
    if baseline == 0:
        return None
    return nvfp4 / baseline


def print_single_summary(summary: Dict[str, Any]) -> None:
    name = summary["name"]

    section(f"{name.upper()} SUMMARY")
    print(f"  URL:   {summary['url']}")
    print(f"  Model: {summary['model']}")
    print(f"  Requests: {summary['requests']}")
    print(f"  Total tokens: {summary['total_tokens']}")
    print()
    print(f"  Latency mean (s):       {summary['latency_mean']:.3f}")
    print(f"  Latency p50 (s):        {summary['latency_p50']:.3f}")
    print(f"  Latency p90 (s):        {summary['latency_p90']:.3f}")
    print(f"  Latency p99 (s):        {summary['latency_p99']:.3f}")
    print()
    print(
        f"  Throughput mean (tok/s): {summary['throughput_tok_per_s_mean']:.3f}"
    )
    print(
        f"  Throughput p50 (tok/s):  {summary['throughput_tok_per_s_p50']:.3f}"
    )
    print(
        f"  Throughput p90 (tok/s):  {summary['throughput_tok_per_s_p90']:.3f}"
    )
    print(
        f"  Throughput p99 (tok/s):  {summary['throughput_tok_per_s_p99']:.3f}"
    )


def print_compare_table(
    baseline: Dict[str, Any], nvfp4: Dict[str, Any]
) -> None:
    banner("FINAL SUMMARY — NVFP4 vs BASELINE (LLM INFERENCE)")

    print(f"Baseline: {baseline['name']}  ({baseline['url']})")
    print(f"NVFP4:    {nvfp4['name']}  ({nvfp4['url']})")
    print()
    print(
        "Note: Ratios are NVFP4/Baseline. "
        "For latency, values < 1.0 = faster. For throughput, > 1.0 = better."
    )
    print()

    rows = [
        ("Requests", baseline["requests"], nvfp4["requests"], False),
        ("Total tokens", baseline["total_tokens"], nvfp4["total_tokens"], False),
        ("Latency mean (s)", baseline["latency_mean"], nvfp4["latency_mean"], True),
        ("Latency p50 (s)", baseline["latency_p50"], nvfp4["latency_p50"], True),
        ("Latency p90 (s)", baseline["latency_p90"], nvfp4["latency_p90"], True),
        ("Latency p99 (s)", baseline["latency_p99"], nvfp4["latency_p99"], True),
        (
            "Throughput mean (tok/s)",
            baseline["throughput_tok_per_s_mean"],
            nvfp4["throughput_tok_per_s_mean"],
            True,
        ),
        (
            "Throughput p50 (tok/s)",
            baseline["throughput_tok_per_s_p50"],
            nvfp4["throughput_tok_per_s_p50"],
            True,
        ),
        (
            "Throughput p90 (tok/s)",
            baseline["throughput_tok_per_s_p90"],
            nvfp4["throughput_tok_per_s_p90"],
            True,
        ),
        (
            "Throughput p99 (tok/s)",
            baseline["throughput_tok_per_s_p99"],
            nvfp4["throughput_tok_per_s_p99"],
            True,
        ),
    ]

    col_name_w = max(len(r[0]) for r in rows) + 2
    header = (
        f"{'Metric'.ljust(col_name_w)} | "
        f"{'Baseline'.rjust(12)} | "
        f"{'NVFP4'.rjust(12)} | "
        f"{'NVFP4/Baseline'.rjust(15)}"
    )
    print(header)
    print("-" * len(header))

    for name, base_val, nv_val, show_ratio in rows:
        if isinstance(base_val, int) and isinstance(nv_val, int):
            base_str = f"{base_val:d}".rjust(12)
            nv_str = f"{nv_val:d}".rjust(12)
        else:
            base_str = f"{base_val:12.3f}"
            nv_str = f"{nv_val:12.3f}"

        if show_ratio and isinstance(base_val, (int, float)):
            r = ratio(float(base_val), float(nv_val))
            ratio_str = f"{r:15.3f}" if r is not None else " " * 15
        else:
            ratio_str = " " * 15

        print(
            f"{name.ljust(col_name_w)} | {base_str} | {nv_str} | {ratio_str}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark NVFP4-quantized model vs baseline over an "
            "OpenAI-compatible TRT-LLM API."
        )
    )
    parser.add_argument(
        "--baseline-url",
        default="http://localhost:8001/v1/chat/completions",
        help="Baseline server URL (OpenAI-compatible /v1/chat/completions).",
    )
    parser.add_argument(
        "--nvfp4-url",
        default="http://localhost:8000/v1/chat/completions",
        help="NVFP4 server URL (OpenAI-compatible /v1/chat/completions).",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to send in the request payload.",
    )
    parser.add_argument(
        "--prompt",
        default="Paris is great because",
        help="Prompt to use for all benchmark requests.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to request for each completion.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Number of measured requests per server (excluding warmup).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="Number of warmup requests per server.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--nvfp4-only",
        action="store_true",
        help="Only benchmark the NVFP4 server (no baseline comparison).",
    )

    args = parser.parse_args()

    banner("DGX SPARK — NVFP4 LLM INFERENCE BENCHMARK")

    # Always run NVFP4; baseline is optional if --nvfp4-only is set.
    nv_summary = run_benchmark_for_server(
        name="nvfp4",
        url=args.nvfp4_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        requests_n=args.requests,
        warmup=args.warmup,
        timeout=args.timeout,
    )

    print_single_summary(nv_summary)

    if args.nvfp4_only:
        print()
        print("NVFP4-only mode enabled; skipping baseline comparison.")
        return

    base_summary = run_benchmark_for_server(
        name="baseline",
        url=args.baseline_url,
        model=args.model,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        requests_n=args.requests,
        warmup=args.warmup,
        timeout=args.timeout,
    )

    print_single_summary(base_summary)
    print_compare_table(base_summary, nv_summary)


if __name__ == "__main__":
    main()
