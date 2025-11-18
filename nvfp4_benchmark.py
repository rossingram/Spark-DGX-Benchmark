#!/usr/bin/env python3
import argparse
import time
import statistics
import json
import sys
from typing import Tuple, List, Dict, Any

import requests


def send_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout: int,
) -> Tuple[float, int]:
    """
    Send a single request to an OpenAI-compatible /v1/chat/completions endpoint,
    following the NVIDIA nvFP4 playbook style payload:
        {
          "model": "...",
          "prompt": "...",
          "max_tokens": ...,
          "temperature": ...,
          "stream": false
        }
    Returns: (latency_seconds, completion_tokens)
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    t0 = time.time()
    resp = requests.post(url, json=payload, timeout=timeout)
    t1 = time.time()

    latency = t1 - t0

    try:
        data = resp.json()
    except json.JSONDecodeError:
        raise RuntimeError(f"Non-JSON response from {url}: {resp.text[:500]}")

    if resp.status_code != 200:
        raise RuntimeError(
            f"Error from {url} (status {resp.status_code}): {json.dumps(data, indent=2)[:500]}"
        )

    usage = data.get("usage", {}) or {}
    completion_tokens = usage.get("completion_tokens")

    # Fallback: approximate tokens by splitting on spaces if usage is missing
    if completion_tokens is None:
        choices = data.get("choices", [])
        if choices:
            text = choices[0].get("message", {}).get("content") or choices[0].get(
                "text", ""
            )
            completion_tokens = max(1, len(text.split()))
        else:
            completion_tokens = 0

    return latency, completion_tokens


def run_benchmark(
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
    Run warmup + benchmark requests and collect latency / throughput stats.
    """
    print(f"\n=== Benchmarking {name} ===")
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

        try:
            latency, completion_tokens = send_request(
                url=url,
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=timeout,
            )
        except Exception as e:
            print(f"  [{label} {i+1}/{total_iters}] ERROR: {e}", file=sys.stderr)
            if not is_warmup:
                # count as failed run
                continue
        else:
            print(
                f"  [{label} {i+1}/{total_iters}] "
                f"latency={latency:.3f}s, completion_tokens={completion_tokens}"
            )

            if not is_warmup:
                latencies.append(latency)
                total_tokens += completion_tokens
                if latency > 0:
                    tokens_per_s.append(completion_tokens / latency)

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
        "throughput_tok_per_s_mean": statistics.fmean(tokens_per_s),
        "throughput_tok_per_s_p50": statistics.median(tokens_per_s),
        "throughput_tok_per_s_p90": percentile(tokens_per_s, 90),
        "throughput_tok_per_s_p99": percentile(tokens_per_s, 99),
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


def print_compare_table(baseline: Dict[str, Any], nvfp4: Dict[str, Any]) -> None:
    """
    Print a simple Before vs After style table.
    """
    def ratio(a: float, b: float) -> float:
        return (b / a) if a else 0.0

    print("\n================ NVFP4 Benchmark Summary ================\n")
    print(f"Baseline: {baseline['name']}  ({baseline['url']})")
    print(f"NVFP4:    {nvfp4['name']}  ({nvfp4['url']})")
    print()

    rows = [
        ("Requests", baseline["requests"], nvfp4["requests"]),
        ("Total tokens", baseline["total_tokens"], nvfp4["total_tokens"]),
        ("Latency mean (s)", baseline["latency_mean"], nvfp4["latency_mean"]),
        ("Latency p50 (s)", baseline["latency_p50"], nvfp4["latency_p50"]),
        ("Latency p90 (s)", baseline["latency_p90"], nvfp4["latency_p90"]),
        ("Latency p99 (s)", baseline["latency_p99"], nvfp4["latency_p99"]),
        (
            "Throughput mean (tok/s)",
            baseline["throughput_tok_per_s_mean"],
            nvfp4["throughput_tok_per_s_mean"],
        ),
        (
            "Throughput p50 (tok/s)",
            baseline["throughput_tok_per_s_p50"],
            nvfp4["throughput_tok_per_s_p50"],
        ),
        (
            "Throughput p90 (tok/s)",
            baseline["throughput_tok_per_s_p90"],
            nvfp4["throughput_tok_per_s_p90"],
        ),
        (
            "Throughput p99 (tok/s)",
            baseline["throughput_tok_per_s_p99"],
            nvfp4["throughput_tok_per_s_p99"],
        ),
    ]

    col_name_w = max(len(r[0]) for r in rows) + 2
    print(f"{'Metric'.ljust(col_name_w)} | Baseline      | NVFP4         | NVFP4/Baseline")
    print("-" * (col_name_w + 46))

    for name, base_val, nv_val in rows:
        if isinstance(base_val, int) and isinstance(nv_val, int):
            base_str = f"{base_val:d}".rjust(12)
            nv_str = f"{nv_val:d}".rjust(12)
        else:
            base_str = f"{base_val:>12.3f}"
            nv_str = f"{nv_val:>12.3f}"
        ratio_val = ratio(base_val, nv_val) if isinstance(base_val, (int, float)) else 0
        ratio_str = f"{ratio_val:>13.3f}"
        print(f"{name.ljust(col_name_w)} | {base_str} | {nv_str} | {ratio_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark baseline vs NVFP4 model over OpenAI-compatible TRT-LLM API."
    )
    parser.add_argument(
        "--baseline-url",
        default="http://localhost:8000/v1/chat/completions",
        help="Baseline server URL",
    )
    parser.add_argument(
        "--nvfp4-url",
        default="http://localhost:8001/v1/chat/completions",
        help="NVFP4 server URL",
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        help="Model name to send in the request payload",
    )
    parser.add_argument(
        "--prompt",
        default="Explain why Paris is great in 2–3 sentences.",
        help="Prompt to use for all benchmark requests",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="Max tokens to request for each completion",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=16,
        help="Number of measured requests per server (excluding warmup)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=4,
        help="Number of warmup requests per server",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds",
    )

    args = parser.parse_args()

    baseline_summary = run_benchmark(
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

    nvfp4_summary = run_benchmark(
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

    print_compare_table(baseline_summary, nvfp4_summary)


if __name__ == "__main__":
    main()
