"""Measure what batching buys, by sweeping the batch-size limit.

Runs the same concurrent workload through the scheduler with ``max_batch_size``
pinned to each value in the sweep. A limit of 1 is the Phase 1 behaviour --
strictly one sequence at a time -- so it doubles as the honest baseline.

    python bench/bench_throughput.py --concurrency 16 --tokens 32 --batches 1,2,4,8

Add ``--http URL`` to drive a running server over HTTP instead of the in-process
scheduler; the in-process path is the default because it isolates the engine from
network and framework overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "In the year 2050, humanity",
    "The three laws of robotics are",
    "To be or not to be, that is",
    "The mitochondria is the",
    "Breaking news this morning:",
]


def percentiles(values):
    ordered = sorted(values)

    def pct(p):
        if not ordered:
            return 0.0
        idx = min(int(round(p / 100 * (len(ordered) - 1))), len(ordered) - 1)
        return ordered[idx]

    return pct(50), pct(95), pct(99)


async def run_inprocess(engine, batch_limit, concurrency, tokens):
    from server.batcher import DynamicBatcher
    from server.sampling import SamplingParams

    batcher = DynamicBatcher(engine, max_batch_size=batch_limit, window_ms=20)
    await batcher.start()
    try:
        params = SamplingParams(max_tokens=tokens)
        prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]

        start = time.perf_counter()
        results = await asyncio.gather(
            *(batcher.submit(p, params) for p in prompts)
        )
        elapsed = time.perf_counter() - start

        return {
            "elapsed": elapsed,
            "latencies": [r.total_s for r in results],
            "ttfts": [r.ttft_s for r in results],
            "tokens": sum(r.completion_tokens for r in results),
            "mean_batch": batcher.mean_batch_size,
            "steps": batcher.steps,
        }
    finally:
        await batcher.stop()


async def run_http(url, concurrency, tokens):
    import httpx

    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(concurrency)]
    async with httpx.AsyncClient(timeout=300.0) as client:

        async def one(prompt):
            began = time.perf_counter()
            resp = await client.post(
                url, json={"prompt": prompt, "max_tokens": tokens}
            )
            resp.raise_for_status()
            body = resp.json()
            return time.perf_counter() - began, body["usage"]["completion_tokens"]

        start = time.perf_counter()
        pairs = await asyncio.gather(*(one(p) for p in prompts))
        elapsed = time.perf_counter() - start

    return {
        "elapsed": elapsed,
        "latencies": [p[0] for p in pairs],
        "ttfts": [],
        "tokens": sum(p[1] for p in pairs),
        "mean_batch": float("nan"),
        "steps": 0,
    }


async def main_async(args) -> int:
    if args.http:
        stats = await run_http(args.http, args.concurrency, args.tokens)
        report([("http", stats)], args)
        return 0

    import torch

    torch.set_num_threads(1)
    from server.model import LLMEngine

    print(f"loading {args.model} ...", flush=True)
    engine = LLMEngine(model_name=args.model)
    engine.generate_with_kv_cache("warm up", max_new_tokens=4)

    rows = []
    for limit in [int(b) for b in args.batches.split(",") if b.strip()]:
        stats = await run_inprocess(engine, limit, args.concurrency, args.tokens)
        rows.append((limit, stats))
        print(
            f"  batch<={limit:<2}  {stats['elapsed']:6.2f}s  "
            f"{stats['tokens'] / stats['elapsed']:7.1f} tok/s  "
            f"mean batch {stats['mean_batch']:.2f}  steps {stats['steps']}",
            flush=True,
        )
    report(rows, args)
    return 0


def report(rows, args):
    print()
    print(
        f"{args.concurrency} concurrent requests x {args.tokens} tokens "
        f"({args.model}, CPU)"
    )
    print()
    print("| Max batch | Wall clock | Throughput | Mean batch | p50 | p95 | p99 | Speedup |")
    print("|----------:|-----------:|-----------:|-----------:|----:|----:|----:|--------:|")

    baseline = None
    for limit, stats in rows:
        throughput = stats["tokens"] / stats["elapsed"]
        if baseline is None:
            baseline = throughput
        p50, p95, p99 = percentiles(stats["latencies"])
        mean_batch = stats["mean_batch"]
        batch_str = "-" if mean_batch != mean_batch else f"{mean_batch:.2f}"
        print(
            f"| {limit} | {stats['elapsed']:.2f}s | {throughput:.1f} tok/s | "
            f"{batch_str} | {p50:.2f}s | {p95:.2f}s | {p99:.2f}s | "
            f"**{throughput / baseline:.2f}x** |"
        )

    print()
    print(
        "Throughput scales with batch size because a decode step costs roughly "
        "the same whether it advances one sequence or eight -- the model weights "
        "are read once either way."
    )
    print(
        "Under saturation, latency improves too: requests finish sooner because "
        "they stop queueing behind each other. The cost of batching is paid by a "
        "request that arrives alone, which waits out the admission window; the "
        "gain flattens once the batch saturates available compute."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--batches", default="1,2,4,8")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt2"))
    parser.add_argument("--http", default="", help="POST to this URL instead")
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
