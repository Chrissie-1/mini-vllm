"""Measure what the KV cache actually buys.

Compares cached decoding against the naive baseline that re-runs the full
sequence every step, and asserts the two produce identical text -- a speedup that
changes the output is not a speedup.

    python bench/bench_kv_cache.py --tokens 16,32,64,128
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "worker"))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TORCH_THREADS", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

PROMPT = (
    "The history of computing begins long before the electronic computer, with "
    "mechanical calculators and the idea that arithmetic could be automated"
)


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - start


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", default="16,32,64,128")
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "gpt2"))
    args = parser.parse_args()

    import torch

    torch.set_num_threads(1)
    from server.model import LLMEngine

    print(f"loading {args.model} ...", flush=True)
    engine = LLMEngine(model_name=args.model)
    engine.generate_with_kv_cache(PROMPT, max_new_tokens=4)  # warm up

    lengths = [int(t) for t in args.tokens.split(",") if t.strip()]
    rows = []
    for n in lengths:
        cached, cached_s = timed(engine.generate_with_kv_cache, PROMPT, n)
        naive, naive_s = timed(engine.generate_without_cache, PROMPT, n)

        status = "identical" if cached == naive else "MISMATCH"
        if cached != naive:
            print(f"  !! outputs diverged at {n} tokens", file=sys.stderr)

        rows.append(
            (
                n,
                cached_s,
                naive_s,
                n / cached_s,
                n / naive_s,
                naive_s / cached_s,
                status,
            )
        )
        print(
            f"  {n:>4} tokens: cached {cached_s:6.2f}s  naive {naive_s:6.2f}s "
            f"-> {naive_s / cached_s:5.2f}x  ({status})",
            flush=True,
        )

    print()
    print(f"| Tokens | Cached | No cache | Cached tok/s | Naive tok/s | Speedup |")
    print(f"|-------:|-------:|---------:|-------------:|------------:|--------:|")
    for n, c, u, ctps, utps, speedup, _ in rows:
        print(
            f"| {n} | {c:.2f}s | {u:.2f}s | {ctps:.1f} | {utps:.1f} | **{speedup:.2f}x** |"
        )

    print()
    print(
        "Speedup grows with sequence length: the naive path redoes O(n^2) work "
        "in total, the cached path O(n)."
    )
    return 0 if all(r[-1] == "identical" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
