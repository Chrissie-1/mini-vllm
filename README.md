# Mini-vLLM

A miniature LLM inference engine, built from the transformer forward pass up: a
hand-written KV cache, a continuous-batching scheduler, a Go control plane
talking gRPC to Python compute workers, and a Prometheus/Grafana stack watching
all of it.

It is not a wrapper. The batching, the cache surgery, the eviction policy, and
the admission control are implemented here, and every performance claim below
comes from a benchmark in [`bench/`](bench/) that you can re-run.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
  HTTP              │            Go gateway  :8080             │
  clients  ────────▶│                                          │
                    │  • admission control (shed, don't queue) │
                    │  • least-in-flight routing               │
                    │  • health checks, SSE fan-out            │
                    └───────────────┬──────────────────────────┘
                                    │ gRPC
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌───────────────────────┐       ┌───────────────────────┐
        │  Python worker :50051 │       │  Python worker :50051 │
        │                       │       │                       │
        │  DynamicBatcher       │       │  DynamicBatcher       │
        │    └─ LLMEngine       │       │    └─ LLMEngine       │
        │         └─ KV cache   │       │         └─ KV cache   │
        └───────────┬───────────┘       └───────────┬───────────┘
                    │ :9100/metrics                 │
                    └───────────────┬───────────────┘
                                    ▼
                    ┌───────────────────────────────┐
                    │  Prometheus :9090 → Grafana   │
                    └───────────────────────────────┘
```

**Why two languages.** The gateway is I/O-bound and must stay responsive while
workers are saturated; a worker holds the GIL and a model and can only run one
forward pass at a time. In one process, slow inference stalls connection
handling. Split, the gateway keeps accepting and shedding correctly under load,
and workers scale horizontally without the routing layer changing at all.

---

## Quickstart

```bash
docker compose up -d --build

curl -X POST localhost:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "Once upon a time", "max_tokens": 30}'
```

```json
{
  "id": "cmpl-1724...",
  "object": "text_completion",
  "text": ", the world was a place of great beauty and great danger...",
  "finish_reason": "length",
  "worker": "worker1:50051",
  "usage": { "prompt_tokens": 4, "completion_tokens": 30, "total_tokens": 34 }
}
```

Streaming uses the same endpoint with `"stream": true` and returns
server-sent events terminated by `data: [DONE]`.

| Service | URL |
|---|---|
| Gateway | http://localhost:8080 |
| Prometheus | http://localhost:9090 |
| Grafana | http://localhost:3000 (`admin`/`admin`) |

### Without Docker

```bash
make install          # venv + dependencies
make test             # Python and Go test suites
make run-http         # FastAPI worker alone on :8000, no gateway needed
```

`make run-http` is the Phase 1/2 path: the full engine and scheduler, minus the
distributed plumbing. On Windows, run the commands under each target directly —
the paths are already Windows-aware but `make` itself may not be installed.

---

## The KV cache

A decoder-only transformer recomputes the keys and values for every previous
token on every step. Caching them turns each decode step into a forward pass over
*one* token against a stored cache, so generating `n` tokens costs `O(n)` passes
instead of `O(n²)` of redundant recomputation.

```python
# Prefill: one pass over the prompt, fills the cache, emits token #1.
outputs = self.model(input_ids=input_ids, attention_mask=mask,
                     position_ids=positions, use_cache=True)

# Decode: feed only the newest token, plus everything already cached.
outputs = self.model(input_ids=state.next_tokens,
                     attention_mask=mask,
                     position_ids=mask.sum(dim=1, keepdim=True) - 1,
                     past_key_values=kvcache.to_cache(state.cache),
                     use_cache=True)
```

Measured against a baseline that re-runs the whole sequence every step
(`python bench/bench_kv_cache.py`, gpt2, single-threaded CPU):

| Tokens | Cached | No cache | Cached tok/s | Naive tok/s | Speedup |
|-------:|-------:|---------:|-------------:|------------:|--------:|
| 16 | 0.38s | 1.56s | 41.9 | 10.3 | **4.08×** |
| 32 | 0.71s | 3.64s | 45.1 | 8.8 | **5.13×** |
| 64 | 1.39s | 10.30s | 46.1 | 6.2 | **7.42×** |
| 128 | 3.12s | 33.22s | 41.1 | 3.9 | **10.65×** |

The shape matters more than the headline number. Cached throughput is *flat* in
sequence length (41–46 tok/s); the naive path degrades steadily (10.3 → 3.9
tok/s). That divergence is the `O(n)`-vs-`O(n²)` signature, and it means the
speedup keeps growing with longer generations.

Both paths produce **byte-identical output** at every length — asserted by
`test_kv_cache_matches_uncached_recomputation` and re-checked by the benchmark's
exit code in CI. An optimisation that changes the output is a bug, not a
speedup.

> **On the "O(1) per token" claim** you'll see in write-ups of this technique:
> it isn't accurate, and this project doesn't make it. A cached decode step still
> attends over every cached position, so one step is `O(n)` in the cache length,
> not `O(1)`. What the cache eliminates is redundant *recomputation* of prior
> keys and values — which is what takes total generation cost from `O(n²)` to
> `O(n)`. The measured numbers above are what that's actually worth.

---

## Continuous batching

Sequences in a batch finish at different times. Static batching leaves finished
rows idling until the slowest one is done; continuous batching evicts them
immediately and prefills queued requests into the freed slots.

That requires mutating the batch dimension of a live KV cache, which
`transformers` does not expose. [`worker/server/kvcache.py`](worker/server/kvcache.py)
implements the primitives — `select_rows` to evict, `concat_batch` to admit,
`left_pad` to align ragged lengths, `trim_left` to bound growth.

The subtle part is **left padding**. Padding on the left puts every sequence's
newest token in the same column, so one `[batch, 1]` forward pass advances the
whole batch. But the raw column index then overstates each real token's position
and would corrupt the positional embeddings, so `position_ids` are recomputed
from the attention mask's cumulative sum rather than taken from the column index.

Three tests pin this down, and they are the load-bearing ones in the suite:

- a batched sequence's output matches running it alone,
- it is unaffected by *which* other sequences share its batch,
- a sequence that survives an eviction continues exactly as if it had run solo.

Throughput across batch-size limits (`python bench/bench_throughput.py`,
16 concurrent requests × 32 tokens, gpt2, single-threaded CPU):

| Max batch | Wall clock | Throughput | Mean batch | p50 | p95 | p99 | Speedup |
|----------:|-----------:|-----------:|-----------:|----:|----:|----:|--------:|
| 1 | 14.01s | 36.5 tok/s | 1.00 | 7.00s | 13.25s | 14.01s | **1.00×** |
| 2 | 11.40s | 44.9 tok/s | 2.00 | 7.32s | 11.40s | 11.40s | **1.23×** |
| 4 | 6.99s | 73.3 tok/s | 4.00 | 5.10s | 6.99s | 6.99s | **2.01×** |
| 8 | 3.94s | 130.0 tok/s | 8.00 | 3.94s | 3.94s | 3.94s | **3.56×** |
| 16 | 3.87s | 132.1 tok/s | 16.00 | 3.87s | 3.87s | 3.87s | **3.62×** |

**3.6×, not 8×.** Batching wins because a decode step costs about the same
whether it advances one sequence or eight — the model weights are read from
memory once either way. The gain flattens past batch 8 because a single CPU
thread becomes compute-bound: at that point the weights are no longer the
bottleneck, the matmuls are. On a GPU, where the memory-bandwidth headroom is far
larger, this curve keeps climbing much further. Quoting 8× here would be quoting
someone else's hardware.

Note that latency *improves* too, rather than trading off against throughput.
That is specific to a saturated server: requests finish sooner because they stop
queueing behind each other. The cost of batching is paid by a request that
arrives at an idle engine and waits out the 20 ms admission window.

---

## Observability

Both planes export Prometheus metrics, and they deliberately don't overlap — the
gateway reports admission and routing, the worker reports batching and cache
behaviour. One number, one owner.

| Metric | Plane | What it answers |
|---|---|---|
| `gateway_requests_total{outcome}` | control | Is load being shed, or are workers erroring? |
| `gateway_latency_seconds` | control | What do clients actually experience? |
| `gateway_worker_inflight{worker}` | control | Is routing balanced? |
| `llm_batch_size` | compute | Are batches forming, or is everything running solo? |
| `llm_time_to_first_token_seconds` | compute | What does a streaming user wait for? |
| `llm_queue_wait_seconds` | compute | Is the scheduler the bottleneck? |
| `llm_kv_cache_bytes` | compute | How close is the cache to exhausting memory? |
| `llm_kv_cache_{reused,attended}_positions_total` | compute | Cache hit ratio |

The cache hit ratio is exported as two counters rather than a precomputed ratio,
so it can be rated over any window:

```promql
sum(rate(llm_kv_cache_reused_positions_total[5m]))
  / sum(rate(llm_kv_cache_attended_positions_total[5m]))
```

A pre-built Grafana dashboard is provisioned automatically at
[`ops/grafana/dashboards/mini-vllm.json`](ops/grafana/dashboards/mini-vllm.json).

---

## Design decisions

**Shed load, don't queue it.** The gateway's admission control is a fixed-size
semaphore that returns 503 immediately when full, rather than blocking. A client
that has already given up gains nothing from being queued, and an unbounded queue
converts a throughput problem into a memory problem.

**Least-in-flight, not round-robin.** Generation time varies by more than an
order of magnitude with output length, so round-robin reliably parks short
requests behind long ones. Least-in-flight tracks real occupancy and
self-corrects.

**Ejected workers keep being polled.** A failed health check takes a worker out
of rotation but not out of the poll loop, so it rejoins by itself once it
recovers — no restart, no manual intervention.

**Forward passes run in a thread.** The scheduler dispatches every step through
`asyncio.to_thread`. Without it the event loop would block for the whole forward
pass, no requests could arrive mid-step, and batches would never grow past one —
the batching would be real code that never actually batched.

**Generated bindings aren't committed (Go side).** The Go stubs are generated
during the image build, so the wire contract can only come from
[`proto/inference.proto`](proto/inference.proto). Python stubs are committed, as
is conventional, and CI regenerates them and fails on any diff.

---

## Layout

```
mini-vllm/
├── proto/inference.proto        # the wire contract
├── worker/                      # Python compute plane
│   └── server/
│       ├── kvcache.py           # cache surgery: evict, admit, pad, trim
│       ├── model.py             # prefill / decode / eviction / admission
│       ├── batcher.py           # continuous-batching scheduler
│       ├── sampling.py          # greedy, top-k, top-p (per-row in a batch)
│       ├── api.py               # FastAPI + SSE
│       ├── worker.py            # gRPC server
│       └── metrics.py           # Prometheus collectors
├── gateway/                     # Go control plane
│   ├── cmd/server/main.go       # HTTP, SSE proxying, error mapping
│   └── internal/scheduler/      # admission control + least-in-flight routing
├── bench/                       # the benchmarks behind every number above
├── ops/                         # Prometheus config, Grafana dashboard
└── .github/workflows/ci.yml
```

---

## Tests

```bash
make test          # or: cd worker && pytest -q ; cd gateway && go test ./...

# Fast lane: a tiny randomly-initialised checkpoint, no 500 MB download.
TEST_MODEL=hf-internal-testing/tiny-random-gpt2 pytest -q    # ~5s
```

56 Python tests and 7 Go tests. `TEST_MODEL` swaps the checkpoint the suite runs
against: every assertion is about engine mechanics -- cache surgery, batch
invariance, stopping rules -- none of which depend on the weights being any good,
so the tiny model exercises the same code paths in a twentieth of the time. CI
runs both. The suite is built around equivalence rather than
snapshots: fast paths are pinned against slow paths that are obviously correct,
so an optimisation that changes behaviour fails loudly instead of silently
producing different text.

CI additionally builds both images and runs an end-to-end check through the full
HTTP → gRPC → engine path via `docker compose`.

---

## Limits

Worth being straight about what this is and isn't:

- **CPU and gpt2 by default.** Any `AutoModelForCausalLM` works via `MODEL_NAME`,
  but the benchmarks above are single-threaded CPU on a small model.
- **No paged attention.** The cache is a dense tensor per batch, so memory scales
  with `batch × longest_sequence`. Real vLLM's central contribution is paging
  that into fixed blocks to eliminate the padding waste; this doesn't do that.
  `trim_left` bounds growth, which is a much blunter instrument.
- **No prefix sharing.** Two requests with a common prefix each prefill it.
- **Prefill isn't chunked.** A very long prompt blocks the batch for its whole
  prefill pass, which shows up as a TTFT spike for everyone else in the batch.
- **In-memory state only.** No persistence, no auth, no rate limiting per client.
