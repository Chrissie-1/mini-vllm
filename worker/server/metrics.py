"""Prometheus instrumentation.

Latency is recorded as three separate histograms rather than one, because in a
batching server they answer different questions: ``queue_wait`` is scheduler
pressure, ``ttft`` is what a streaming user actually feels, and ``latency`` is
end-to-end. A single number would hide which of the three regressed.

The cache "hit ratio" is exported as two counters rather than a precomputed
ratio, so Prometheus can rate() them over an arbitrary window:

    rate(llm_kv_cache_reused_positions_total[5m])
      / rate(llm_kv_cache_attended_positions_total[5m])

Reused positions are those served from the KV cache; attended positions are what
an uncached implementation would have had to recompute.
"""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

REQUESTS = Counter(
    "llm_requests_total",
    "Requests by terminal status.",
    ["status"],
    registry=REGISTRY,
)

LATENCY = Histogram(
    "llm_latency_seconds",
    "End-to-end request latency.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)

TTFT = Histogram(
    "llm_time_to_first_token_seconds",
    "Submission until the first token is produced.",
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
    registry=REGISTRY,
)

QUEUE_WAIT = Histogram(
    "llm_queue_wait_seconds",
    "Time spent waiting for admission into a batch.",
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 5),
    registry=REGISTRY,
)

TOKENS = Counter(
    "llm_tokens_generated_total",
    "Completion tokens produced.",
    registry=REGISTRY,
)

PROMPT_TOKENS = Counter(
    "llm_prompt_tokens_total",
    "Prompt tokens consumed.",
    registry=REGISTRY,
)

BATCH_SIZE = Histogram(
    "llm_batch_size",
    "Rows advanced per forward pass.",
    buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
    registry=REGISTRY,
)

RUNNING = Gauge(
    "llm_running_requests",
    "Sequences currently in the batch.",
    registry=REGISTRY,
)

QUEUED = Gauge(
    "llm_queued_requests",
    "Requests waiting for admission.",
    registry=REGISTRY,
)

CACHE_BYTES = Gauge(
    "llm_kv_cache_bytes",
    "Resident size of the KV cache.",
    registry=REGISTRY,
)

CACHE_TOKENS = Gauge(
    "llm_kv_cache_tokens",
    "Cached positions along the sequence axis.",
    registry=REGISTRY,
)

CACHE_REUSED = Counter(
    "llm_kv_cache_reused_positions_total",
    "Positions served from the KV cache instead of being recomputed.",
    registry=REGISTRY,
)

CACHE_ATTENDED = Counter(
    "llm_kv_cache_attended_positions_total",
    "Positions attended over; the denominator of the cache hit ratio.",
    registry=REGISTRY,
)

MODEL_INFO = Gauge(
    "llm_model_info",
    "Static labels describing the loaded model.",
    ["model", "device"],
    registry=REGISTRY,
)


def record_result(result) -> None:
    """Record one successful completion."""
    REQUESTS.labels(status=result.finish_reason).inc()
    LATENCY.observe(result.total_s)
    TTFT.observe(result.ttft_s)
    QUEUE_WAIT.observe(result.queued_s)
    TOKENS.inc(result.completion_tokens)
    PROMPT_TOKENS.inc(result.prompt_tokens)


def record_error(status: str = "error") -> None:
    REQUESTS.labels(status=status).inc()


def observe_batcher(batcher) -> None:
    """Refresh the gauges from live scheduler state. Called on scrape."""
    stats = batcher.stats()
    RUNNING.set(stats["running"])
    QUEUED.set(stats["queued"])
    CACHE_BYTES.set(stats["cache_bytes"])
    CACHE_TOKENS.set(stats["cache_len"])
