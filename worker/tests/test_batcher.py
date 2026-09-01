"""Scheduler behaviour.

Batching is only worth anything if it is invisible: a request must produce the
same text whether it ran alone or shared a batch with seven others. That is what
most of these assert. The rest pin the scheduler's contract -- windows, queue
limits, streaming, and cleanup.
"""

import asyncio

import pytest

from server.batcher import DynamicBatcher, QueueFullError
from server.sampling import SamplingParams

pytestmark = pytest.mark.asyncio

PROMPTS = [
    "Once upon a time",
    "The capital of France is",
    "def fibonacci(n):",
    "In the year 2050, humanity",
]


@pytest.fixture
async def batcher(engine):
    b = DynamicBatcher(engine, max_batch_size=4, window_ms=20)
    await b.start()
    try:
        yield b
    finally:
        await b.stop()


async def test_single_submission_matches_direct_generation(batcher, engine):
    result = await batcher.submit("Once upon a time", SamplingParams(max_tokens=12))
    assert result.text == engine.generate_with_kv_cache(
        "Once upon a time", max_new_tokens=12
    )
    assert result.completion_tokens == 12
    assert result.finish_reason == "length"
    assert result.prompt_tokens == len(engine.encode("Once upon a time"))


async def test_concurrent_requests_match_solo_generation(batcher, engine):
    """The equivalence that makes batching safe to turn on."""
    expected = [engine.generate_with_kv_cache(p, max_new_tokens=10) for p in PROMPTS]

    results = await asyncio.gather(
        *(batcher.submit(p, SamplingParams(max_tokens=10)) for p in PROMPTS)
    )
    assert [r.text for r in results] == expected


async def test_requests_actually_share_a_batch(batcher):
    """Guards against a regression where every request runs in a batch of one."""
    await asyncio.gather(
        *(batcher.submit(p, SamplingParams(max_tokens=10)) for p in PROMPTS)
    )
    assert batcher.mean_batch_size > 1.0
    # Four requests of ten tokens each cost far fewer than 40 forward passes.
    assert batcher.steps < 40


async def test_batch_never_exceeds_its_limit(engine):
    b = DynamicBatcher(engine, max_batch_size=2, window_ms=10)
    await b.start()
    try:
        await asyncio.gather(
            *(b.submit(p, SamplingParams(max_tokens=6)) for p in PROMPTS)
        )
        assert b.mean_batch_size <= 2.0
    finally:
        await b.stop()


async def test_latecomer_is_admitted_into_a_running_batch(batcher, engine):
    """Continuous batching: arriving mid-flight must not change the output."""
    expected = engine.generate_with_kv_cache("def fibonacci(n):", max_new_tokens=8)

    long_running = asyncio.create_task(
        batcher.submit("The capital of France is", SamplingParams(max_tokens=40))
    )
    await asyncio.sleep(0.25)  # let the first request get well underway
    latecomer = await batcher.submit("def fibonacci(n):", SamplingParams(max_tokens=8))

    assert latecomer.text == expected
    assert not long_running.done()  # it really was still running
    await long_running


async def test_mixed_length_requests_all_complete(batcher):
    lengths = [3, 15, 6, 20]
    results = await asyncio.gather(
        *(
            batcher.submit(p, SamplingParams(max_tokens=n))
            for p, n in zip(PROMPTS, lengths)
        )
    )
    assert [r.completion_tokens for r in results] == lengths


async def test_stream_deltas_reassemble_into_the_full_text(batcher, engine):
    chunks = [
        c async for c in batcher.stream("Once upon a time", SamplingParams(max_tokens=12))
    ]
    assert chunks, "expected at least one delta"
    assert "".join(chunks) == engine.generate_with_kv_cache(
        "Once upon a time", max_new_tokens=12
    )


async def test_streaming_and_blocking_requests_coexist(batcher):
    async def drain():
        return "".join(
            [c async for c in batcher.stream(PROMPTS[0], SamplingParams(max_tokens=10))]
        )

    streamed, blocking = await asyncio.gather(
        drain(), batcher.submit(PROMPTS[0], SamplingParams(max_tokens=10))
    )
    assert streamed == blocking.text


async def test_queue_full_is_rejected(engine):
    b = DynamicBatcher(engine, max_batch_size=1, window_ms=0, max_queue_size=2)
    # Not started: nothing drains the queue, so admission pressure is deterministic.
    b.add(b._make_request("a", None, "", False))
    b.add(b._make_request("b", None, "", False))
    with pytest.raises(QueueFullError):
        b.add(b._make_request("c", None, "", False))
    assert b.rejected == 1


async def test_timeout_abandons_a_queued_request(engine):
    b = DynamicBatcher(engine, max_batch_size=1, window_ms=0)
    # Never started, so the request can only time out.
    with pytest.raises(asyncio.TimeoutError):
        await b.submit("hello", SamplingParams(max_tokens=5), timeout=0.05)
    assert b.queue == []


async def test_stats_report_scheduler_state(batcher):
    await batcher.submit("Once upon a time", SamplingParams(max_tokens=5))
    stats = batcher.stats()
    assert stats["completed"] == 1
    assert stats["tokens_generated"] == 5
    assert stats["max_batch_size"] == 4
    assert stats["running"] == 0 and stats["queued"] == 0
    assert stats["mean_batch_size"] > 0


async def test_stop_is_idempotent_and_clears_state(engine):
    b = DynamicBatcher(engine, max_batch_size=2)
    await b.start()
    await b.submit("Once upon a time", SamplingParams(max_tokens=3))
    await b.stop()
    await b.stop()
    assert b.queue == [] and b.running == {}
