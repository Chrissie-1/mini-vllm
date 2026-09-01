"""Continuous batching scheduler.

The engine can advance a whole batch in one forward pass, but requests arrive one
at a time. This module is the bridge: a single scheduler loop owns the
:class:`~server.model.BatchState`, and concurrent callers hand it work and await
a future.

Two batching behaviours, and the difference matters:

*Dynamic batching* is the cold-start case. When the engine is idle, the first
arrival does **not** run immediately -- the loop waits up to ``window_ms`` to see
who else shows up, then prefills them together. One request pays a bounded
latency tax; the batch it forms pays far less compute per request.

*Continuous batching* is the steady state. A running batch does not drain before
accepting new work: as soon as a sequence hits EOS it is evicted from the batch
and the KV cache, and a queued request is prefilled straight into the freed slot.
The alternative -- static batching -- would leave those rows idling until the
longest sequence in the batch finished.

The forward passes are dispatched with :func:`asyncio.to_thread`, which is what
lets the event loop keep accepting requests *during* a step. Without it the queue
could never fill and batches would always be size 1.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator, List, Optional

from . import kvcache, metrics
from .config import settings
from .model import BatchState, LLMEngine, Seq
from .sampling import SamplingParams


class QueueFullError(RuntimeError):
    """Raised when the admission queue is saturated; surfaces as HTTP 503."""


@dataclass
class Result:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    request_id: str
    queued_s: float
    ttft_s: float
    total_s: float


@dataclass
class Request:
    """A queued or running request and the plumbing to deliver its result."""

    seq: Seq
    future: asyncio.Future
    stream: Optional[asyncio.Queue] = None
    published: int = 0  # tokens already handed to the stream consumer
    published_text: str = ""
    submitted_at: float = field(default_factory=time.monotonic)
    started_at: Optional[float] = None


class DynamicBatcher:
    """Owns the batch state and the loop that advances it."""

    def __init__(
        self,
        engine: LLMEngine,
        max_batch_size: int = None,
        window_ms: int = None,
        max_queue_size: int = None,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size or settings.max_batch_size
        self.window_ms = settings.batch_window_ms if window_ms is None else window_ms
        self.max_queue_size = max_queue_size or settings.max_queue_size

        self.queue: List[Request] = []
        self.running: dict[int, Request] = {}  # seq_id -> Request
        self._state = BatchState()
        self._task: Optional[asyncio.Task] = None
        self._alive = False
        self._work = asyncio.Event()

        # Observability counters, read by the metrics exporter.
        self.steps = 0
        self.batched_rows = 0  # sum of batch sizes over steps -> mean occupancy
        self.completed = 0
        self.rejected = 0
        self.tokens_generated = 0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self) -> None:
        if self._alive:
            return
        self._alive = True
        self._task = asyncio.create_task(self._loop(), name="mini-vllm-scheduler")

    async def stop(self) -> None:
        self._alive = False
        self._work.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        for req in [*self.queue, *self.running.values()]:
            if not req.future.done():
                req.future.cancel()
        self.queue.clear()
        self.running.clear()
        self._state = BatchState()

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def add(self, request: Request) -> None:
        """Enqueue a prepared request. Raises if the queue is saturated."""
        if len(self.queue) >= self.max_queue_size:
            self.rejected += 1
            raise QueueFullError(
                f"queue is full ({self.max_queue_size} waiting)"
            )
        self.queue.append(request)
        self._work.set()

    def _make_request(
        self,
        prompt: str,
        params: Optional[SamplingParams],
        request_id: str,
        streaming: bool,
    ) -> Request:
        seq = self.engine.make_sequence(
            prompt, params, request_id=request_id or uuid.uuid4().hex
        )
        return Request(
            seq=seq,
            future=asyncio.get_running_loop().create_future(),
            stream=asyncio.Queue() if streaming else None,
        )

    async def submit(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        request_id: str = "",
        timeout: Optional[float] = None,
    ) -> Result:
        """Submit a prompt and await the completed generation."""
        req = self._make_request(prompt, params, request_id, streaming=False)
        self.add(req)
        try:
            return await asyncio.wait_for(
                req.future, timeout or settings.request_timeout_s
            )
        except asyncio.TimeoutError:
            self._abandon(req)
            raise

    async def stream(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        request_id: str = "",
    ) -> AsyncIterator[str]:
        """Submit a prompt and yield text deltas as they are produced."""
        req = self._make_request(prompt, params, request_id, streaming=True)
        self.add(req)
        try:
            while True:
                chunk = await asyncio.wait_for(
                    req.stream.get(), settings.request_timeout_s
                )
                if chunk is None:  # sentinel: generation finished
                    break
                yield chunk
        except (asyncio.TimeoutError, asyncio.CancelledError):
            self._abandon(req)
            raise
        # Surface an engine-side error that the sentinel path would otherwise hide.
        if req.future.done() and req.future.exception() is not None:
            raise req.future.exception()

    def _abandon(self, req: Request) -> None:
        """Drop a request whose caller has gone away."""
        if req in self.queue:
            self.queue.remove(req)
        # A running sequence is marked finished so the next eviction sweeps it up;
        # tearing it out of the KV cache mid-step would corrupt the batch.
        if req.seq.seq_id in self.running and not req.seq.finished:
            req.seq.finish_reason = "abandoned"

    # ------------------------------------------------------------------
    # The scheduler loop
    # ------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._alive:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - defensive
                self._fail_all(exc)

    async def _tick(self) -> None:
        # Nothing running and nothing waiting: sleep until someone submits.
        if self._state.empty and not self.queue:
            self._work.clear()
            await self._work.wait()
            return

        # Cold start: hold the admission window open to accumulate a batch.
        if self._state.empty and len(self.queue) < self.max_batch_size:
            await self._gather_window()

        newcomers = self._take_admissions()
        if newcomers:
            for req in newcomers:
                req.started_at = time.monotonic()
                self.running[req.seq.seq_id] = req
            seqs = [r.seq for r in newcomers]
            self._state = await asyncio.to_thread(self.engine.admit, self._state, seqs)
        elif not self._state.empty:
            self._state = await asyncio.to_thread(self.engine.decode_step, self._state)
        else:
            return

        self.steps += 1
        self.batched_rows += len(self._state)
        self._observe(newcomers)

        self._publish_deltas()
        self._state, finished = await asyncio.to_thread(
            self.engine.evict_finished, self._state
        )
        for seq in finished:
            self._complete(seq)

    def _observe(self, newcomers: List[Request]) -> None:
        """Account for what the step just did, in cache-hit terms.

        A prefill attends over the prompt and reuses nothing. A decode step
        attends over the whole cache but computes only the newest position, so
        every earlier position is a hit -- which is exactly the work the cache
        saved versus recomputing the sequence from scratch.
        """
        metrics.BATCH_SIZE.observe(len(self._state))
        if newcomers:
            attended = sum(len(r.seq.prompt_ids) for r in newcomers)
            metrics.CACHE_ATTENDED.inc(attended)
            return
        cached = kvcache.cache_length(self._state.cache)
        rows = len(self._state)
        metrics.CACHE_ATTENDED.inc(rows * cached)
        metrics.CACHE_REUSED.inc(rows * max(cached - 1, 0))

    async def _gather_window(self) -> None:
        """Wait up to ``window_ms`` for more arrivals, cutting short once full."""
        if self.window_ms <= 0:
            return
        deadline = time.monotonic() + self.window_ms / 1000.0
        while (
            self._alive
            and len(self.queue) < self.max_batch_size
            and time.monotonic() < deadline
        ):
            await asyncio.sleep(min(0.002, max(deadline - time.monotonic(), 0)))

    def _take_admissions(self) -> List[Request]:
        """Pop as many queued requests as there is room for in the batch."""
        capacity = self.max_batch_size - len(self._state)
        if capacity <= 0 or not self.queue:
            return []
        admitted, self.queue = self.queue[:capacity], self.queue[capacity:]
        return admitted

    # ------------------------------------------------------------------
    # Delivery
    # ------------------------------------------------------------------

    def _publish_deltas(self) -> None:
        """Push newly generated text to streaming consumers."""
        for seq in self._state.seqs:
            req = self.running.get(seq.seq_id)
            if req is None or req.stream is None:
                continue
            if len(seq.output_ids) <= req.published:
                continue
            # Decode the whole output and diff: a token can be a partial UTF-8
            # sequence, so decoding tokens individually would emit mojibake.
            text = self.engine.decode(seq.output_ids)
            delta, req.published_text = text[len(req.published_text):], text
            req.published = len(seq.output_ids)
            if delta:
                req.stream.put_nowait(delta)

    def _complete(self, seq: Seq) -> None:
        req = self.running.pop(seq.seq_id, None)
        self.completed += 1
        self.tokens_generated += len(seq.output_ids)
        if req is None:
            return

        now = time.monotonic()
        result = Result(
            text=self.engine.decode(seq.output_ids),
            prompt_tokens=len(seq.prompt_ids),
            completion_tokens=len(seq.output_ids),
            finish_reason=seq.finish_reason or "stop",
            request_id=seq.request_id,
            queued_s=(req.started_at or now) - req.submitted_at,
            ttft_s=(seq.first_token_at or now) - req.submitted_at,
            total_s=now - req.submitted_at,
        )
        if req.stream is not None:
            self._publish_final(req, result)
            req.stream.put_nowait(None)
        if not req.future.done():
            req.future.set_result(result)

    def _publish_final(self, req: Request, result: Result) -> None:
        delta = result.text[len(req.published_text):]
        if delta:
            req.stream.put_nowait(delta)
            req.published_text = result.text

    def _fail_all(self, exc: Exception) -> None:
        """A step blew up; fail everything in flight rather than hanging callers."""
        for req in list(self.running.values()):
            if not req.future.done():
                req.future.set_exception(exc)
            if req.stream is not None:
                req.stream.put_nowait(None)
        self.running.clear()
        self._state = BatchState()

    # ------------------------------------------------------------------

    @property
    def mean_batch_size(self) -> float:
        return self.batched_rows / self.steps if self.steps else 0.0

    def stats(self) -> dict:
        return {
            "running": len(self.running),
            "queued": len(self.queue),
            "batch_size": len(self._state),
            "max_batch_size": self.max_batch_size,
            "steps": self.steps,
            "mean_batch_size": round(self.mean_batch_size, 2),
            "completed": self.completed,
            "rejected": self.rejected,
            "tokens_generated": self.tokens_generated,
            **self.engine.cache_stats(self._state),
        }
