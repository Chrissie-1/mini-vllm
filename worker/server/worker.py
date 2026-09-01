"""gRPC inference worker.

The compute plane. The gateway owns HTTP, admission and routing; this process
owns the model and nothing else. It runs the same :class:`DynamicBatcher` the
HTTP app uses, so a request arriving over gRPC lands in the same batch as one
arriving over HTTP.

``grpc.aio`` rather than the threadpool server, because the batcher is an asyncio
component: handlers must await the scheduler's future on the same event loop that
drives it.
"""

from __future__ import annotations

import asyncio
import logging
import signal

import grpc
from prometheus_client import start_http_server

from . import metrics
from .batcher import DynamicBatcher, QueueFullError
from .config import settings
from .model import LLMEngine
from .pb import inference_pb2, inference_pb2_grpc
from .sampling import SamplingParams

log = logging.getLogger("mini-vllm.worker")


def _params(request) -> SamplingParams:
    return SamplingParams(
        max_tokens=request.max_tokens or 50,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p or 1.0,
    )


class InferenceServicer(inference_pb2_grpc.InferenceServicer):
    def __init__(self, batcher: DynamicBatcher):
        self.batcher = batcher

    async def Generate(self, request, context):
        try:
            result = await self.batcher.submit(
                request.prompt, _params(request), request.request_id
            )
        except QueueFullError as exc:
            metrics.record_error("rejected")
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except asyncio.TimeoutError:
            metrics.record_error("timeout")
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "generation timed out")

        return inference_pb2.GenerateResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
            request_id=result.request_id,
        )

    async def GenerateStream(self, request, context):
        try:
            async for delta in self.batcher.stream(
                request.prompt, _params(request), request.request_id
            ):
                yield inference_pb2.Token(text=delta, done=False)
        except QueueFullError as exc:
            metrics.record_error("rejected")
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(exc))
        except asyncio.TimeoutError:
            metrics.record_error("timeout")
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "generation timed out")
        yield inference_pb2.Token(text="", done=True, finish_reason="stop")

    async def Health(self, request, context):
        stats = self.batcher.stats()
        return inference_pb2.HealthResponse(
            ready=True,
            model=self.batcher.engine.model_name,
            running=stats["running"],
            queued=stats["queued"],
            max_batch_size=stats["max_batch_size"],
        )


class MetricsRefresher:
    """Keeps the Prometheus gauges current between scrapes.

    The standalone exporter has no request hook to piggyback on, so scheduler
    depth is sampled on a timer instead.
    """

    def __init__(self, batcher: DynamicBatcher, interval: float = 1.0):
        self.batcher = batcher
        self.interval = interval
        self._task = None

    async def start(self):
        self._task = asyncio.create_task(self._run(), name="metrics-refresher")

    async def _run(self):
        while True:
            metrics.observe_batcher(self.batcher)
            await asyncio.sleep(self.interval)

    async def stop(self):
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


async def serve() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )

    engine = LLMEngine()
    metrics.MODEL_INFO.labels(model=engine.model_name, device=str(engine.device)).set(1)
    batcher = DynamicBatcher(engine)
    await batcher.start()

    refresher = MetricsRefresher(batcher)
    await refresher.start()
    start_http_server(settings.metrics_port, registry=metrics.REGISTRY)
    log.info("metrics on :%d/metrics", settings.metrics_port)

    server = grpc.aio.server(
        options=[
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
            ("grpc.max_send_message_length", 16 * 1024 * 1024),
        ]
    )
    inference_pb2_grpc.add_InferenceServicer_to_server(
        InferenceServicer(batcher), server
    )
    server.add_insecure_port(f"[::]:{settings.grpc_port}")
    await server.start()
    log.info(
        "worker ready: model=%s device=%s grpc=:%d max_batch=%d",
        engine.model_name,
        engine.device,
        settings.grpc_port,
        batcher.max_batch_size,
    )

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:
            # Windows: fall back to KeyboardInterrupt from the outer runner.
            pass

    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        log.info("draining")
        await server.stop(grace=10)
        await refresher.stop()
        await batcher.stop()


def main() -> None:
    try:
        asyncio.run(serve())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
