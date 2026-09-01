"""HTTP surface for the worker.

Exposes an OpenAI-shaped ``/v1/completions`` so existing clients and load
generators work unmodified, plus ``/metrics`` for Prometheus and ``/stats`` for
eyeballing scheduler state during a load test.

The model is loaded once in the lifespan hook rather than at import time, so that
importing this module (in tests, or under ``--reload``) does not pull 500 MB of
weights off disk.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from . import metrics
from .batcher import DynamicBatcher, QueueFullError
from .config import settings
from .model import LLMEngine
from .sampling import SamplingParams


class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=50, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_k: int = Field(default=0, ge=0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    stream: bool = False

    def sampling(self) -> SamplingParams:
        return SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    text: str
    finish_reason: str
    usage: Usage


@asynccontextmanager
async def lifespan(app: FastAPI):
    engine = LLMEngine()
    batcher = DynamicBatcher(engine)
    await batcher.start()
    app.state.engine = engine
    app.state.batcher = batcher
    metrics.MODEL_INFO.labels(
        model=engine.model_name, device=str(engine.device)
    ).set(1)
    try:
        yield
    finally:
        await batcher.stop()


app = FastAPI(title="Mini-vLLM", version="0.1.0", lifespan=lifespan)


def get_batcher(request: Request) -> DynamicBatcher:
    batcher: Optional[DynamicBatcher] = getattr(request.app.state, "batcher", None)
    if batcher is None:
        raise HTTPException(status_code=503, detail="engine is still starting")
    return batcher


@app.post("/v1/completions", response_model=None)
async def completions(
    body: CompletionRequest, batcher: DynamicBatcher = Depends(get_batcher)
):
    request_id = f"cmpl-{uuid.uuid4().hex[:16]}"
    if body.stream:
        return StreamingResponse(
            _sse(batcher, body, request_id),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = await batcher.submit(body.prompt, body.sampling(), request_id)
    except QueueFullError as exc:
        metrics.record_error("rejected")
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except asyncio.TimeoutError as exc:
        metrics.record_error("timeout")
        raise HTTPException(status_code=504, detail="generation timed out") from exc

    metrics.record_result(result)
    return CompletionResponse(
        id=request_id,
        created=int(time.time()),
        model=batcher.engine.model_name,
        text=result.text,
        finish_reason=result.finish_reason,
        usage=Usage(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
    )


async def _sse(
    batcher: DynamicBatcher, body: CompletionRequest, request_id: str
) -> AsyncIterator[str]:
    """Server-sent events, terminated with OpenAI's ``[DONE]`` sentinel."""
    try:
        async for delta in batcher.stream(body.prompt, body.sampling(), request_id):
            payload = {"id": request_id, "text": delta, "finish_reason": None}
            yield f"data: {json.dumps(payload)}\n\n"
    except QueueFullError as exc:
        metrics.record_error("rejected")
        yield f"data: {json.dumps({'error': str(exc)})}\n\n"
        yield "data: [DONE]\n\n"
        return
    except asyncio.CancelledError:
        # Client hung up; the scheduler drops the sequence on the next sweep.
        raise
    yield f"data: {json.dumps({'id': request_id, 'text': '', 'finish_reason': 'stop'})}\n\n"
    yield "data: [DONE]\n\n"


@app.get("/health")
async def health(request: Request):
    batcher = getattr(request.app.state, "batcher", None)
    if batcher is None:
        return JSONResponse({"ready": False}, status_code=503)
    return {
        "ready": True,
        "model": batcher.engine.model_name,
        "device": str(batcher.engine.device),
        **batcher.stats(),
    }


@app.get("/stats")
async def stats(batcher: DynamicBatcher = Depends(get_batcher)):
    return batcher.stats()


@app.get("/metrics")
async def prometheus_metrics(request: Request):
    batcher = getattr(request.app.state, "batcher", None)
    if batcher is not None:
        metrics.observe_batcher(batcher)
    return Response(generate_latest(metrics.REGISTRY), media_type=CONTENT_TYPE_LATEST)
