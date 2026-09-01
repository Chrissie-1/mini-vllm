"""HTTP contract tests.

The app is wired to the session's already-loaded engine rather than letting the
lifespan hook load its own copy -- a second set of weights is a needless half
gigabyte, and it makes the response assertions checkable against a known engine.
"""

import json

import pytest
from fastapi.testclient import TestClient

from server import api


@pytest.fixture
def client(engine, monkeypatch):
    monkeypatch.setattr(api, "LLMEngine", lambda *a, **k: engine)
    with TestClient(api.app) as c:
        yield c


def test_completion_returns_generated_text(client, engine):
    response = client.post(
        "/v1/completions", json={"prompt": "Once upon a time", "max_tokens": 12}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["text"] == engine.generate_with_kv_cache(
        "Once upon a time", max_new_tokens=12
    )
    assert body["finish_reason"] == "length"
    assert body["object"] == "text_completion"
    assert body["id"].startswith("cmpl-")


def test_usage_accounting_adds_up(client):
    body = client.post(
        "/v1/completions", json={"prompt": "The capital of France is", "max_tokens": 8}
    ).json()
    usage = body["usage"]
    assert usage["completion_tokens"] == 8
    assert usage["prompt_tokens"] > 0
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


def test_streaming_reassembles_to_the_blocking_response(client, engine):
    payload = {"prompt": "Once upon a time", "max_tokens": 10, "stream": True}
    with client.stream("POST", "/v1/completions", json=payload) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            line[len("data: "):]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert events[-1] == "[DONE]"
    text = "".join(json.loads(e).get("text", "") for e in events[:-1])
    assert text == engine.generate_with_kv_cache("Once upon a time", max_new_tokens=10)


def test_invalid_max_tokens_is_rejected(client):
    response = client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 0})
    assert response.status_code == 422


def test_missing_prompt_is_rejected(client):
    assert client.post("/v1/completions", json={"max_tokens": 5}).status_code == 422


def test_temperature_out_of_range_is_rejected(client):
    response = client.post(
        "/v1/completions", json={"prompt": "hi", "max_tokens": 5, "temperature": 9.0}
    )
    assert response.status_code == 422


def test_health_reports_readiness(client, engine):
    body = client.get("/health").json()
    assert body["ready"] is True
    assert body["model"] == engine.model_name
    assert "max_batch_size" in body


def test_stats_expose_scheduler_counters(client):
    client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
    stats = client.get("/stats").json()
    assert stats["completed"] >= 1
    assert stats["tokens_generated"] >= 3
    assert stats["mean_batch_size"] > 0


def test_metrics_are_exported_in_prometheus_format(client):
    client.post("/v1/completions", json={"prompt": "hi", "max_tokens": 3})
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]

    body = response.text
    for metric in (
        "llm_requests_total",
        "llm_latency_seconds",
        "llm_time_to_first_token_seconds",
        "llm_queue_wait_seconds",
        "llm_tokens_generated_total",
        "llm_batch_size",
        "llm_kv_cache_bytes",
        "llm_kv_cache_reused_positions_total",
        "llm_kv_cache_attended_positions_total",
        "llm_model_info",
    ):
        assert metric in body, f"{metric} missing from /metrics"


def test_cache_hit_counters_advance_during_decoding(client):
    def read(name):
        for line in client.get("/metrics").text.splitlines():
            if line.startswith(name + " "):
                return float(line.split()[1])
        return 0.0

    before = read("llm_kv_cache_reused_positions_total")
    client.post("/v1/completions", json={"prompt": "Once upon a time", "max_tokens": 10})
    # Ten decode steps, each reusing every previously cached position.
    assert read("llm_kv_cache_reused_positions_total") > before
