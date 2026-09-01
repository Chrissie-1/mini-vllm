"""Runtime settings, overridable by environment variable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return default if raw is None or raw == "" else int(raw)


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return default if raw is None or raw == "" else float(raw)


@dataclass
class Settings:
    # Model
    model_name: str = field(default_factory=lambda: _env_str("MODEL_NAME", "gpt2"))
    device: str = field(default_factory=lambda: _env_str("DEVICE", "cpu"))
    torch_threads: int = field(default_factory=lambda: _env_int("TORCH_THREADS", 0))

    # Scheduler
    max_batch_size: int = field(default_factory=lambda: _env_int("MAX_BATCH_SIZE", 8))
    batch_window_ms: int = field(default_factory=lambda: _env_int("BATCH_WINDOW_MS", 20))
    max_queue_size: int = field(default_factory=lambda: _env_int("MAX_QUEUE_SIZE", 256))
    request_timeout_s: float = field(
        default_factory=lambda: _env_float("REQUEST_TIMEOUT_S", 120.0)
    )

    # Generation caps
    max_new_tokens: int = field(default_factory=lambda: _env_int("MAX_NEW_TOKENS", 256))
    max_context: int = field(default_factory=lambda: _env_int("MAX_CONTEXT", 1024))

    # Serving
    http_port: int = field(default_factory=lambda: _env_int("HTTP_PORT", 8000))
    grpc_port: int = field(default_factory=lambda: _env_int("GRPC_PORT", 50051))
    metrics_port: int = field(default_factory=lambda: _env_int("METRICS_PORT", 9100))


settings = Settings()
