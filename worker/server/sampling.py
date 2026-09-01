"""Logits -> next token. Greedy when temperature is 0, else top-k/top-p sampling."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .config import settings


@dataclass
class SamplingParams:
    max_tokens: int = 50
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0

    def clamp(self) -> "SamplingParams":
        """Bound client-supplied values to what the engine will actually honour."""
        return SamplingParams(
            max_tokens=max(1, min(int(self.max_tokens), settings.max_new_tokens)),
            temperature=max(0.0, min(float(self.temperature), 2.0)),
            top_k=max(0, int(self.top_k)),
            top_p=min(max(float(self.top_p), 0.0), 1.0) or 1.0,
        )


def sample(logits: torch.Tensor, params: SamplingParams) -> torch.Tensor:
    """Pick one token per row from ``[batch, vocab]`` logits."""
    if params.temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / params.temperature

    if params.top_k > 0:
        k = min(params.top_k, logits.shape[-1])
        kth = logits.topk(k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if params.top_p < 1.0:
        ordered, order = logits.sort(dim=-1, descending=True)
        cumulative = ordered.softmax(dim=-1).cumsum(dim=-1)
        # Keep everything up to and including the token that crosses top_p.
        drop = cumulative - ordered.softmax(dim=-1) > params.top_p
        ordered = ordered.masked_fill(drop, float("-inf"))
        logits = torch.empty_like(logits).scatter_(-1, order, ordered)

    return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)


def sample_batch(logits: torch.Tensor, params: list[SamplingParams]) -> torch.Tensor:
    """Sample a ragged batch where each row carries its own parameters.

    Rows sharing identical parameters are sampled in one vectorised call; only
    genuinely distinct settings cost an extra pass.
    """
    if not params:
        return torch.empty(0, dtype=torch.long, device=logits.device)

    groups: dict[tuple, list[int]] = {}
    for row, p in enumerate(params):
        groups.setdefault((p.temperature, p.top_k, p.top_p), []).append(row)

    out = torch.empty(logits.shape[0], dtype=torch.long, device=logits.device)
    for (temperature, top_k, top_p), rows in groups.items():
        idx = torch.as_tensor(rows, dtype=torch.long, device=logits.device)
        out[idx] = sample(
            logits.index_select(0, idx),
            SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p),
        )
    return out
