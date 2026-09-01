"""Batch-dimension surgery on transformer KV caches.

A KV cache is a per-layer pair of tensors shaped ``[batch, heads, seq, head_dim]``.
Continuous batching means the batch dimension is *mutable at runtime*: finished
sequences are evicted mid-flight and freshly prefilled ones are spliced in.
Neither operation is exposed by ``transformers``, so the primitives live here.

Everything below operates on the "legacy" tuple-of-pairs representation, which is
stable across ``transformers`` releases. :func:`to_cache` / :func:`from_cache`
convert at the model boundary.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

# tuple(layers) of (key, value), each [batch, heads, seq, head_dim]
LegacyCache = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

try:  # transformers >= 4.36 hands back a Cache object
    from transformers.cache_utils import DynamicCache

    _HAS_DYNAMIC_CACHE = True
except ImportError:  # pragma: no cover - very old transformers
    DynamicCache = None  # type: ignore[assignment]
    _HAS_DYNAMIC_CACHE = False


def from_cache(past_key_values) -> LegacyCache:
    """Normalise whatever the model returned into the legacy tuple form."""
    if past_key_values is None:
        return ()
    if _HAS_DYNAMIC_CACHE and isinstance(past_key_values, DynamicCache):
        return tuple(past_key_values.to_legacy_cache())
    return tuple((k, v) for k, v in past_key_values)


def to_cache(legacy: LegacyCache):
    """Convert back into the object the model expects for ``past_key_values``."""
    if not legacy:
        return None
    if _HAS_DYNAMIC_CACHE:
        return DynamicCache.from_legacy_cache(legacy)
    return legacy


def cache_length(legacy: LegacyCache) -> int:
    """Number of cached positions (the sequence axis)."""
    return 0 if not legacy else legacy[0][0].shape[2]


def batch_size(legacy: LegacyCache) -> int:
    return 0 if not legacy else legacy[0][0].shape[0]


def select_rows(legacy: LegacyCache, rows: Sequence[int]) -> LegacyCache:
    """Keep only ``rows`` of the batch — this is how a finished sequence is evicted.

    Uses ``index_select`` rather than fancy indexing so the result is a compact
    tensor rather than a view holding the whole original batch alive.
    """
    if not legacy:
        return ()
    if not rows:
        return ()
    idx = torch.as_tensor(list(rows), dtype=torch.long, device=legacy[0][0].device)
    return tuple(
        (k.index_select(0, idx), v.index_select(0, idx)) for k, v in legacy
    )


def left_pad(legacy: LegacyCache, pad: int) -> LegacyCache:
    """Prepend ``pad`` zeroed positions to the sequence axis.

    Left padding (rather than right) is what keeps a ragged batch aligned: every
    sequence's *last* position lands at the same index, so a decode step is a
    single ``[batch, 1]`` forward pass. The padded slots are masked out by the
    attention mask, so their contents never reach the softmax.
    """
    if pad <= 0 or not legacy:
        return legacy
    out = []
    for k, v in legacy:
        b, h, _, d = k.shape
        zeros_k = torch.zeros(b, h, pad, d, dtype=k.dtype, device=k.device)
        zeros_v = torch.zeros(b, h, pad, d, dtype=v.dtype, device=v.device)
        out.append((torch.cat([zeros_k, k], dim=2), torch.cat([zeros_v, v], dim=2)))
    return tuple(out)


def concat_batch(a: LegacyCache, b: LegacyCache) -> LegacyCache:
    """Stack two caches along the batch axis, left-padding the shorter one.

    This is the admission path: an in-flight batch at length 40 absorbing a newly
    prefilled sequence of length 12 pads the newcomer to 40 and concatenates.
    """
    if not a:
        return b
    if not b:
        return a
    la, lb = cache_length(a), cache_length(b)
    if la < lb:
        a = left_pad(a, lb - la)
    elif lb < la:
        b = left_pad(b, la - lb)
    return tuple(
        (torch.cat([ka, kb], dim=0), torch.cat([va, vb], dim=0))
        for (ka, va), (kb, vb) in zip(a, b)
    )


def trim_left(legacy: LegacyCache, keep: int) -> LegacyCache:
    """Drop the oldest positions, keeping the most recent ``keep``.

    Bounds cache growth once a batch approaches the model's context window.
    """
    if not legacy or keep >= cache_length(legacy):
        return legacy
    return tuple((k[:, :, -keep:, :], v[:, :, -keep:, :]) for k, v in legacy)


def cache_bytes(legacy: LegacyCache) -> int:
    """Resident size of the cache, for the memory gauge."""
    return sum(
        k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in legacy
    )


def left_pad_sequences(
    sequences: List[List[int]], pad_id: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad token id lists into ``(input_ids, attention_mask)``."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(sequences), width), dtype=torch.long, device=device)
    for row, seq in enumerate(sequences):
        ids[row, width - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=device)
        mask[row, width - len(seq) :] = 1
    return ids, mask


def positions_from_mask(mask: torch.Tensor) -> torch.Tensor:
    """Position ids that ignore left padding.

    With left padding the raw column index overstates every real token's position,
    which would corrupt the rotary/learned positional embeddings. Counting only
    unmasked tokens restores the true position.
    """
    return (mask.cumsum(dim=-1) - 1).clamp(min=0)
