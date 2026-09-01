"""Unit tests for the cache surgery primitives -- no model required."""

import torch

from server import kvcache


def fake_cache(batch=3, layers=2, heads=4, seq=5, dim=8):
    """A cache whose contents encode their own coordinates, so shuffles are visible."""
    out = []
    for layer in range(layers):
        k = torch.arange(batch, dtype=torch.float32).view(batch, 1, 1, 1).expand(
            batch, heads, seq, dim
        ) + layer * 100
        out.append((k.contiguous(), (k + 0.5).contiguous()))
    return tuple(out)


def test_shape_helpers():
    cache = fake_cache(batch=3, seq=5)
    assert kvcache.batch_size(cache) == 3
    assert kvcache.cache_length(cache) == 5
    assert kvcache.cache_length(()) == 0
    assert kvcache.batch_size(()) == 0


def test_select_rows_keeps_row_identity():
    cache = fake_cache(batch=4)
    picked = kvcache.select_rows(cache, [3, 0])
    assert kvcache.batch_size(picked) == 2
    # Row contents are the original row index, so ordering is checkable.
    assert picked[0][0][0, 0, 0, 0].item() == 3.0
    assert picked[0][0][1, 0, 0, 0].item() == 0.0
    # Second layer carries its +100 offset through the same permutation.
    assert picked[1][0][0, 0, 0, 0].item() == 103.0


def test_select_rows_empty_selection_yields_empty_cache():
    assert kvcache.select_rows(fake_cache(), []) == ()


def test_left_pad_prepends_zeros():
    cache = fake_cache(batch=2, seq=5)
    padded = kvcache.left_pad(cache, 3)
    assert kvcache.cache_length(padded) == 8
    assert torch.all(padded[0][0][:, :, :3, :] == 0)
    # The real positions survive, shifted right.
    assert torch.equal(padded[0][0][:, :, 3:, :], cache[0][0])


def test_left_pad_noop_for_zero():
    cache = fake_cache()
    assert kvcache.left_pad(cache, 0) is cache


def test_concat_batch_aligns_ragged_lengths():
    short = fake_cache(batch=1, seq=3)
    long = fake_cache(batch=2, seq=7)
    merged = kvcache.concat_batch(long, short)
    assert kvcache.batch_size(merged) == 3
    # Everything is padded up to the longest, never truncated down.
    assert kvcache.cache_length(merged) == 7
    # The short sequence's real content sits at the right-hand end.
    assert torch.all(merged[0][0][2, :, :4, :] == 0)


def test_concat_batch_with_empty_side():
    cache = fake_cache()
    assert kvcache.concat_batch((), cache) is cache
    assert kvcache.concat_batch(cache, ()) is cache


def test_trim_left_keeps_most_recent():
    cache = fake_cache(batch=1, seq=10)
    trimmed = kvcache.trim_left(cache, 4)
    assert kvcache.cache_length(trimmed) == 4
    assert torch.equal(trimmed[0][0], cache[0][0][:, :, -4:, :])
    # Asking to keep more than exists is a no-op.
    assert kvcache.trim_left(cache, 99) is cache


def test_cache_bytes_matches_manual_arithmetic():
    cache = fake_cache(batch=2, layers=2, heads=4, seq=5, dim=8)
    per_tensor = 2 * 4 * 5 * 8 * 4  # batch*heads*seq*dim*sizeof(float32)
    assert kvcache.cache_bytes(cache) == per_tensor * 2 * 2  # k+v, two layers


def test_left_pad_sequences_pads_on_the_left():
    ids, mask = kvcache.left_pad_sequences(
        [[1, 2, 3], [9]], pad_id=0, device=torch.device("cpu")
    )
    assert ids.tolist() == [[1, 2, 3], [0, 0, 9]]
    assert mask.tolist() == [[1, 1, 1], [0, 0, 1]]


def test_positions_ignore_left_padding():
    _, mask = kvcache.left_pad_sequences(
        [[1, 2, 3], [9]], pad_id=0, device=torch.device("cpu")
    )
    positions = kvcache.positions_from_mask(mask)
    # The padded row's single real token is position 0, not position 2.
    assert positions.tolist() == [[0, 1, 2], [0, 0, 0]]


def test_legacy_roundtrip_through_transformers_cache():
    cache = fake_cache()
    restored = kvcache.from_cache(kvcache.to_cache(cache))
    assert kvcache.cache_length(restored) == kvcache.cache_length(cache)
    assert torch.equal(restored[0][0], cache[0][0])
    assert kvcache.to_cache(()) is None
