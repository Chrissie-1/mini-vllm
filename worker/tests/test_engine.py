"""Engine correctness.

The load-bearing tests are the equivalence ones. An optimisation that changes the
output is a bug, so every fast path is pinned against a slower path that is
obviously correct:

* cached decoding == full recomputation every step
* batched decoding == running each prompt alone
* a sequence's output is unchanged by which other sequences share its batch
"""

import pytest
import torch

from server import kvcache
from server.model import BatchState
from server.sampling import SamplingParams

GREEDY = SamplingParams(temperature=0.0)


def test_kv_cache_matches_uncached_recomputation(engine):
    """The headline claim: caching is a pure speed optimisation, not an approximation."""
    prompt = "The capital of France is"
    cached = engine.generate_with_kv_cache(prompt, max_new_tokens=25)
    uncached = engine.generate_without_cache(prompt, max_new_tokens=25)
    assert cached == uncached


def test_batched_output_matches_single_sequence(engine):
    """Left padding must not perturb a shorter sequence's positions."""
    prompts = [
        "Once upon a time",
        "In a distant galaxy far beyond the reach of any telescope, there",
        "def fibonacci(n):",
    ]
    batched = engine.generate_batch(prompts, max_new_tokens=20)
    for prompt, expected in zip(prompts, batched):
        assert engine.generate_with_kv_cache(prompt, max_new_tokens=20) == expected


def test_output_is_independent_of_batch_composition(engine):
    """A sequence must not be able to attend to its neighbours' padding."""
    alone = engine.generate_batch(["Once upon a time"], max_new_tokens=15)[0]
    with_neighbours = engine.generate_batch(
        ["Once upon a time", "x" * 4, "The quick brown fox jumps over the lazy dog and"],
        max_new_tokens=15,
    )[0]
    assert alone == with_neighbours


def test_prefill_emits_exactly_one_token(engine):
    seqs = [engine.make_sequence("Hello world", GREEDY)]
    state = engine.prefill(seqs)
    assert len(seqs[0].output_ids) == 1
    assert kvcache.cache_length(state.cache) == len(seqs[0].prompt_ids)
    assert state.next_tokens.shape == (1, 1)


def test_decode_step_grows_cache_by_one(engine):
    state = engine.prefill([engine.make_sequence("Hello world", GREEDY)])
    before = kvcache.cache_length(state.cache)
    engine.decode_step(state)
    assert kvcache.cache_length(state.cache) == before + 1


def test_max_tokens_is_respected(engine):
    seq = engine.make_sequence("Count: 1 2 3", SamplingParams(max_tokens=7))
    state = engine.prefill([seq])
    while not seq.finished:
        engine.decode_step(state)
    assert len(seq.output_ids) == 7
    assert seq.finish_reason == "length"


def _logits_forcing(engine, token_id, rows=1):
    """Logits whose argmax is ``token_id``, to drive the stopping rules directly."""
    logits = torch.full((rows, engine.model.config.vocab_size), -1e9)
    logits[:, token_id] = 1.0
    return logits


def test_eos_stops_generation(engine):
    seq = engine.make_sequence("hi", SamplingParams(max_tokens=50))
    state = BatchState(seqs=[seq])
    engine._emit(state, _logits_forcing(engine, engine.eos_token_id))

    assert seq.output_ids == [engine.eos_token_id]
    assert seq.finish_reason == "stop"


def test_finished_rows_are_not_extended_by_later_steps(engine):
    """Once a row is done its token list must freeze, even while the batch runs on."""
    seq = engine.make_sequence("hi", SamplingParams(max_tokens=1))
    state = BatchState(seqs=[seq])
    engine._emit(state, _logits_forcing(engine, 500))
    assert seq.finish_reason == "length"

    engine._emit(state, _logits_forcing(engine, 501))
    assert seq.output_ids == [500]


def test_context_limit_stops_generation(engine):
    """A sequence that reaches the model's context window stops on 'length'."""
    seq = engine.make_sequence("hi", SamplingParams(max_tokens=10**6))
    seq.prompt_ids = list(range(engine.max_context - 1))
    state = BatchState(seqs=[seq])
    engine._emit(state, _logits_forcing(engine, 500))
    assert seq.finish_reason == "length"


def test_evict_finished_shrinks_batch_and_cache(engine):
    """Finishing one row must remove exactly that row from the KV cache."""
    seqs = [
        engine.make_sequence("Once upon a time", SamplingParams(max_tokens=2)),
        engine.make_sequence("The capital of France is", SamplingParams(max_tokens=30)),
    ]
    state = engine.prefill(seqs)
    state = engine.decode_step(state)  # short sequence hits its limit here

    assert seqs[0].finished and not seqs[1].finished
    state, done = engine.evict_finished(state)

    assert [s.seq_id for s in done] == [seqs[0].seq_id]
    assert len(state) == 1
    assert kvcache.batch_size(state.cache) == 1
    assert state.mask.shape[0] == 1
    assert state.seqs[0].seq_id == seqs[1].seq_id


def test_eviction_preserves_the_survivors_output(engine):
    """The survivor must continue exactly as if it had run alone throughout."""
    long_prompt = "The capital of France is"
    solo = engine.generate_with_kv_cache(long_prompt, max_new_tokens=12)

    seqs = [
        engine.make_sequence("hello there", SamplingParams(max_tokens=3)),
        engine.make_sequence(long_prompt, SamplingParams(max_tokens=12)),
    ]
    state = engine.prefill(seqs)
    while not all(s.finished for s in seqs):
        state, _ = engine.evict_finished(state)
        if state.empty:
            break
        state = engine.decode_step(state)

    assert engine.decode(seqs[1].output_ids) == solo


def test_admit_splices_a_new_sequence_into_a_running_batch(engine):
    """Continuous batching: a latecomer joins mid-flight and is unaffected by it."""
    latecomer_prompt = "def fibonacci(n):"
    solo = engine.generate_with_kv_cache(latecomer_prompt, max_new_tokens=10)

    running = engine.make_sequence(
        "The capital of France is", SamplingParams(max_tokens=30)
    )
    state = engine.prefill([running])
    for _ in range(5):  # let the batch get ahead before admitting
        state = engine.decode_step(state)

    latecomer = engine.make_sequence(latecomer_prompt, SamplingParams(max_tokens=10))
    state = engine.admit(state, [latecomer])
    assert len(state) == 2
    assert kvcache.batch_size(state.cache) == 2

    while not latecomer.finished:
        state = engine.decode_step(state)
    assert engine.decode(latecomer.output_ids) == solo


def test_admit_into_empty_state_is_a_plain_prefill(engine):
    seq = engine.make_sequence("Hello", GREEDY)
    state = engine.admit(BatchState(), [seq])
    assert len(state) == 1
    assert len(seq.output_ids) == 1


def test_admit_with_no_newcomers_returns_the_same_state(engine):
    state = engine.prefill([engine.make_sequence("Hello", GREEDY)])
    assert engine.admit(state, []) is state


def test_empty_prompt_is_handled(engine):
    """An empty prompt has no final position to read logits from; it must not crash."""
    assert isinstance(engine.generate_with_kv_cache("", max_new_tokens=3), str)


def test_prompt_longer_than_context_is_truncated(engine):
    seq = engine.make_sequence("word " * 4000, GREEDY)
    assert len(seq.prompt_ids) <= engine.max_context - 1


def test_sampling_params_are_clamped(engine):
    seq = engine.make_sequence(
        "hi", SamplingParams(max_tokens=10**6, temperature=99.0, top_p=0.0)
    )
    assert seq.params.max_tokens <= engine.max_context
    assert seq.params.temperature <= 2.0
    assert 0.0 < seq.params.top_p <= 1.0


@pytest.mark.parametrize(
    "params",
    [
        SamplingParams(max_tokens=8, temperature=0.8, top_k=50),
        SamplingParams(max_tokens=8, temperature=0.8, top_p=0.9),
        SamplingParams(max_tokens=8, temperature=1.0, top_k=10, top_p=0.95),
    ],
)
def test_stochastic_sampling_stays_in_vocabulary(engine, params):
    torch.manual_seed(0)
    seq = engine.make_sequence("Once upon a time", params)
    state = engine.prefill([seq])
    while not seq.finished:
        state = engine.decode_step(state)
    assert len(seq.output_ids) <= 8
    assert all(0 <= t < engine.model.config.vocab_size for t in seq.output_ids)


def test_mixed_sampling_params_in_one_batch(engine):
    """Rows with different temperatures must be sampled with their own settings."""
    torch.manual_seed(0)
    seqs = [
        engine.make_sequence("Once upon a time", SamplingParams(max_tokens=6)),
        engine.make_sequence(
            "Once upon a time", SamplingParams(max_tokens=6, temperature=1.5, top_k=100)
        ),
    ]
    state = engine.prefill(seqs)
    while not all(s.finished for s in seqs):
        state, _ = engine.evict_finished(state)
        if state.empty:
            break
        state = engine.decode_step(state)
    # The greedy row must match greedy decoding done on its own.
    assert engine.decode(seqs[0].output_ids) == engine.generate_with_kv_cache(
        "Once upon a time", max_new_tokens=6
    )
