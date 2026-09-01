"""The inference engine: KV-cached prefill/decode over a mutable batch.

Two ideas carry the whole file.

**KV caching.** A decoder-only transformer re-derives the same keys and values for
every prior token on every step. Caching them turns each decode step from a
forward pass over the whole sequence into a forward pass over one token, so
generating ``n`` tokens costs ``O(n)`` passes over a growing cache instead of
``O(n^2)`` recomputation. (Attention itself still scans the cache, so a single
step is ``O(n)`` in the cache length, not ``O(1)`` -- the saving is the
elimination of redundant *recomputation*, and it is large: see
``bench/bench_kv_cache.py``.)

**Continuous batching.** Sequences in a batch finish at different times. Rather
than idling finished rows until the slowest one is done, the engine evicts them
from the batch -- and from the KV cache -- and admits waiting requests into the
freed slots. The batch is left-padded so every row's newest token sits in the
same column, which is what lets one ``[batch, 1]`` forward pass advance all of
them together.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import kvcache
from .config import settings
from .sampling import SamplingParams, sample_batch

_seq_counter = itertools.count(1)


@dataclass
class Seq:
    """One in-flight generation request."""

    prompt_ids: List[int]
    params: SamplingParams = field(default_factory=SamplingParams)
    request_id: str = ""
    seq_id: int = field(default_factory=lambda: next(_seq_counter))
    output_ids: List[int] = field(default_factory=list)
    finish_reason: Optional[str] = None
    arrived_at: float = field(default_factory=time.monotonic)
    first_token_at: Optional[float] = None

    @property
    def finished(self) -> bool:
        return self.finish_reason is not None

    @property
    def total_length(self) -> int:
        return len(self.prompt_ids) + len(self.output_ids)


@dataclass
class BatchState:
    """A batch mid-flight: its sequences, its KV cache, and its attention mask.

    Invariant: ``len(seqs) == mask.shape[0] == batch dim of cache``, and row ``i``
    of both the mask and the cache belongs to ``seqs[i]``.
    """

    seqs: List[Seq] = field(default_factory=list)
    cache: kvcache.LegacyCache = ()
    mask: Optional[torch.Tensor] = None
    next_tokens: Optional[torch.Tensor] = None  # [batch, 1]

    def __len__(self) -> int:
        return len(self.seqs)

    @property
    def empty(self) -> bool:
        return not self.seqs


class LLMEngine:
    """Loads the model and drives prefill/decode over a :class:`BatchState`."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_context: Optional[int] = None,
    ):
        self.model_name = model_name or settings.model_name
        self.device = torch.device(device or settings.device)
        self.max_context = max_context or settings.max_context

        if settings.torch_threads > 0:
            torch.set_num_threads(settings.torch_threads)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding keeps every row's last real token in the final column.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        window = getattr(self.model.config, "n_positions", None) or getattr(
            self.model.config, "max_position_embeddings", self.max_context
        )
        self.max_context = min(self.max_context, int(window))

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def encode(self, prompt: str) -> List[int]:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            # An empty prompt has no last position to read logits from; seed with EOS.
            ids = [self.eos_token_id]
        # Leave room for at least one generated token.
        return ids[-(self.max_context - 1):]

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def make_sequence(
        self,
        prompt: str,
        params: Optional[SamplingParams] = None,
        request_id: str = "",
    ) -> Seq:
        return Seq(
            prompt_ids=self.encode(prompt),
            params=(params or SamplingParams()).clamp(),
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Prefill / decode
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def prefill(self, seqs: List[Seq]) -> BatchState:
        """One forward pass over the prompts; fills the cache and emits token #1."""
        if not seqs:
            return BatchState()

        input_ids, mask = kvcache.left_pad_sequences(
            [s.prompt_ids for s in seqs], self.pad_token_id, self.device
        )
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=mask,
            position_ids=kvcache.positions_from_mask(mask),
            use_cache=True,
        )
        state = BatchState(
            seqs=list(seqs),
            cache=kvcache.from_cache(outputs.past_key_values),
            mask=mask,
        )
        self._emit(state, outputs.logits[:, -1, :])
        return state

    @torch.inference_mode()
    def decode_step(self, state: BatchState) -> BatchState:
        """Advance every row by exactly one token, reusing the cache."""
        if state.empty:
            return state

        # The mask grows by one column for the token we are about to consume;
        # its position is "how many real tokens precede it".
        mask = torch.cat(
            [
                state.mask,
                torch.ones((len(state), 1), dtype=torch.long, device=self.device),
            ],
            dim=1,
        )
        position_ids = mask.sum(dim=1, keepdim=True) - 1

        outputs = self.model(
            input_ids=state.next_tokens,
            attention_mask=mask,
            position_ids=position_ids,
            past_key_values=kvcache.to_cache(state.cache),
            use_cache=True,
        )
        state.cache = kvcache.from_cache(outputs.past_key_values)
        state.mask = mask
        self._emit(state, outputs.logits[:, -1, :])
        return state

    def _emit(self, state: BatchState, logits: torch.Tensor) -> None:
        """Sample one token per row, append it, and apply the stopping rules."""
        tokens = sample_batch(logits, [s.params for s in state.seqs])
        now = time.monotonic()
        for row, seq in enumerate(state.seqs):
            if seq.finished:
                continue
            token = int(tokens[row].item())
            seq.output_ids.append(token)
            if seq.first_token_at is None:
                seq.first_token_at = now
            if token == self.eos_token_id:
                seq.finish_reason = "stop"
            elif len(seq.output_ids) >= seq.params.max_tokens:
                seq.finish_reason = "length"
            elif seq.total_length >= self.max_context:
                seq.finish_reason = "length"
        state.next_tokens = tokens.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Batch mutation -- the continuous-batching primitives
    # ------------------------------------------------------------------

    def evict_finished(self, state: BatchState) -> Tuple[BatchState, List[Seq]]:
        """Split finished sequences out of the batch, shrinking cache and mask."""
        if state.empty:
            return state, []
        keep = [i for i, s in enumerate(state.seqs) if not s.finished]
        done = [s for s in state.seqs if s.finished]
        if not done:
            return state, []
        if not keep:
            return BatchState(), done

        idx = torch.as_tensor(keep, dtype=torch.long, device=self.device)
        survivors = BatchState(
            seqs=[state.seqs[i] for i in keep],
            cache=kvcache.select_rows(state.cache, keep),
            mask=state.mask.index_select(0, idx),
            next_tokens=state.next_tokens.index_select(0, idx),
        )
        # Rows may now share a common run of left padding; reclaim it.
        return self._compact(survivors), done

    def _compact(self, state: BatchState) -> BatchState:
        """Drop leading columns that are padding for *every* remaining row."""
        if state.empty or state.mask is None:
            return state
        lead = int((state.mask.cumsum(dim=1) == 0).all(dim=0).sum().item())
        if lead == 0:
            return state
        state.mask = state.mask[:, lead:]
        state.cache = tuple(
            (k[:, :, lead:, :], v[:, :, lead:, :]) for k, v in state.cache
        )
        return state

    @torch.inference_mode()
    def admit(self, state: BatchState, newcomers: List[Seq]) -> BatchState:
        """Prefill ``newcomers`` and splice them into a running batch."""
        if not newcomers:
            return state
        fresh = self.prefill(newcomers)
        if state.empty:
            return fresh

        # Align the two masks on the sequence axis before concatenating.
        width = max(state.mask.shape[1], fresh.mask.shape[1])
        return BatchState(
            seqs=state.seqs + fresh.seqs,
            cache=kvcache.concat_batch(state.cache, fresh.cache),
            mask=torch.cat(
                [_pad_mask_left(state.mask, width), _pad_mask_left(fresh.mask, width)],
                dim=0,
            ),
            next_tokens=torch.cat([state.next_tokens, fresh.next_tokens], dim=0),
        )

    # ------------------------------------------------------------------
    # Convenience entry points
    # ------------------------------------------------------------------

    def generate_with_kv_cache(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        params: Optional[SamplingParams] = None,
    ) -> str:
        """Single-prompt generation. Phase 1's public surface."""
        params = params or SamplingParams()
        params.max_tokens = max_new_tokens
        seq = self.make_sequence(prompt, params)
        state = self.prefill([seq])
        while not seq.finished:
            state = self.decode_step(state)
        return self.decode(seq.output_ids)

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 50,
        params: Optional[SamplingParams] = None,
    ) -> List[str]:
        """Static batch: all prompts start together, run until all are done."""
        base = params or SamplingParams()
        seqs = [
            self.make_sequence(
                p,
                SamplingParams(
                    max_tokens=max_new_tokens,
                    temperature=base.temperature,
                    top_k=base.top_k,
                    top_p=base.top_p,
                ),
            )
            for p in prompts
        ]
        state = self.prefill(seqs)
        while not state.empty:
            state, _ = self.evict_finished(state)
            if state.empty:
                break
            state = self.decode_step(state)
        return [self.decode(s.output_ids) for s in seqs]

    @torch.inference_mode()
    def generate_without_cache(self, prompt: str, max_new_tokens: int = 50) -> str:
        """Deliberately naive baseline: re-run the full sequence every step.

        Kept only so the benchmark can measure what the cache actually buys.
        """
        ids = self.encode(prompt)
        generated: List[int] = []
        for _ in range(max_new_tokens):
            window = (ids + generated)[-self.max_context:]
            logits = self.model(
                input_ids=torch.tensor([window], dtype=torch.long, device=self.device),
                use_cache=False,
            ).logits
            token = int(logits[0, -1, :].argmax(dim=-1).item())
            generated.append(token)
            if token == self.eos_token_id:
                break
        return self.decode(generated)

    # ------------------------------------------------------------------

    def cache_stats(self, state: BatchState) -> dict:
        return {
            "batch_size": len(state),
            "cache_len": kvcache.cache_length(state.cache),
            "cache_bytes": kvcache.cache_bytes(state.cache),
        }


def _pad_mask_left(mask: torch.Tensor, width: int) -> torch.Tensor:
    pad = width - mask.shape[1]
    if pad <= 0:
        return mask
    return torch.cat(
        [
            torch.zeros((mask.shape[0], pad), dtype=mask.dtype, device=mask.device),
            mask,
        ],
        dim=1,
    )
