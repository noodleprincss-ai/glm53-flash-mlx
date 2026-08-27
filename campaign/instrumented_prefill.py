"""Source-pinned pure-prefill instrumentation for mlx-vlm 0.6.17.

This module intentionally lives beside the campaign harness, not in the pinned
runtime or installed mlx-vlm package.  The prompt loop mirrors the installed
``mlx_vlm.generate.ar.generate_step`` at the pinned source hash.  Its only
additional synchronization is the final benchmark boundary after the last
prompt forward and before sampling.
"""

from __future__ import annotations

import functools
import hashlib
import importlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx

PINNED_AR_SHA256 = "7076aedecab82c42cdaf520f6e1e02466ccf0f22c99af82d9ce4a414f4bd9fab"


class SourcePinError(RuntimeError):
    """Raised when the installed mlx-vlm generation source is not the pin."""


@dataclass(frozen=True)
class ChunkTiming:
    kind: str
    token_start: int
    token_stop: int
    seconds: float


@dataclass
class PrefillResult:
    source_path: str
    source_sha256: str
    effective_prompt_tokens: int
    prefill_seconds: float
    prompt_rate_tokens_per_second: float
    chunk_timings: list[ChunkTiming]
    final_prompt_logits: mx.array
    final_prompt_logprobs: mx.array
    first_token_id: int
    generated_token_ids: list[int]
    generated_logits: list[mx.array]
    generated_logprobs: list[mx.array]
    prompt_cache: list[Any]


def assert_pinned_ar_source() -> tuple[Any, Path, str]:
    """Import and verify the exact installed source copied by this driver."""
    ar = importlib.import_module("mlx_vlm.generate.ar")
    path = Path(ar.__file__).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != PINNED_AR_SHA256:
        raise SourcePinError(
            f"mlx-vlm ar.py hash {digest} at {path} does not match "
            f"campaign pin {PINNED_AR_SHA256}"
        )
    return ar, path, digest


def _eval_cache(prompt_cache: list[Any]) -> None:
    mx.eval([entry.state for entry in prompt_cache])


def instrumented_prefill(
    input_ids: mx.array,
    model: Any,
    pixel_values: Any = None,
    mask: Any = None,
    *,
    temperature: float = 0.0,
    repetition_penalty: float | None = None,
    repetition_context_size: int | None = 20,
    presence_penalty: float | None = None,
    presence_context_size: int | None = 20,
    frequency_penalty: float | None = None,
    frequency_context_size: int | None = 20,
    top_p: float = 1.0,
    min_p: float = 0.0,
    top_k: int = 0,
    top_n_sigma: float = 0.0,
    p_less: bool = False,
    typical_p: float = 1.0,
    logit_bias: dict[int, float] | None = None,
    prompt_cache: list[Any] | None = None,
    max_kv_size: int | None = None,
    kv_bits: float | None = None,
    kv_key_bits: float | None = None,
    kv_value_bits: float | None = None,
    kv_key_scheme: str | None = None,
    kv_value_scheme: str | None = None,
    kv_group_size: int = 64,
    kv_quant_scheme: str = "affine",
    quantized_kv_start: int = 0,
    sampler: Callable[[mx.array], mx.array] | None = None,
    logits_processors: list[Callable[[mx.array, mx.array], mx.array]] | None = None,
    prefill_step_size: int | None = 512,
    seed: int | None = None,
    continuation_tokens: int = 1,
    on_start: Callable[[], None] | None = None,
    on_stop: Callable[[], None] | None = None,
    trace_forward: Callable[[str, int, Any, list[Any]], None] | None = None,
    **kwargs: Any,
) -> PrefillResult:
    """Run the pinned prompt loop and time only prompt model submissions.

    ``continuation_tokens`` includes the first sampled token and is evaluated
    after the stop-before-sampling boundary.  It exists for differential tests;
    production pure-prefill timing normally requests one token.
    """
    if continuation_tokens < 1:
        raise ValueError("continuation_tokens must include at least the first token")

    ar, source_path, source_sha = assert_pinned_ar_source()
    quantize_cache_fn = functools.partial(
        ar._generate_module_override("maybe_quantize_kv_cache", ar.maybe_quantize_kv_cache),
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
        kv_quant_scheme=kv_quant_scheme,
        kv_key_bits=kv_key_bits,
        kv_value_bits=kv_value_bits,
        kv_key_scheme=kv_key_scheme,
        kv_value_scheme=kv_value_scheme,
    )

    if sampler is None:
        if (
            seed is not None
            and temperature > 0
            and min_p == ar.DEFAULT_MIN_P
            and top_k == ar.DEFAULT_TOP_K
            and top_n_sigma == ar.DEFAULT_TOP_N_SIGMA
            and not p_less
            and typical_p == 1.0
        ):
            sampler = ar._PositionedTargetSampler(
                temperature=temperature, top_p=top_p, seed=seed
            )
        else:
            sampler = ar._generate_module_override("make_sampler", ar.make_sampler)(
                temp=temperature,
                top_p=top_p,
                min_p=min_p,
                top_k=top_k,
                top_n_sigma=top_n_sigma,
                p_less=p_less,
                typical_p=typical_p,
            )

    processors = ar._generate_module_override(
        "make_logits_processors", ar.make_logits_processors
    )(
        logit_bias,
        repetition_penalty,
        repetition_context_size,
        presence_penalty,
        presence_context_size,
        frequency_penalty,
        frequency_context_size,
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    original_input_ids = input_ids
    effective_prompt_tokens = int(input_ids.size)
    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)
    target_sample_position = 0
    if prompt_cache is None:
        prompt_cache = ar.cache.make_prompt_cache(
            model.language_model, max_kv_size=max_kv_size
        )

    started_at: float | None = None
    stopped_at: float | None = None
    chunks: list[ChunkTiming] = []

    def start_boundary() -> None:
        nonlocal started_at
        if started_at is None:
            if on_start is not None:
                on_start()
            started_at = time.perf_counter()

    def stop_boundary() -> None:
        nonlocal stopped_at
        stopped_at = time.perf_counter()
        if on_stop is not None:
            on_stop()

    with mx.stream(ar.generation_stream):
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        inputs_embeds = embedding_output.inputs_embeds
        kwargs.update(
            {
                key: value
                for key, value in embedding_output.to_dict().items()
                if key != "inputs_embeds" and value is not None
            }
        )
        if prefill_step_size is not None and not ar._chunked_prefill_enabled(
            model,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            prompt_cache=prompt_cache,
            draft_model=None,
            draft_kind="dflash",
            prefill_kwargs=kwargs,
        ):
            prefill_step_size = None

        should_chunk = (
            prefill_step_size is not None
            and inputs_embeds.shape[1] > prefill_step_size
        )
        processed_tokens = 0
        if prefill_step_size is not None and should_chunk:
            while inputs_embeds.shape[1] > 1:
                n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                chunk_kwargs = kwargs
                if getattr(model.language_model, "supports_logits_to_keep", False):
                    chunk_kwargs = {**kwargs, "logits_to_keep": 1}
                start_boundary()
                chunk_started = time.perf_counter()
                outputs = model.language_model(
                    inputs=input_ids[:, :n_to_process],
                    inputs_embeds=inputs_embeds[:, :n_to_process],
                    cache=prompt_cache,
                    n_to_process=n_to_process,
                    **chunk_kwargs,
                )
                quantize_cache_fn(prompt_cache)
                _eval_cache(prompt_cache)
                chunk_stopped = time.perf_counter()
                chunks.append(
                    ChunkTiming(
                        "prefill_chunk",
                        processed_tokens,
                        processed_tokens + n_to_process,
                        chunk_stopped - chunk_started,
                    )
                )
                if trace_forward is not None:
                    trace_forward("prefill_chunk", n_to_process, outputs, prompt_cache)
                processed_tokens += n_to_process
                inputs_embeds = inputs_embeds[:, n_to_process:]
                input_ids = input_ids[:, n_to_process:]
                mx.clear_cache()
            input_ids = input_ids[:, -1:]

        step_kwargs = kwargs
        if getattr(model.language_model, "supports_logits_to_keep", False):
            step_kwargs = {**step_kwargs, "logits_to_keep": 1}
        start_boundary()
        final_started = time.perf_counter()
        outputs = model.language_model(
            input_ids,
            inputs_embeds=inputs_embeds,
            cache=prompt_cache,
            **step_kwargs,
        )
        logits = outputs.logits[:, -1, :]
        if len(processors) > 0 and len(input_ids) > 0:
            tokens = mx.concat([tokens, input_ids.flatten()])
            for processor in processors:
                logits = processor(tokens, logits)
        quantize_cache_fn(prompt_cache)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)

        # The sole added production synchronization: after the final prompt
        # forward/cache state, before sampling.  No per-layer eval is added.
        mx.eval(logits, logprobs, [entry.state for entry in prompt_cache])
        final_stopped = time.perf_counter()
        chunks.append(
            ChunkTiming(
                "final_prompt_transition",
                processed_tokens,
                effective_prompt_tokens,
                final_stopped - final_started,
            )
        )
        if trace_forward is not None:
            trace_forward(
                "final_prompt_transition", int(input_ids.shape[1]), outputs, prompt_cache
            )
        stop_boundary()

    if started_at is None or stopped_at is None:
        raise AssertionError("prefill timing boundaries were not reached")

    y = ar._sample_with_positions(
        sampler, logprobs, row_ids=[0] * logprobs.shape[0], positions=[0]
    )
    target_sample_position = logprobs.shape[0]
    mx.eval(y, logprobs)
    generated = [int(y.item())]
    generated_logits = [logits]
    generated_logprobs = [logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs]

    # Untimed continuation uses the same _step ordering as the public loop.
    current_y = y
    for _ in range(1, continuation_tokens):
        with mx.stream(ar.generation_stream):
            decode_kwargs = kwargs
            if getattr(model.language_model, "supports_logits_to_keep", False):
                decode_kwargs = {**decode_kwargs, "logits_to_keep": 1}
            decode_outputs = model.language_model(
                current_y[None], cache=prompt_cache, **decode_kwargs
            )
            decode_logits = decode_outputs.logits[:, -1, :]
            if len(processors) > 0:
                tokens = mx.concat([tokens, current_y.flatten()])
                for processor in processors:
                    decode_logits = processor(tokens, decode_logits)
            quantize_cache_fn(prompt_cache)
            decode_logprobs = decode_logits - mx.logsumexp(
                decode_logits, axis=-1, keepdims=True
            )
            next_y = ar._sample_with_positions(
                sampler,
                decode_logprobs,
                row_ids=[0] * decode_logprobs.shape[0],
                positions=[target_sample_position],
            )
            target_sample_position += decode_logprobs.shape[0]
        mx.eval(next_y, decode_logprobs)
        current_y = next_y
        generated.append(int(current_y.item()))
        generated_logits.append(decode_logits)
        generated_logprobs.append(
            decode_logprobs.squeeze(0)
            if decode_logprobs.shape[0] == 1
            else decode_logprobs
        )

    elapsed = stopped_at - started_at
    return PrefillResult(
        source_path=str(source_path),
        source_sha256=source_sha,
        effective_prompt_tokens=effective_prompt_tokens,
        prefill_seconds=elapsed,
        prompt_rate_tokens_per_second=effective_prompt_tokens / elapsed,
        chunk_timings=chunks,
        final_prompt_logits=logits,
        final_prompt_logprobs=logprobs.squeeze(0) if logprobs.shape[0] == 1 else logprobs,
        first_token_id=generated[0],
        generated_token_ids=generated,
        generated_logits=generated_logits,
        generated_logprobs=generated_logprobs,
        prompt_cache=prompt_cache,
    )
