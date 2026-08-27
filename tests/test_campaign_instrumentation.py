"""Differential tests for the source-pinned prefill boundary."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np
import unittest

from campaign.instrumented_prefill import PINNED_AR_SHA256, assert_pinned_ar_source, instrumented_prefill


def legacy_parity_module():
    path = Path(__file__).with_name("test_parity.py")
    spec = importlib.util.spec_from_file_location("legacy_parity", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def freeze(value: Any) -> Any:
    if isinstance(value, mx.array):
        mx.eval(value)
        return np.array(value)
    if isinstance(value, (tuple, list)):
        return [freeze(item) for item in value]
    if isinstance(value, dict):
        return {key: freeze(item) for key, item in value.items()}
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return repr(value)


def cache_snapshot(prompt_cache: list[Any]) -> list[Any]:
    return [
        {
            "offset": getattr(entry, "offset", None),
            "state": freeze(entry.state),
        }
        for entry in prompt_cache
    ]


def assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_nested_equal(a, b)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_nested_equal(left[key], right[key])
    else:
        assert left == right


class TracedLanguageModel:
    def __init__(self, target: Any, records: list[dict[str, Any]]):
        self._target = target
        self._records = records

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        outputs = self._target(*args, **kwargs)
        prompt_cache = kwargs["cache"]
        mx.eval(outputs.logits, [entry.state for entry in prompt_cache])
        if "inputs_embeds" in kwargs and kwargs["inputs_embeds"] is not None:
            token_count = int(kwargs["inputs_embeds"].shape[1])
            has_embeddings = True
        elif args:
            token_count = int(args[0].shape[1])
            has_embeddings = False
        else:
            token_count = int(kwargs["inputs"].shape[1])
            has_embeddings = True
        self._records.append(
            {
                "token_count": token_count,
                "has_embeddings": has_embeddings,
                "logits": np.array(outputs.logits.astype(mx.float32)),
                "cache": cache_snapshot(prompt_cache),
            }
        )
        return outputs


class TracedModel:
    def __init__(self, target: Any, records: list[dict[str, Any]]):
        self._target = target
        self.language_model = TracedLanguageModel(target.language_model, records)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._target, name)

    def get_input_embeddings(self, *args: Any, **kwargs: Any) -> Any:
        return self._target.get_input_embeddings(*args, **kwargs)


def build_model() -> Any:
    parity = legacy_parity_module()
    return parity.build_mlx(parity.build_hf(seed=7))


def run_public(ids: list[int], step: int | None, tokens: int = 5):
    import importlib
    ar = importlib.import_module("mlx_vlm.generate.ar")
    records: list[dict[str, Any]] = []
    model = TracedModel(build_model(), records)
    prompt_cache = ar.cache.make_prompt_cache(model.language_model)
    generator = ar.generate_step(
        mx.array([ids]), model, None, None,
        max_tokens=tokens, temperature=0.0,
        prefill_step_size=step, prompt_cache=prompt_cache,
    )
    yielded = []
    logprobs = []
    for _ in range(tokens):
        token, lp = next(generator)
        yielded.append(int(token))
        mx.eval(lp)
        logprobs.append(np.array(lp.astype(mx.float32)))
    generator.close()
    mx.synchronize()
    return records, yielded, logprobs



class InstrumentationTests(unittest.TestCase):
    def test_source_pin_is_exact(self):
        _, _, digest = assert_pinned_ar_source()
        self.assertEqual(digest, PINNED_AR_SHA256)

    def test_prefill_matches_public_generate_step_across_chunk_remainders(self):
        # step=4 covers every remainder 0..3 plus an unchunked prompt.
        cases = [(3, None)] + [(length, 4) for length in range(5, 13)]
        for length, step in cases:
            with self.subTest(length=length, step=step):
                ids = [2 + ((index * 11) % 101) for index in range(length)]
                public_records, public_tokens, public_logprobs = run_public(ids, step)
                instrument_records: list[dict[str, Any]] = []

                def trace(kind: str, token_count: int, outputs: Any, cache: list[Any]) -> None:
                    instrument_records.append(
                        {
                            "kind": kind,
                            "token_count": token_count,
                            "logits": np.array(outputs.logits.astype(mx.float32)),
                            "cache": cache_snapshot(cache),
                        }
                    )

                result = instrumented_prefill(
                    mx.array([ids]), build_model(), None, None,
                    temperature=0.0, prefill_step_size=step,
                    continuation_tokens=5, trace_forward=trace,
                )
                prompt_call_count = len(instrument_records)
                public_prompt = public_records[:prompt_call_count]
                self.assertEqual(
                    [row["token_count"] for row in public_prompt],
                    [row["token_count"] for row in instrument_records],
                )
                for public, instrument in zip(public_prompt, instrument_records):
                    assert_nested_equal(public["cache"], instrument["cache"])
                np.testing.assert_array_equal(
                    public_prompt[-1]["logits"][:, -1, :],
                    np.array(result.final_prompt_logits.astype(mx.float32)),
                )
                self.assertEqual(result.first_token_id, public_tokens[0])
                self.assertEqual(result.generated_token_ids, public_tokens)
                # The first post-prompt model call consumes the first sampled
                # token and produces the first generated-token logits.
                np.testing.assert_array_equal(
                    public_records[prompt_call_count]["logits"][:, -1, :],
                    np.array(result.generated_logits[1].astype(mx.float32)),
                )
                for observed, expected in zip(result.generated_logprobs, public_logprobs):
                    np.testing.assert_array_equal(np.array(observed.astype(mx.float32)), expected)


if __name__ == "__main__":
    unittest.main()
