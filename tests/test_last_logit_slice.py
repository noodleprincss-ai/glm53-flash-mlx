"""Focused contract tests for generation-only terminal-logit slicing."""
from __future__ import annotations
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import unittest
from glm53_flash_mlx.glm5_next.language import LanguageModel

class _Core(nn.Module):
    def __init__(self, hidden: int):
        super().__init__(); self.hidden = hidden; self.calls = []
    def __call__(self, inputs, cache=None, inputs_embeds=None, logits_to_keep=None):
        del cache
        out = inputs_embeds
        if out is None:
            out = mx.arange(inputs.size * self.hidden, dtype=mx.float32).reshape(*inputs.shape, self.hidden)
        self.calls.append({"sequence": int(out.shape[1]), "logits_to_keep": logits_to_keep})
        if logits_to_keep: out = out[:, -int(logits_to_keep):, :]
        return out

@dataclass
class _Args:
    hidden_size: int = 8
    vocab_size: int = 13
    tie_word_embeddings: bool = False
    model_type: str = "test"

def _language_model() -> LanguageModel:
    model = LanguageModel.__new__(LanguageModel); nn.Module.__init__(model)
    model.args = _Args(); model.config = model.args; model.model_type = model.args.model_type
    model.model = _Core(model.args.hidden_size)
    model.lm_head = nn.Linear(model.args.hidden_size, model.args.vocab_size, bias=False)
    model.lm_head.weight = mx.arange(model.args.vocab_size * model.args.hidden_size, dtype=mx.float32).reshape(model.args.vocab_size, model.args.hidden_size) / 100
    return model

class LastLogitSliceTests(unittest.TestCase):
    def test_unchunked_128_and_512_preserve_last_logits(self):
        for length in (128, 512):
            with self.subTest(length=length):
                model = _language_model(); ids = mx.arange(length, dtype=mx.int32)[None]
                full = model(ids).logits; sliced = model(ids, logits_to_keep=1).logits; mx.eval(full, sliced)
                self.assertEqual(sliced.shape, (1, 1, model.args.vocab_size))
                np.testing.assert_allclose(np.array(sliced), np.array(full[:, -1:, :]), rtol=2e-7, atol=2e-3)
                self.assertEqual(model.model.calls[-1], {"sequence": length, "logits_to_keep": 1})


    def test_chunked_2049_terminal_logits_and_cache_state_are_identical(self):
        model = _language_model(); ids = mx.arange(2049, dtype=mx.int32)[None]
        cache_full, cache_sliced = [], []; offset = 0; full_last = sliced_last = None
        for size in (512, 512, 512, 512, 1):
            piece = ids[:, offset:offset + size]
            full_last = model(piece).logits[:, -1:, :]; cache_full.extend(piece.tolist()[0])
            sliced_last = model(piece, logits_to_keep=1).logits; cache_sliced.extend(piece.tolist()[0]); offset += size
        mx.eval(full_last, sliced_last)
        np.testing.assert_array_equal(np.array(sliced_last), np.array(full_last))
        self.assertEqual(cache_sliced, cache_full)
        self.assertEqual(cache_full, ids.tolist()[0])

    def test_multi_logit_and_default_public_shapes(self):
        model = _language_model(); ids = mx.arange(9, dtype=mx.int32)[None]
        self.assertEqual(model(ids).logits.shape, (1, 9, model.args.vocab_size))
        self.assertEqual(model(ids, logits_to_keep=3).logits.shape, (1, 3, model.args.vocab_size))
        self.assertIs(model.supports_logits_to_keep, True)

if __name__ == "__main__":
    unittest.main()
