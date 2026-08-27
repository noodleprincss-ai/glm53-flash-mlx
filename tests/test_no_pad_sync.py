"""Focused correctness and source-contract tests for C2 no-pad synchronization removal."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace

import mlx.core as mx
from mlx_vlm.models.cache import BatchKVCache, KVCache

from glm53_flash_mlx.glm5_next.language import Glm5NextIndexer


def make_indexer(*, topk: int = 4, kpool: int = 4) -> Glm5NextIndexer:
    return Glm5NextIndexer(
        SimpleNamespace(
            hidden_size=4,
            index_n_heads=1,
            index_head_dim=2,
            index_topk=topk,
            index_kpool=kpool,
            index_kpool_always_select_tail=True,
            q_lora_rank=2,
        )
    )


def inputs(length: int, batch: int = 1):
    # Nonconstant values keep pool/order paths realistic while remaining deterministic.
    x = mx.arange(batch * length * 4, dtype=mx.float32).reshape(batch, length, 4) / 97
    qr = x[..., :2]
    return x, qr


class NoPadSyncTests(unittest.TestCase):
    def test_source_has_no_host_scalar_sync_or_legacy_no_pad_fact(self):
        source = inspect.getsource(Glm5NextIndexer.__call__)
        self.assertNotIn("bool(mx.all(valid))", source)
        self.assertNotIn("_no_pad", source)
        self.assertIn("type(cache) is KVCache", source)
        self.assertIn("_padding_free", source)

    def test_exact_2048_2049_sparse_threshold(self):
        indexer = make_indexer(topk=2048)
        x, qr = inputs(2048)
        self.assertIsNone(indexer(x, qr, None, cache=KVCache()))

        indexer = make_indexer(topk=2048)
        x, qr = inputs(2049)
        selected = indexer(x, qr, None, cache=KVCache())
        self.assertIsNotNone(selected)
        self.assertEqual(selected.shape[:3], (1, 1, 2049))
        mx.eval(selected)
        last = selected[0, 0, -1].tolist()
        self.assertTrue(all(i == -1 or 0 <= i < 2049 for i in last))

    def test_real_padding_and_causal_visibility(self):
        indexer = make_indexer(topk=4)
        x, qr = inputs(9)
        valid = mx.array([[False, False, True, True, True, True, True, True, True]])
        selected = indexer(x, qr, valid, cache=None)
        mx.eval(selected)
        rows = selected[0, 0].tolist()
        self.assertTrue(all(i == -1 for row in rows[:2] for i in row))
        for query, row in enumerate(rows[2:], start=2):
            self.assertTrue(all(i == -1 or 2 <= i <= query for i in row))

    def test_single_stream_reuses_suffix_but_batch_cache_fails_closed(self):
        def run(cache, prefill_mask=None):
            indexer = make_indexer(topk=4)
            pooled_lengths = []
            original = indexer._pooled_states

            def recording(keys, gate_scores, valid):
                pooled_lengths.append(keys.shape[1])
                return original(keys, gate_scores, valid)

            indexer._pooled_states = recording
            x, qr = inputs(9)
            prefill = indexer(x, qr, prefill_mask, cache=cache)
            mx.eval(prefill)
            x, qr = inputs(1)
            decode = indexer(x, qr, None, cache=cache)
            mx.eval(decode)
            return pooled_lengths, decode

        single_lengths, single = run(KVCache())
        padded = mx.array([[False, False, True, True, True, True, True, True, True]])
        padded_lengths, padded_decode = run(KVCache(), padded)
        all_true_lengths, _ = run(KVCache(), mx.ones((1, 9), dtype=mx.bool_))
        batch_lengths, batch = run(BatchKVCache([0]))
        self.assertEqual(single_lengths, [9, 2])
        self.assertEqual(padded_lengths, [9, 10])
        self.assertEqual(all_true_lengths, [9, 10])
        self.assertEqual(batch_lengths, [9, 10])
        self.assertEqual(single.shape, padded_decode.shape)
        self.assertEqual(single.shape, batch.shape)

    def test_duplicate_indices_are_idempotent_when_materialized_as_sparse_mask(self):
        # The always-selected tail can overlap a selected pool. Mask materialization
        # must remain a set operation: duplicates may not toggle or erase visibility.
        indices = mx.array([[[[2, 2, 3, -1]]]], dtype=mx.int32)
        valid = indices >= 0
        safe = mx.where(valid, indices, 5)
        mask = mx.zeros((1, 1, 1, 6), dtype=mx.bool_)
        mask = mx.put_along_axis(mask, safe, mx.array(True), axis=-1)[..., :5]
        mx.eval(mask)
        self.assertEqual(mask[0, 0, 0].tolist(), [False, False, True, True, False])


if __name__ == "__main__":
    unittest.main()
