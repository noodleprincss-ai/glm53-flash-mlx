"""Focused exactness and lifecycle tests for lightning-indexer pool reuse."""

from __future__ import annotations

from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from mlx_vlm.models.cache import BatchKVCache, KVCache, RotatingKVCache

from glm53_flash_mlx.glm5_next.language import Glm5NextIndexer, _IndexerPoolState


def make_indexer(*, topk: int = 8, kpool: int = 4) -> Glm5NextIndexer:
    args = SimpleNamespace(
        hidden_size=8,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=topk,
        index_kpool=kpool,
        index_kpool_always_select_tail=True,
        q_lora_rank=4,
    )
    indexer = Glm5NextIndexer(args)
    # Nonzero deterministic weights make both pooled keys and query-dependent
    # selection load-bearing while avoiding ties in the normal parity cases.
    indexer.wq_b.weight = mx.arange(32, dtype=mx.float32).reshape(8, 4) / 31
    indexer.wk.weight = mx.arange(32, dtype=mx.float32).reshape(4, 8) / 29
    indexer.weights_proj.weight = mx.arange(16, dtype=mx.float32).reshape(2, 8) / 17
    indexer.index_kpool_compress_ape = mx.arange(16, dtype=mx.float32).reshape(4, 4) / 19
    indexer.index_kpool_compress_gate = mx.arange(32, dtype=mx.float32).reshape(4, 8) / 23
    return indexer


def tensors(start: int, length: int, batch: int = 1) -> tuple[mx.array, mx.array]:
    values = np.arange(start * 8, (start + length) * 8, dtype=np.float32)
    values = np.sin(values / 13).reshape(1, length, 8)
    x = np.concatenate([values + b / 10 for b in range(batch)], axis=0)
    qr = np.cos(np.arange(start * 4, (start + length) * 4, dtype=np.float32) / 11)
    qr = np.broadcast_to(qr.reshape(1, length, 4), (batch, length, 4)).copy()
    return mx.array(x), mx.array(qr)


def run_chunks(indexer: Glm5NextIndexer, chunks: list[int], *, force_full: bool = False):
    cache = KVCache()
    outputs = []
    position = 0
    for length in chunks:
        if force_full and hasattr(cache, "_pool"):
            cache._pool = None
        x, qr = tensors(position, length)
        out = indexer(x, qr, None, cache)
        if out is not None:
            mx.eval(out)
            outputs.append(np.array(out))
        else:
            outputs.append(None)
        position += length
    return outputs, cache


def assert_outputs_equal(left, right):
    assert len(left) == len(right)
    for a, b in zip(left, right):
        if a is None or b is None:
            assert a is b
        else:
            np.testing.assert_array_equal(a, b)


def test_2048_boundary_and_chunk_remainders_match_full_rebuild():
    cases = [
        [2048, 1],
        [2047, 2],
        [2049, 1, 1, 1],
        [2049, 3, 5, 7, 511],
        [2049, 512, 509, 3, 1],
    ]
    for chunks in cases:
        cached, cache = run_chunks(make_indexer(topk=2048), chunks)
        rebuilt, _ = run_chunks(make_indexer(topk=2048), chunks, force_full=True)
        assert_outputs_equal(cached, rebuilt)
        if sum(chunks) > 2048:
            assert isinstance(cache._pool, _IndexerPoolState)
            assert cache._pool.sequence_length == sum(chunks)

def test_reuse_recomputes_only_prior_tail_and_new_suffix():
    indexer = make_indexer(topk=8)
    calls = []
    original = indexer._pooled_states

    def recording(keys, gate, valid):
        calls.append(keys.shape[1])
        return original(keys, gate, valid)

    indexer._pooled_states = recording
    cache = KVCache()
    pos = 0
    for length in (9, 6, 1, 8):
        x, qr = tensors(pos, length)
        out = indexer(x, qr, None, cache)
        mx.eval(out)
        pos += length
    # 9 is the first full build. Thereafter: prior tail 1 + 6, prior tail 3 + 1,
    # then an aligned 8-token suffix. A full rebuild would be 15, 16, and 24.
    assert calls == [9, 7, 4, 8]


def test_token_decode_long_run_matches_full_rebuild():
    chunks = [9] + [1] * 260
    cached, _ = run_chunks(make_indexer(), chunks)
    rebuilt, _ = run_chunks(make_indexer(), chunks, force_full=True)
    assert_outputs_equal(cached, rebuilt)


def test_reset_trim_state_replace_and_extract_fail_closed():
    indexer = make_indexer()
    calls = []
    original = indexer._pooled_states

    def recording(keys, gate, valid):
        calls.append(keys.shape[1])
        return original(keys, gate, valid)

    indexer._pooled_states = recording
    cache = KVCache()
    x, qr = tensors(0, 9)
    mx.eval(indexer(x, qr, None, cache))
    state = cache.state

    # Trim invalidates by length even though the backing object is unchanged.
    cache.trim(1)
    x, qr = tensors(8, 1)
    mx.eval(indexer(x, qr, None, cache))
    assert calls[-1] == 9

    # Replacing same-length state invalidates by source-array identity.
    cache.state = tuple(mx.array(np.array(v)) for v in state)
    x, qr = tensors(9, 1)
    mx.eval(indexer(x, qr, None, cache))
    assert calls[-1] == 10

    # Extract returns a fresh KVCache without custom state: full rebuild.
    fork = cache.extract(0)
    x, qr = tensors(10, 1)
    mx.eval(indexer(x, qr, None, fork))
    assert calls[-1] == 11

    # A model/cache reset is simply a fresh cache with no state.
    fresh = KVCache()
    x, qr = tensors(0, 9)
    mx.eval(indexer(x, qr, None, fresh))
    assert calls[-1] == 9


def test_partial_pools_duplicates_invalid_minus_one_and_causal_tail():
    indexer = make_indexer(topk=4)
    # Duplicate keys are legal; padding/partial slots must remain invalid and -1.
    keys = mx.zeros((1, 7, 4))
    gate = mx.zeros((1, 7, 4))
    valid = mx.array([[True, True, True, True, True, False, False]])
    pk, pi, pv = indexer._pooled_states(keys, gate, valid)
    mx.eval(pk, pi, pv)
    np.testing.assert_array_equal(np.array(pi), [[[0, 1, 2, 3], [4, -1, -1, -1]]])
    np.testing.assert_array_equal(np.array(pv), [[True, False]])

    # Explicit padding makes the cache non-reusable. Invalid queries are all -1,
    # while causal tail positions never point beyond the query position.
    cache = KVCache()
    x, qr = tensors(0, 9)
    mask = mx.array([[True] * 8 + [False]])
    out = indexer(x, qr, mask, cache)
    mx.eval(out)
    arr = np.array(out)
    assert np.all(arr[0, :, -1] == -1)
    for query, row in enumerate(arr[0, 0, :-1]):
        assert np.all(row[(row >= 0)] <= query)
    x, qr = tensors(9, 1)
    indexer(x, qr, mx.array([[True]]), cache)
    assert cache._pool.sequence_length == 10


def test_batch_and_unknown_or_rotating_cache_never_reuse():
    indexer = make_indexer()
    calls = []
    original = indexer._pooled_states

    def recording(keys, gate, valid):
        calls.append(keys.shape[1])
        return original(keys, gate, valid)

    indexer._pooled_states = recording
    batch = BatchKVCache([0, 0])
    for pos, length in ((0, 9), (9, 1)):
        x, qr = tensors(pos, length, batch=2)
        out = indexer(x, qr, None, batch)
        mx.eval(out)
    assert calls[-2:] == [9, 10]

    rotating = RotatingKVCache(max_size=16)
    for pos, length in ((0, 9), (9, 1)):
        x, qr = tensors(pos, length)
        out = indexer(x, qr, None, rotating)
        mx.eval(out)
    assert calls[-2:] == [9, 10]


def main() -> int:
    tests = [
        test_2048_boundary_and_chunk_remainders_match_full_rebuild,
        test_reuse_recomputes_only_prior_tail_and_new_suffix,
        test_token_decode_long_run_matches_full_rebuild,
        test_reset_trim_state_replace_and_extract_fail_closed,
        test_partial_pools_duplicates_invalid_minus_one_and_causal_tail,
        test_batch_and_unknown_or_rotating_cache_never_reuse,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
