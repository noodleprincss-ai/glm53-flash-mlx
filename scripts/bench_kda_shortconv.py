#!/usr/bin/env python3
"""Exact GLM-5.3 KDA decode-shape benchmark; allocates under 1 MiB, never model weights."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import mlx.core as mx
import mlx.nn as nn

from glm53_flash_mlx.glm5_next.language import _kda_short_conv_step


def run_case(fn, x, state, warmup: int, iterations: int, repeats: int):
    for _ in range(warmup):
        y, state = fn(x, state)
        mx.eval(y, state)
    samples = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        for _ in range(iterations):
            y, state = fn(x, state)
            mx.eval(y, state)
        samples.append((time.perf_counter_ns() - start) / iterations / 1e3)
    return samples


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    ap.add_argument("--json")
    args = ap.parse_args()

    dtype = getattr(mx, args.dtype)
    B, L, H, D, K = 1, 1, 64, 128, 4
    C = 3 * H * D
    x = mx.random.normal((B, L, C)).astype(dtype)
    state = mx.random.normal((B, K - 1, C)).astype(dtype)
    conv = nn.Conv1d(C, C, K, groups=C, bias=False)
    conv.weight = mx.random.normal(conv.weight.shape).astype(dtype)
    mx.eval(x, state, conv.weight)

    def parent(x, state):
        xp = mx.concatenate([state, x], axis=1)
        return nn.silu(conv(xp)), mx.contiguous(xp[:, -3:, :])

    def fused(x, state):
        return _kda_short_conv_step(x, state, conv.weight)

    # Report every repeat and its range so scheduler/thermal variance remains visible.
    parent_us = run_case(parent, x, state, args.warmup, args.iterations, args.repeats)
    fused_us = run_case(fused, x, state, args.warmup, args.iterations, args.repeats)
    result = {
        "shape": {"B": B, "L": L, "heads": H, "head_dim": D, "conv_channels": C, "taps": K},
        "dtype": args.dtype,
        "eval_boundary": "mx.eval(output, new_state) after every recurrent step",
        "iterations_per_repeat": args.iterations,
        "repeats": args.repeats,
        "parent_us": parent_us,
        "fused_us": fused_us,
        "parent_median_us": statistics.median(parent_us),
        "fused_median_us": statistics.median(fused_us),
        "parent_range_us": [min(parent_us), max(parent_us)],
        "fused_range_us": [min(fused_us), max(fused_us)],
    }
    result["speedup"] = result["parent_median_us"] / result["fused_median_us"]
    result["saved_us_per_kda_layer"] = result["parent_median_us"] - result["fused_median_us"]
    result["ceiling_ms_per_token_34_layers"] = result["saved_us_per_kda_layer"] * 34 / 1000
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
