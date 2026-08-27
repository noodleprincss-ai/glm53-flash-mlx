"""Focused exact-shape correctness tests for the KDA decode short-conv kernel."""

import unittest

import mlx.core as mx
import mlx.nn as nn

from glm53_flash_mlx.glm5_next.language import _kda_short_conv_step


class KdaShortConvKernelTest(unittest.TestCase):
    def test_four_tap_output_and_state_dtype_variants(self):
        for dtype, atol in (
            (mx.float32, 1e-6),
            (mx.float16, 4e-3),
            (mx.bfloat16, 4e-2),
        ):
            with self.subTest(dtype=dtype):
                C = 3 * 4 * 16
                x = mx.random.normal((1, 1, C)).astype(dtype)
                state = mx.random.normal((1, 3, C)).astype(dtype)
                conv = nn.Conv1d(C, C, 4, groups=C, bias=False)
                conv.weight = mx.random.normal(conv.weight.shape).astype(dtype)
                xp = mx.concatenate([state, x], axis=1)
                expected_y = nn.silu(conv(xp))
                expected_state = mx.contiguous(xp[:, -3:, :])
                actual_y, actual_state = _kda_short_conv_step(x, state, conv.weight)
                mx.eval(expected_y, expected_state, actual_y, actual_state)
                self.assertTrue(
                    mx.allclose(actual_y, expected_y, atol=atol, rtol=0).item()
                )
                self.assertTrue(mx.array_equal(actual_state, expected_state).item())

    def test_recurrent_state_matches_parent_for_multiple_steps(self):
        C = 3 * 4 * 16
        dtype = mx.bfloat16
        state_parent = mx.zeros((1, 3, C), dtype=dtype)
        state_fused = mx.zeros((1, 3, C), dtype=dtype)
        conv = nn.Conv1d(C, C, 4, groups=C, bias=False)
        conv.weight = (mx.random.normal(conv.weight.shape) * 0.1).astype(dtype)
        for _ in range(16):
            x = mx.random.normal((1, 1, C)).astype(dtype)
            xp = mx.concatenate([state_parent, x], axis=1)
            y_parent = nn.silu(conv(xp))
            state_parent = mx.contiguous(xp[:, -3:, :])
            y_fused, state_fused = _kda_short_conv_step(x, state_fused, conv.weight)
            mx.eval(y_parent, state_parent, y_fused, state_fused)
            self.assertTrue(
                mx.allclose(y_fused, y_parent, atol=0.03125, rtol=0).item()
            )
            self.assertTrue(mx.array_equal(state_fused, state_parent).item())


if __name__ == "__main__":
    unittest.main()
