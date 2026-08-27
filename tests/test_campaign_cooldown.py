"""Focused synthetic tests for the macOS cooldown safety gate."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from campaign.locking import CampaignLock, audit_lock
from campaign.paired_runner import assess_cooldown, persist_result, wait_teardown


def sample(
    index: int,
    *,
    swap: int = 0,
    pageins: int = 0,
    pageouts: int = 0,
    compressor: int = 0,
    free: int = 80,
    rss: int = 100,
) -> dict[str, int | str]:
    return {
        "captured_at_utc": f"sample-{index}",
        "swap_used_bytes": swap,
        "pageins": pageins,
        "pageouts": pageouts,
        "compressor_pages": compressor,
        "memory_free_percent": free,
        "self_rss_bytes": 1,
        "other_rss_bytes": rss - 1,
        "combined_rss_bytes": rss,
        "physical_memory_bytes": 1000,
    }


def assess(samples: list[dict[str, int | str]], process_gone: bool = True):
    return assess_cooldown(
        samples,
        process_gone=process_gone,
        max_swap_growth=64,
        max_pageout_growth=8,
        max_compressor_growth=8,
        min_free_percent=10,
        host_rss_ceiling_bytes=900,
    )


class CooldownTests(unittest.TestCase):
    def test_transient_ticks_do_not_masquerade_as_sustained_pressure(self):
        rows = [sample(0), sample(1, compressor=1), sample(2, compressor=1), sample(3, compressor=1)]
        result = assess(rows)
        self.assertFalse(result["hard_failure"])
        self.assertEqual(result["deltas"]["compressor_pages"], 1)

    def test_monotonic_benign_pageins_are_telemetry_only(self):
        rows = [sample(index, pageins=index * 47) for index in range(4)]
        result = assess(rows)
        self.assertFalse(result["hard_failure"])
        self.assertTrue(result["sustained_growth"]["pageins"])
        self.assertTrue(result["pageins_corroborating"])
        self.assertEqual(result["pageins_role"], "telemetry_and_corroboration_only")

    def test_genuine_multi_signal_pressure_fails_closed(self):
        rows = [
            sample(index, swap=index * 32, pageins=index * 50, pageouts=index * 4,
                   compressor=index * 4, free=8 - index, rss=850 + index * 20)
            for index in range(4)
        ]
        result = assess(rows)
        self.assertTrue(result["hard_failure"])
        joined = " | ".join(result["failure_reasons"])
        self.assertIn("swap_used_bytes", joined)
        self.assertIn("pageouts", joined)
        self.assertIn("compressor_pages", joined)
        self.assertIn("memory pressure", joined)
        self.assertIn("combined_rss_bytes", joined)

    def test_process_not_gone_is_a_hard_failure(self):
        result = assess([sample(index) for index in range(4)], process_gone=False)
        self.assertTrue(result["hard_failure"])
        self.assertIn("benchmark process tree did not tear down", result["failure_reasons"])


class EvidencePersistenceTests(unittest.TestCase):
    def test_result_write_failure_uses_fallback_and_lock_is_cleaned_up(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock_path = root / "paired.lock"
            output = root / "primary" / "result.json"
            emergency = root / "emergency"
            value = {"run_id": "synthetic", "status": "failed", "raw_runs": [
                {"cooldown_samples": [sample(index) for index in range(4)],
                 "failure_reason": "synthetic abort"}
            ]}

            def fail_primary(path: Path, result: dict):
                raise OSError(f"synthetic write failure: {path}")

            with CampaignLock(lock_path, "synthetic"):
                error = persist_result(output, emergency, value, writer=fail_primary)
                self.assertIsInstance(error, OSError)
            self.assertEqual(audit_lock(lock_path)["state"], "available")
            fallback = Path(value["emergency_result_path"])
            persisted = json.loads(fallback.read_text())
            self.assertEqual(len(persisted["raw_runs"][0]["cooldown_samples"]), 4)
            self.assertEqual(persisted["raw_runs"][0]["failure_reason"], "synthetic abort")
            self.assertIn("result_write_error", persisted)

    def test_wait_teardown_reports_lingering_process_instead_of_dropping_evidence(self):
        # PID 1 is guaranteed to survive a zero-second synthetic teardown window.
        result = wait_teardown(1, [], 0)
        self.assertFalse(result["process_gone"])
        self.assertEqual(result["alive_pids"], [1])


if __name__ == "__main__":
    unittest.main()
