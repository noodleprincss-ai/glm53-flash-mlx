"""Focused manifest and fail-closed provenance tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from campaign.provenance import ProvenanceError, file_identity, parse_hf_manifest, verify_checkpoint_stats


class ProvenanceTests(unittest.TestCase):
    def test_parse_hf_manifest_is_exact_json_input(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.psv"
            path.write_text("model-00001.safetensors|3|abc|https://example.invalid/a\n")
            self.assertEqual(parse_hf_manifest(path), {
                "model-00001.safetensors": {
                    "size": 3,
                    "sha256": "abc",
                    "url": "https://example.invalid/a",
                }
            })

    def test_checkpoint_stat_verification_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"ok": True}) + "\n")
            manifest = {"model_files": {path.name: file_identity(path)}}
            verify_checkpoint_stats(manifest)
            path.write_text(json.dumps({"ok": False}) + "\n")
            with self.assertRaisesRegex(ProvenanceError, "checkpoint identity mismatch"):
                verify_checkpoint_stats(manifest)

    def test_all_campaign_manifests_are_strict_json(self):
        root = Path.home() / "experiments/glm53-flash-mlx-opt-v1/manifests"
        for path in root.rglob("*.json"):
            json.loads(path.read_text())
        self.assertFalse(list(root.rglob("*.yaml")))
        self.assertFalse(list(root.rglob("*.yml")))


if __name__ == "__main__":
    unittest.main()
