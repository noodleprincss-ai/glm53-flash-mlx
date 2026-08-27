"""Focused manifest and fail-closed provenance tests."""

from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from campaign.provenance import (
    ProvenanceError, file_identity, parse_hf_manifest, verified_shard_evidence,
    verify_checkpoint_stats, verify_model_root,
)


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

    def test_model_root_is_bound_and_every_file_must_be_under_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"; root.mkdir()
            path = root / "config.json"; path.write_text("{}\n")
            manifest = {"model_root": str(root.resolve()), "files": {"config.json": file_identity(path)}}
            self.assertEqual(verify_model_root(manifest, root), root.resolve())
            other = Path(directory) / "other"; other.mkdir()
            with self.assertRaisesRegex(ProvenanceError, "--model-root"):
                verify_model_root(manifest, other)
            manifest["files"]["config.json"]["path"] = str((Path(directory) / "escaped.json").resolve())
            (Path(directory) / "escaped.json").write_text("{}\n")
            with self.assertRaisesRegex(ProvenanceError, "not an immediate child"):
                verify_model_root(manifest, root)

    def test_downloader_status_url_revision_and_digest_are_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"; status = Path(directory) / "status"
            root.mkdir(); status.mkdir(); shard = root / "model-00001.safetensors"; shard.write_bytes(b"abc")
            digest = hashlib.sha256(b"abc").hexdigest()
            url = "https://huggingface.co/o/r/resolve/revision123/model-00001.safetensors?download=true"
            (status / f"{shard.name}.json").write_text(json.dumps({
                "status": "complete", "url": url, "dest": str(shard),
                "expected_bytes": 3, "current_bytes": 3, "updated": "now"}) + "\n")
            (status / f"{shard.name}.log").write_text("Checksum verified\n")
            record = {"size": 3, "sha256": digest, "url": url}
            row = verified_shard_evidence(root, record, shard.name, status, rehash=True)
            self.assertTrue(row["evidence"]["full_rehash_performed"])
            bad = json.loads((status / f"{shard.name}.json").read_text()); bad["url"] = url.replace("revision123", "wrong")
            (status / f"{shard.name}.json").write_text(json.dumps(bad))
            with self.assertRaisesRegex(ProvenanceError, "URL/revision"):
                verified_shard_evidence(root, record, shard.name, status, rehash=False)

    def test_all_campaign_manifests_are_strict_json(self):
        root = Path.home() / "experiments/glm53-flash-mlx-opt-v1/manifests"
        for path in root.rglob("*.json"):
            json.loads(path.read_text())
        self.assertFalse(list(root.rglob("*.yaml")))
        self.assertFalse(list(root.rglob("*.yml")))


if __name__ == "__main__":
    unittest.main()
