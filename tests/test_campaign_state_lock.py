"""Recovery tests for campaign advisory locks and paired state staging."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
import unittest
from pathlib import Path

from campaign.locking import CampaignLock, LockError, audit_lock, process_start_identity, recover_lock
from campaign.provenance import canonical_sha256, sha256_file
from campaign.state import StateError, load_selected, stage_pair


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


class LockTests(unittest.TestCase):
    def test_advisory_lock_owner_and_explicit_stale_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "campaign.lock"
            with CampaignLock(path, "test"):
                self.assertEqual(audit_lock(path)["state"], "active")
                with self.assertRaises(LockError):
                    recover_lock(path, "must refuse live lock")
            self.assertEqual(audit_lock(path)["state"], "available")
            path.write_text(json.dumps({"state": "held", "hostname": platform.node(), "pid": 999999,
                                        "process_start_identity": "missing"}) + "\n")
            self.assertEqual(audit_lock(path)["state"], "stale_metadata")
            self.assertEqual(recover_lock(path, "unit-test stale owner")["state"], "recovered")


class StateRecoveryTests(unittest.TestCase):
    def test_simulated_partial_stage_restores_both_selected_tips(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); repos = []
            for name in ("runtime", "mlx"):
                repo = root / name; subprocess.check_call(["git", "init", "-q", "-b", "main", str(repo)])
                subprocess.check_call(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"])
                subprocess.check_call(["git", "-C", str(repo), "config", "user.name", "Test"])
                (repo / "value").write_text("selected\n"); subprocess.check_call(["git", "-C", str(repo), "add", "value"])
                subprocess.check_call(["git", "-C", str(repo), "commit", "-q", "-m", "selected"])
                selected = git(repo, "rev-parse", "HEAD"); tree = git(repo, "rev-parse", "HEAD^{tree}")
                git(repo, "tag", f"selected-{name}")
                (repo / "value").write_text("candidate\n"); subprocess.check_call(["git", "-C", str(repo), "commit", "-qam", "candidate"])
                candidate = git(repo, "rev-parse", "HEAD"); git(repo, "reset", "--hard", selected)
                repos.append((repo, selected, tree, candidate))
            def file_ref(name: str, content: str, canonical: bool = False):
                path = root / name; path.write_text(content)
                row = {"path": str(path), "sha256": sha256_file(path)}
                if canonical:
                    value = json.loads(content); row = {"path": str(path), "file_sha256": sha256_file(path),
                                                          "canonical_sha256": canonical_sha256(value)}
                return row
            env = file_ref("environment.json", "{}\n", True); model = file_ref("model.json", "{}\n", True)
            baseline = file_ref("baseline.json", "{}\n"); config = file_ref("config.json", "{}\n")
            prompt = file_ref("prompt.json", "{}\n"); golden = file_ref("golden.json", "{}\n")
            venv = root / "venv"; (venv / "bin").mkdir(parents=True); python = venv / "bin/python"; python.write_text("python\n"); python.chmod(0o555)
            lock = root / "lock.txt"; lock.write_text("lock\n")
            state = {"schema_version": 2, "id": "selected", "ordinal": 0, "status": "selected_control",
                "baseline_reproduction": baseline, "cherry_picks": [], "configuration": config,
                "environment_manifest": env, "model_manifest": model,
                "fixtures": {"fixture_set_sha256": "fixture", "golden_128": golden, "prompt_manifest": prompt},
                "flags": {}, "validation_run_ids": ["test"],
                "runtime": {"branch": "main", "sha": repos[0][1], "tree": repos[0][2], "tag": "selected-runtime"},
                "mlx": {"branch": "main", "sha": repos[1][1], "tree": repos[1][2], "tag": "selected-mlx"},
                "wheel_venv": {"path": str(venv), "lock_path": str(lock), "lock_sha256": sha256_file(lock),
                               "python_sha256": sha256_file(python), "wheel_payload_sha256": "payload", "native_binary_sha256": "native"}}
            manifest = root / "state.json"; manifest.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            pointer = root / "selected.json"; pointer.write_text(json.dumps({"schema_version": 2, "selected_id": "selected", "ordinal": 0,
                "manifest_path": str(manifest), "canonical_content_sha256": canonical_sha256(state),
                "manifest_file_sha256": sha256_file(manifest), "selected_at_utc": "now"}, indent=2) + "\n")
            load_selected(pointer, repos[0][0], repos[1][0])
            with self.assertRaisesRegex(StateError, "simulated failure"):
                stage_pair(pointer, repos[0][0], repos[1][0], repos[0][3], repos[1][3], root / "stage.lock", "runtime")
            self.assertEqual(git(repos[0][0], "rev-parse", "HEAD"), repos[0][1])
            self.assertEqual(git(repos[1][0], "rev-parse", "HEAD"), repos[1][1])


if __name__ == "__main__": unittest.main()
