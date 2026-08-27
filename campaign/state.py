"""Canonical paired integration-state validation, selection, staging, and restore."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from campaign.locking import CampaignLock
from campaign.provenance import canonical_sha256, sha256_file


class StateError(RuntimeError):
    """Raised when an integration state is malformed or cannot be restored."""


STATE_REQUIRED = {
    "baseline_reproduction", "cherry_picks", "configuration", "environment_manifest",
    "fixtures", "flags", "id", "mlx", "model_manifest", "ordinal", "runtime",
    "schema_version", "status", "validation_run_ids", "wheel_venv",
}
POINTER_REQUIRED = {
    "canonical_content_sha256", "manifest_file_sha256", "manifest_path", "ordinal",
    "schema_version", "selected_at_utc", "selected_id",
}


def run_git(worktree: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(worktree), *args], text=True, capture_output=True)
    if result.returncode:
        raise StateError(f"git {' '.join(args)} failed in {worktree}: {result.stderr.strip()}")
    return result.stdout.strip()


def validate_file_reference(reference: dict[str, Any], canonical: bool = False) -> None:
    path = Path(reference["path"]).resolve(strict=True)
    if "sha256" in reference and sha256_file(path) != reference["sha256"]:
        raise StateError(f"file hash mismatch: {path}")
    if "file_sha256" in reference and sha256_file(path) != reference["file_sha256"]:
        raise StateError(f"file byte hash mismatch: {path}")
    if canonical or "canonical_sha256" in reference:
        value = json.loads(path.read_text())
        if canonical_sha256(value) != reference["canonical_sha256"]:
            raise StateError(f"canonical hash mismatch: {path}")


def validate_repository(reference: dict[str, Any], worktree: Path | None = None) -> None:
    if worktree is not None:
        if run_git(worktree, "rev-parse", "HEAD") != reference["sha"]:
            raise StateError(f"repository SHA mismatch: {worktree}")
        if run_git(worktree, "rev-parse", "HEAD^{tree}") != reference["tree"]:
            raise StateError(f"repository tree mismatch: {worktree}")
        if run_git(worktree, "branch", "--show-current") != reference["branch"]:
            raise StateError(f"repository branch mismatch: {worktree}")
        if run_git(worktree, "status", "--porcelain=v2"):
            raise StateError(f"repository is dirty: {worktree}")
        if reference.get("tag") and run_git(worktree, "rev-list", "-n", "1", reference["tag"]) != reference["sha"]:
            raise StateError(f"tag does not resolve to selected SHA: {reference['tag']}")


def validate_state(value: dict[str, Any], runtime_worktree: Path | None = None,
                   mlx_worktree: Path | None = None) -> str:
    if set(value) != STATE_REQUIRED:
        raise StateError(f"state schema keys differ: missing={STATE_REQUIRED-set(value)}, extra={set(value)-STATE_REQUIRED}")
    if value["schema_version"] != 2 or not isinstance(value["cherry_picks"], list):
        raise StateError("unsupported state schema")
    for key in ("baseline_reproduction", "configuration"):
        validate_file_reference(value[key])
    validate_file_reference(value["environment_manifest"], canonical=True)
    validate_file_reference(value["model_manifest"], canonical=True)
    for key in ("golden_128", "prompt_manifest"):
        validate_file_reference(value["fixtures"][key])
    venv = Path(value["wheel_venv"]["path"]).resolve(strict=True)
    if venv.is_symlink():
        raise StateError("selected venv may not be a symlink")
    lock = Path(value["wheel_venv"]["lock_path"]).resolve(strict=True)
    if sha256_file(lock) != value["wheel_venv"]["lock_sha256"]:
        raise StateError("selected lock hash mismatch")
    python = (venv / "bin/python").absolute()
    if not python.exists() or sha256_file(python.resolve(strict=True)) != value["wheel_venv"]["python_sha256"]:
        raise StateError("selected interpreter identity mismatch")
    validate_repository(value["runtime"], runtime_worktree)
    validate_repository(value["mlx"], mlx_worktree)
    return canonical_sha256(value)


def load_selected(pointer_path: Path, runtime_worktree: Path | None = None,
                  mlx_worktree: Path | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    pointer = json.loads(pointer_path.read_text())
    if set(pointer) != POINTER_REQUIRED or pointer["schema_version"] != 2:
        raise StateError("selected pointer schema mismatch")
    manifest_path = Path(pointer["manifest_path"]).resolve(strict=True)
    state = json.loads(manifest_path.read_text())
    digest = validate_state(state, runtime_worktree, mlx_worktree)
    if digest != pointer["canonical_content_sha256"] or sha256_file(manifest_path) != pointer["manifest_file_sha256"]:
        raise StateError("selected pointer hash mismatch")
    if state["id"] != pointer["selected_id"] or state["ordinal"] != pointer["ordinal"]:
        raise StateError("selected pointer identity mismatch")
    return pointer, state


def atomic_select(pointer_path: Path, manifest_path: Path, selected_at_utc: str) -> dict[str, Any]:
    state = json.loads(manifest_path.read_text()); digest = validate_state(state)
    pointer = {"schema_version": 2, "selected_id": state["id"], "ordinal": state["ordinal"],
               "manifest_path": str(manifest_path.resolve(strict=True)),
               "canonical_content_sha256": digest, "manifest_file_sha256": sha256_file(manifest_path),
               "selected_at_utc": selected_at_utc}
    temporary = pointer_path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, pointer_path)
    return pointer


def assert_clean_at(worktree: Path, sha: str, tree: str) -> None:
    if run_git(worktree, "rev-parse", "HEAD") != sha or run_git(worktree, "rev-parse", "HEAD^{tree}") != tree:
        raise StateError(f"failed to restore {worktree} to {sha}/{tree}")
    if run_git(worktree, "status", "--porcelain=v2"):
        raise StateError(f"restored worktree is dirty: {worktree}")


def restore_pair(runtime: Path, mlx: Path, state: dict[str, Any]) -> None:
    run_git(runtime, "reset", "--hard", state["runtime"]["sha"])
    run_git(mlx, "reset", "--hard", state["mlx"]["sha"])
    assert_clean_at(runtime, state["runtime"]["sha"], state["runtime"]["tree"])
    assert_clean_at(mlx, state["mlx"]["sha"], state["mlx"]["tree"])


def stage_pair(pointer: Path, runtime: Path, mlx: Path, runtime_target: str,
               mlx_target: str, lock_path: Path, simulate_failure_after: str | None) -> None:
    with CampaignLock(lock_path, "paired-state-stage"):
        _, preceding = load_selected(pointer, runtime, mlx)
        try:
            run_git(runtime, "reset", "--hard", runtime_target)
            if simulate_failure_after == "runtime":
                raise StateError("simulated failure after runtime staging")
            run_git(mlx, "reset", "--hard", mlx_target)
            if simulate_failure_after == "mlx":
                raise StateError("simulated failure after MLX staging")
            if run_git(runtime, "status", "--porcelain=v2") or run_git(mlx, "status", "--porcelain=v2"):
                raise StateError("staged worktree is dirty")
        except BaseException:
            restore_pair(runtime, mlx, preceding)
            raise


def main() -> int:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate"); validate.add_argument("--selected", type=Path, required=True)
    validate.add_argument("--runtime-worktree", type=Path); validate.add_argument("--mlx-worktree", type=Path)
    select = sub.add_parser("select"); select.add_argument("--selected", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True); select.add_argument("--selected-at-utc", required=True)
    stage = sub.add_parser("stage"); stage.add_argument("--selected", type=Path, required=True)
    stage.add_argument("--runtime-worktree", type=Path, required=True); stage.add_argument("--mlx-worktree", type=Path, required=True)
    stage.add_argument("--runtime-target", required=True); stage.add_argument("--mlx-target", required=True)
    stage.add_argument("--lock", type=Path, required=True); stage.add_argument("--simulate-failure-after", choices=("runtime", "mlx"))
    restore = sub.add_parser("restore"); restore.add_argument("--selected", type=Path, required=True)
    restore.add_argument("--runtime-worktree", type=Path, required=True); restore.add_argument("--mlx-worktree", type=Path, required=True)
    restore.add_argument("--lock", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "validate":
        pointer, _ = load_selected(args.selected, args.runtime_worktree, args.mlx_worktree); print(json.dumps(pointer, sort_keys=True))
    elif args.command == "select":
        print(json.dumps(atomic_select(args.selected, args.manifest, args.selected_at_utc), sort_keys=True))
    elif args.command == "stage":
        stage_pair(args.selected, args.runtime_worktree, args.mlx_worktree, args.runtime_target, args.mlx_target, args.lock, args.simulate_failure_after)
    else:
        with CampaignLock(args.lock, "paired-state-restore"):
            _, state = load_selected(args.selected); restore_pair(args.runtime_worktree, args.mlx_worktree, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
