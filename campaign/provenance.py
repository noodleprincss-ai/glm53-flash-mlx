"""Capture and enforce campaign provenance without paging model shards."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REQUIRED_SMALL_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "generation_config.json",
    "chat_template.jinja",
)


class ProvenanceError(RuntimeError):
    """Raised when a campaign identity assertion fails."""


def sha256_file(path: Path, block_size: int = 4 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, check=check, text=True, capture_output=True)
    return result.stdout.strip()


def file_identity(path: Path) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    stat = canonical.stat()
    return {
        "path": str(canonical),
        "device": stat.st_dev,
        "inode": stat.st_ino,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def git_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "tree": command("git", "-C", str(path), "rev-parse", "HEAD^{tree}"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status_porcelain_v2": command(
            "git", "-C", str(path), "status", "--porcelain=v2"
        ).splitlines(),
    }


def parse_hf_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        name, size, digest, url = line.split("|", 3)
        records[name] = {
            "size": int(size),
            "sha256": digest or None,
            "url": url,
        }
    return records


def verified_shard_evidence(
    model_root: Path, record: dict[str, Any], name: str, status_dir: Path
) -> dict[str, Any]:
    shard = model_root / name
    status_path = status_dir / f"{name}.json"
    log_path = status_dir / f"{name}.log"
    if not record.get("sha256"):
        raise ProvenanceError(f"HF manifest lacks SHA-256 for {name}")
    status = json.loads(status_path.read_text())
    identity = file_identity(shard)
    if status.get("status") != "complete":
        raise ProvenanceError(f"download status is not complete for {name}")
    if Path(status.get("dest", "")).resolve() != shard.resolve():
        raise ProvenanceError(f"download status destination mismatch for {name}")
    if status.get("expected_bytes") != record["size"] or identity["size"] != record["size"]:
        raise ProvenanceError(f"size mismatch for {name}")
    log_text = log_path.read_text(errors="replace")
    if "Checksum verified" not in log_text:
        raise ProvenanceError(f"download log lacks checksum verification for {name}")
    return {
        **identity,
        "sha256": record["sha256"],
        "sha256_mode": "reused_verified_download_evidence",
        "evidence": {
            "hf_manifest_url": record["url"],
            "status_path": str(status_path.resolve()),
            "status_sha256": sha256_file(status_path),
            "status_updated": status.get("updated"),
            "log_path": str(log_path.resolve()),
            "log_sha256": sha256_file(log_path),
            "checksum_verified_marker": True,
        },
    }


def distribution_records() -> list[dict[str, Any]]:
    records = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        direct_text = dist.read_text("direct_url.json")
        direct = json.loads(direct_text) if direct_text else None
        records.append({"name": name, "version": dist.version, "direct_url": direct})
    return sorted(records, key=lambda row: row["name"])


def module_identity(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve(strict=True)
    return {**file_identity(path), "sha256": sha256_file(path)}


def installed_distribution_identity(name: str) -> dict[str, Any]:
    """Content-address an installed wheel payload and its native binaries."""
    dist = metadata.distribution(name)
    rows: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for relative in sorted(dist.files or [], key=str):
        relative_text = str(relative)
        if "__pycache__" in relative_text or relative_text.endswith(".pyc"):
            continue
        path = Path(dist.locate_file(relative)).resolve(strict=True)
        if not path.is_file():
            continue
        row = {"relative_path": relative_text, "size": path.stat().st_size, "sha256": sha256_file(path)}
        rows.append(row)
        if path.suffix in {".so", ".dylib", ".metallib"}:
            native.append({**file_identity(path), "sha256": row["sha256"]})
    payload = hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    direct_text = dist.read_text("direct_url.json")
    return {
        "name": name,
        "version": dist.version,
        "direct_url": json.loads(direct_text) if direct_text else None,
        "installed_payload_sha256": payload,
        "files": rows,
        "native_binaries": native,
    }


def capture_manifest(args: argparse.Namespace) -> dict[str, Any]:
    runtime = args.runtime.resolve(strict=True)
    reference = args.reference_checkout.resolve(strict=True)
    model_root = args.model_root.resolve(strict=True)
    venv_alias = args.venv_alias.absolute()
    venv_target = venv_alias.resolve(strict=True)
    hf_manifest = args.hf_manifest.resolve(strict=True)
    status_dir = args.download_status_dir.resolve(strict=True)

    if runtime == reference or runtime in reference.parents or reference in runtime.parents:
        raise ProvenanceError("campaign runtime and reference checkout must be distinct")
    if runtime == model_root or runtime in model_root.parents or model_root in runtime.parents:
        raise ProvenanceError("model root must not be in a campaign worktree")
    if Path(sys.executable).absolute() != venv_alias / "bin/python":
        raise ProvenanceError(
            f"sys.executable {sys.executable} is not selected alias {venv_alias / 'bin/python'}"
        )

    records = parse_hf_manifest(hf_manifest)
    model_files: dict[str, Any] = {}
    for name in REQUIRED_SMALL_MODEL_FILES:
        if name not in records:
            raise ProvenanceError(f"HF manifest lacks required file {name}")
        path = model_root / name
        identity = file_identity(path)
        if identity["size"] != records[name]["size"]:
            raise ProvenanceError(f"size mismatch for {name}")
        model_files[name] = {
            **identity,
            "sha256": sha256_file(path),
            "sha256_mode": "computed_outside_timing_block",
        }
    shards = sorted(model_root.glob("model-*.safetensors"))
    if not shards:
        raise ProvenanceError("no safetensor shards found")
    for shard in shards:
        if shard.name not in records:
            raise ProvenanceError(f"HF manifest lacks {shard.name}")
        model_files[shard.name] = verified_shard_evidence(
            model_root, records[shard.name], shard.name, status_dir
        )

    runtime_identity = git_identity(runtime)
    if runtime_identity["status_porcelain_v2"]:
        raise ProvenanceError("campaign runtime worktree is dirty")
    reference_identity = git_identity(reference)

    imports = {
        name: module_identity(name)
        for name in ("glm53_flash_mlx", "mlx.core", "mlx_vlm", "transformers")
    }
    runtime_import = Path(imports["glm53_flash_mlx"]["path"])
    if runtime not in runtime_import.parents:
        raise ProvenanceError(f"runtime import escaped worktree: {runtime_import}")
    for name in ("mlx.core", "mlx_vlm", "transformers"):
        imported = Path(imports[name]["path"])
        if venv_target not in imported.parents:
            raise ProvenanceError(f"{name} import escaped immutable venv: {imported}")

    python_real = Path(sys.executable).resolve(strict=True)
    relevant_environment = {
        key: value
        for key, value in sorted(os.environ.items())
        if key in {"HOME", "LOGNAME", "PATH", "PYTHONNOUSERSITE", "PYTHONPATH", "TMPDIR", "USER", "VIRTUAL_ENV"}
        or key.startswith(("MLX_", "METAL_"))
    }
    forbidden = [key for key in ("PYTHONHOME", "CONDA_PREFIX") if key in os.environ]
    if forbidden:
        raise ProvenanceError(f"forbidden environment variables retained: {forbidden}")
    if os.environ.get("PYTHONNOUSERSITE") != "1":
        raise ProvenanceError("PYTHONNOUSERSITE must equal 1")
    if os.environ.get("PYTHONPATH") != str(runtime):
        raise ProvenanceError("PYTHONPATH is not the exact campaign worktree")

    import mlx.core as mx

    return {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "macos_product_version": command("sw_vers", "-productVersion"),
            "macos_build": command("sw_vers", "-buildVersion"),
            "machine": platform.machine(),
            "physical_memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
            "mlx_device_info": mx.device_info(),
        },
        "runtime": runtime_identity,
        "reference_checkout": reference_identity,
        "python": {
            "executable_alias": str(Path(sys.executable).absolute()),
            "executable_real": str(python_real),
            "executable_sha256": sha256_file(python_real),
            "version": sys.version,
            "venv_alias": str(venv_alias),
            "venv_target": str(venv_target),
        },
        "imports": imports,
        "distributions": distribution_records(),
        "installed_wheel_identities": {
            name: installed_distribution_identity(name)
            for name in ("mlx", "mlx-vlm", "transformers")
        },
        "sys_path": sys.path,
        "environment": relevant_environment,
        "user_site_disabled": True,
        "model_root": str(model_root),
        "hf_manifest": {
            **file_identity(hf_manifest),
            "sha256": sha256_file(hf_manifest),
        },
        "model_files": model_files,
    }


def verify_checkpoint_stats(manifest: dict[str, Any]) -> None:
    for name, expected in manifest["model_files"].items():
        current = file_identity(Path(expected["path"]))
        for field in ("path", "device", "inode", "size", "mtime_ns"):
            if current[field] != expected[field]:
                raise ProvenanceError(
                    f"checkpoint identity mismatch for {name}.{field}: "
                    f"{current[field]!r} != {expected[field]!r}; rehash outside timing"
                )


def verify_runtime_manifest(manifest: dict[str, Any], runtime: Path) -> None:
    current = git_identity(runtime)
    for field in ("path", "head", "tree"):
        if current[field] != manifest["runtime"][field]:
            raise ProvenanceError(f"runtime {field} mismatch")
    if current["status_porcelain_v2"]:
        raise ProvenanceError("runtime worktree is dirty")
    for name, expected in manifest["imports"].items():
        current_import = module_identity(name)
        if current_import["path"] != expected["path"] or current_import["sha256"] != expected["sha256"]:
            raise ProvenanceError(f"import identity mismatch for {name}")
    for name, expected in manifest["installed_wheel_identities"].items():
        current_dist = installed_distribution_identity(name)
        if current_dist["installed_payload_sha256"] != expected["installed_payload_sha256"]:
            raise ProvenanceError(f"installed wheel payload mismatch for {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--runtime", type=Path, required=True)
    capture.add_argument("--reference-checkout", type=Path, required=True)
    capture.add_argument("--venv-alias", type=Path, required=True)
    capture.add_argument("--model-root", type=Path, required=True)
    capture.add_argument("--hf-manifest", type=Path, required=True)
    capture.add_argument("--download-status-dir", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--runtime", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "capture":
        manifest = capture_manifest(args)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    else:
        manifest = json.loads(args.manifest.read_text())
        verify_checkpoint_stats(manifest)
        verify_runtime_manifest(manifest, args.runtime.resolve(strict=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
