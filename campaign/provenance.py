"""Capture and enforce content-addressed campaign provenance.

All expensive checkpoint hashing is explicit and happens before benchmark timing.
Per-load verification uses the recorded stat tuple and binds the requested model
root to the manifest's canonical root.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
import platform
import site
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

REQUIRED_SMALL_MODEL_FILES = (
    "config.json", "model.safetensors.index.json", "tokenizer.json",
    "tokenizer_config.json", "processor_config.json", "generation_config.json",
    "chat_template.jinja",
)
PINNED_DISTRIBUTIONS = ("mlx", "mlx-vlm", "transformers")
ALLOWED_ENVIRONMENT_KEYS = {
    "GLM53_CAMPAIGN_ROOT", "GLM53_MODEL_ROOT", "HOME", "LOGNAME", "PATH",
    "PYTHONNOUSERSITE", "PYTHONPATH", "PYTHONSAFEPATH", "TMPDIR", "USER",
    "VIRTUAL_ENV",
}
FORBIDDEN_ENVIRONMENT_PREFIXES = ("CONDA_", "METAL_", "MLX_")


class ProvenanceError(RuntimeError):
    """Raised when a campaign identity assertion fails."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_reference(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    return {
        "path": str(path.resolve(strict=True)),
        "canonical_sha256": canonical_sha256(value),
        "file_sha256": sha256_file(path),
    }


def write_content_addressed(directory: Path, kind: str, value: dict[str, Any]) -> Path:
    digest = canonical_sha256(value)
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{kind}-{digest}.json"
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if destination.exists() and destination.read_text() != payload:
        raise ProvenanceError(f"content-address collision: {destination}")
    temporary = destination.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(payload)
    os.replace(temporary, destination)
    return destination


def command(*args: str, check: bool = True) -> str:
    result = subprocess.run(args, check=check, text=True, capture_output=True)
    return result.stdout.strip()


def file_identity(path: Path) -> dict[str, Any]:
    canonical = path.resolve(strict=True)
    stat = canonical.stat()
    return {"path": str(canonical), "device": stat.st_dev, "inode": stat.st_ino,
            "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def git_identity(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)),
        "head": command("git", "-C", str(path), "rev-parse", "HEAD"),
        "tree": command("git", "-C", str(path), "rev-parse", "HEAD^{tree}"),
        "branch": command("git", "-C", str(path), "branch", "--show-current"),
        "status_porcelain_v2": command("git", "-C", str(path), "status", "--porcelain=v2").splitlines(),
    }


def parse_hf_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line:
            continue
        name, size, digest, url = line.split("|", 3)
        records[name] = {"size": int(size), "sha256": digest or None, "url": url}
    return records


def revision_from_hf_url(url: str) -> str:
    parts = [part for part in urlparse(url).path.split("/") if part]
    try:
        return parts[parts.index("resolve") + 1]
    except (ValueError, IndexError) as exc:
        raise ProvenanceError(f"cannot extract revision from URL: {url}") from exc


def verified_shard_evidence(model_root: Path, record: dict[str, Any], name: str,
                            status_dir: Path, *, rehash: bool) -> dict[str, Any]:
    shard = (model_root / name).resolve(strict=True)
    status_path = status_dir / f"{name}.json"
    log_path = status_dir / f"{name}.log"
    expected_digest = record.get("sha256")
    if not expected_digest:
        raise ProvenanceError(f"HF manifest lacks SHA-256 for {name}")
    status = json.loads(status_path.read_text())
    identity = file_identity(shard)
    if status.get("status") != "complete":
        raise ProvenanceError(f"download status is not complete for {name}")
    if Path(status.get("dest", "")).resolve() != shard:
        raise ProvenanceError(f"download status destination mismatch for {name}")
    if status.get("url") != record["url"]:
        raise ProvenanceError(f"download status URL/revision mismatch for {name}")
    revision = revision_from_hf_url(record["url"])
    if revision_from_hf_url(status["url"]) != revision:
        raise ProvenanceError(f"download revision mismatch for {name}")
    if status.get("expected_bytes") != record["size"] or status.get("current_bytes") != record["size"]:
        raise ProvenanceError(f"download status byte count mismatch for {name}")
    if identity["size"] != record["size"]:
        raise ProvenanceError(f"size mismatch for {name}")
    log_text = log_path.read_text(errors="replace")
    if "Checksum verified" not in log_text:
        raise ProvenanceError(f"download log lacks checksum verification for {name}")
    actual_digest = sha256_file(shard) if rehash else expected_digest
    if actual_digest != expected_digest:
        raise ProvenanceError(f"full shard digest mismatch for {name}")
    return {
        **identity,
        "sha256": expected_digest,
        "sha256_mode": "computed_full_out_of_band" if rehash else "bound_download_status_url_revision_and_digest",
        "evidence": {
            "hf_manifest_url": record["url"], "revision": revision,
            "expected_sha256": expected_digest,
            "status_url": status["url"], "status_path": str(status_path.resolve()),
            "status_sha256": sha256_file(status_path), "status_updated": status.get("updated"),
            "log_path": str(log_path.resolve()), "log_sha256": sha256_file(log_path),
            "checksum_verified_marker": True,
            "full_rehash_performed": rehash,
        },
    }


def distribution_records() -> list[dict[str, Any]]:
    rows = []
    for dist in metadata.distributions():
        direct_text = dist.read_text("direct_url.json")
        rows.append({
            "name": dist.metadata["Name"].lower().replace("_", "-"),
            "version": dist.version,
            "direct_url": json.loads(direct_text) if direct_text else None,
        })
    return sorted(rows, key=lambda row: row["name"])


def read_distribution_lock(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def module_identity(name: str) -> dict[str, Any]:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve(strict=True)
    return {**file_identity(path), "sha256": sha256_file(path)}


def installed_distribution_identity(name: str) -> dict[str, Any]:
    dist = metadata.distribution(name)
    rows: list[dict[str, Any]] = []
    native: list[dict[str, Any]] = []
    for relative in sorted(dist.files or [], key=str):
        text = str(relative)
        if "__pycache__" in text or text.endswith(".pyc"):
            continue
        path = Path(dist.locate_file(relative)).resolve(strict=True)
        if not path.is_file():
            continue
        row = {"relative_path": text, "size": path.stat().st_size, "sha256": sha256_file(path)}
        rows.append(row)
        if path.suffix in {".so", ".dylib", ".metallib"}:
            native.append({**file_identity(path), "sha256": row["sha256"]})
    direct_text = dist.read_text("direct_url.json")
    return {
        "name": name, "version": dist.version,
        "direct_url": json.loads(direct_text) if direct_text else None,
        "installed_payload_sha256": hashlib.sha256(canonical_json_bytes(rows)).hexdigest(),
        "files": rows, "native_binaries": native,
    }


def relevant_environment() -> dict[str, str]:
    return {key: value for key, value in sorted(os.environ.items())
            if key in ALLOWED_ENVIRONMENT_KEYS or key.startswith(("MLX_", "METAL_", "CONDA_"))}


def verify_clean_environment(runtime: Path, venv: Path) -> None:
    forbidden = [key for key in os.environ if key == "PYTHONHOME" or key.startswith(FORBIDDEN_ENVIRONMENT_PREFIXES)]
    if forbidden:
        raise ProvenanceError(f"forbidden environment variables retained: {sorted(forbidden)}")
    expected = {
        "PYTHONNOUSERSITE": "1", "PYTHONSAFEPATH": "1",
        "PYTHONPATH": str(runtime), "VIRTUAL_ENV": str(venv),
    }
    for key, value in expected.items():
        if os.environ.get(key) != value:
            raise ProvenanceError(f"{key} must equal {value!r}")
    if site.ENABLE_USER_SITE is not False:
        raise ProvenanceError("Python user site is enabled")


def verify_sys_path(runtime: Path, venv: Path) -> None:
    resolved = [Path(row or os.getcwd()).resolve() for row in sys.path]
    if resolved.count(runtime) != 1:
        raise ProvenanceError(f"runtime must occur exactly once on sys.path: {sys.path!r}")
    campaign_worktrees = runtime.parents[1]
    escaped = [row for row in resolved if campaign_worktrees in row.parents and runtime not in (row, *row.parents)]
    if escaped:
        raise ProvenanceError(f"another campaign worktree is on sys.path: {escaped}")
    user_site = Path(site.getusersitepackages()).resolve()
    if any(row == user_site or user_site in row.parents for row in resolved):
        raise ProvenanceError("user site appears on sys.path")
    if Path(os.getcwd()).resolve() in resolved:
        raise ProvenanceError("CWD injection remains on sys.path; use Python safe-path mode")
    if Path(sys.prefix).resolve() != venv:
        raise ProvenanceError(f"sys.prefix escaped selected venv: {sys.prefix}")


def capture_model_manifest(model_root: Path, hf_manifest: Path, status_dir: Path,
                           *, full_rehash: bool) -> dict[str, Any]:
    model_root = model_root.resolve(strict=True)
    records = parse_hf_manifest(hf_manifest)
    files: dict[str, Any] = {}
    for name in REQUIRED_SMALL_MODEL_FILES:
        if name not in records:
            raise ProvenanceError(f"HF manifest lacks required file {name}")
        path = (model_root / name).resolve(strict=True)
        if model_root not in path.parents:
            raise ProvenanceError(f"model file escaped root: {path}")
        identity = file_identity(path)
        if identity["size"] != records[name]["size"]:
            raise ProvenanceError(f"size mismatch for {name}")
        files[name] = {**identity, "sha256": sha256_file(path),
                       "sha256_mode": "computed_outside_timing_block"}
    shards = sorted(model_root.glob("model-*.safetensors"))
    if not shards:
        raise ProvenanceError("no safetensor shards found")
    for shard in shards:
        if shard.name not in records:
            raise ProvenanceError(f"HF manifest lacks {shard.name}")
        files[shard.name] = verified_shard_evidence(
            model_root, records[shard.name], shard.name, status_dir, rehash=full_rehash)
    return {
        "schema_version": 2, "kind": "model_provenance",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_root": str(model_root),
        "hf_manifest": {**file_identity(hf_manifest), "sha256": sha256_file(hf_manifest)},
        "full_rehash": {
            "performed": full_rehash,
            "completed_at_utc": datetime.now(timezone.utc).isoformat() if full_rehash else None,
            "file_count": len(files), "total_bytes": sum(row["size"] for row in files.values()),
        },
        "files": files,
    }


def capture_environment_manifest(runtime: Path, reference: Path, venv: Path,
                                 lock_path: Path, model_reference: dict[str, Any]) -> dict[str, Any]:
    runtime = runtime.resolve(strict=True); reference = reference.resolve(strict=True); venv = venv.resolve(strict=True)
    if runtime == reference or runtime in reference.parents or reference in runtime.parents:
        raise ProvenanceError("campaign runtime and reference checkout must be distinct")
    expected_executable = (venv / "bin/python").absolute()
    if Path(sys.executable).absolute() != expected_executable:
        raise ProvenanceError(f"unexpected sys.executable: {sys.executable}")
    verify_clean_environment(runtime, venv); verify_sys_path(runtime, venv)
    runtime_identity = git_identity(runtime)
    if runtime_identity["status_porcelain_v2"]:
        raise ProvenanceError("campaign runtime worktree is dirty")
    imports = {name: module_identity(name) for name in ("glm53_flash_mlx", "mlx.core", "mlx_vlm", "transformers")}
    if runtime not in Path(imports["glm53_flash_mlx"]["path"]).parents:
        raise ProvenanceError("runtime import escaped worktree")
    for name in ("mlx.core", "mlx_vlm", "transformers"):
        if venv not in Path(imports[name]["path"]).parents:
            raise ProvenanceError(f"{name} import escaped immutable venv")
    distributions = distribution_records()
    locked = read_distribution_lock(lock_path)
    if distributions != locked:
        raise ProvenanceError("installed distribution set does not exactly match stock lock")
    import mlx.core as mx
    python_real = Path(sys.executable).resolve(strict=True)
    return {
        "schema_version": 2, "kind": "environment_provenance",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": {"hostname": socket.gethostname(), "platform": platform.platform(),
                 "macos_product_version": command("sw_vers", "-productVersion"),
                 "macos_build": command("sw_vers", "-buildVersion"), "machine": platform.machine(),
                 "physical_memory_bytes": int(command("sysctl", "-n", "hw.memsize")),
                 "mlx_device_info": mx.device_info()},
        "runtime": runtime_identity, "reference_checkout": git_identity(reference),
        "python": {"executable": str(expected_executable), "executable_real": str(python_real),
                   "executable_sha256": sha256_file(python_real), "version": sys.version,
                   "venv": str(venv), "prefix": str(Path(sys.prefix).resolve())},
        "lock": {**file_identity(lock_path), "sha256": sha256_file(lock_path)},
        "imports": imports, "distributions": distributions,
        "installed_wheel_identities": {name: installed_distribution_identity(name) for name in PINNED_DISTRIBUTIONS},
        "sys_path": list(sys.path), "environment": relevant_environment(),
        "user_site_disabled": site.ENABLE_USER_SITE is False,
        "model_manifest": model_reference,
    }


def verify_model_root(manifest: dict[str, Any], requested_root: Path) -> Path:
    expected_root = Path(manifest["model_root"]).resolve(strict=True)
    actual_root = requested_root.resolve(strict=True)
    if actual_root != expected_root:
        raise ProvenanceError(f"--model-root {actual_root} != provenance model_root {expected_root}")
    for name, record in manifest["files"].items():
        path = Path(record["path"]).resolve(strict=True)
        if path.parent != expected_root or path.name != name:
            raise ProvenanceError(f"model file is not an immediate child of model root: {path}")
    return expected_root


def verify_checkpoint_stats(manifest: dict[str, Any], requested_root: Path | None = None) -> None:
    files = manifest.get("files", manifest.get("model_files"))
    if files is None:
        raise ProvenanceError("model manifest lacks files")
    if requested_root is not None:
        verify_model_root(manifest, requested_root)
    for name, expected in files.items():
        current = file_identity(Path(expected["path"]))
        for field in ("path", "device", "inode", "size", "mtime_ns"):
            if current[field] != expected[field]:
                raise ProvenanceError(f"checkpoint identity mismatch for {name}.{field}; rehash outside timing")


def verify_runtime_manifest(manifest: dict[str, Any], runtime: Path) -> None:
    runtime = runtime.resolve(strict=True); venv = Path(manifest["python"].get("venv", manifest["python"].get("venv_alias"))).resolve(strict=True)
    current = git_identity(runtime)
    for field in ("path", "head", "tree", "branch"):
        if current[field] != manifest["runtime"][field]:
            raise ProvenanceError(f"runtime {field} mismatch")
    if current["status_porcelain_v2"]:
        raise ProvenanceError("runtime worktree is dirty")
    expected_executable = Path(manifest["python"].get("executable", manifest["python"].get("executable_alias"))).absolute()
    if Path(sys.executable).absolute() != expected_executable:
        raise ProvenanceError("interpreter path mismatch")
    real = Path(sys.executable).resolve(strict=True)
    if str(real) != manifest["python"]["executable_real"] or sha256_file(real) != manifest["python"]["executable_sha256"]:
        raise ProvenanceError("interpreter real path/hash mismatch")
    verify_clean_environment(runtime, venv); verify_sys_path(runtime, venv)
    if sha256_file(Path(manifest["lock"]["path"])) != manifest["lock"]["sha256"]:
        raise ProvenanceError("lock hash mismatch")
    if distribution_records() != manifest["distributions"]:
        raise ProvenanceError("distribution set mismatch")
    for name, expected in manifest["imports"].items():
        observed = module_identity(name)
        if observed["path"] != expected["path"] or observed["sha256"] != expected["sha256"]:
            raise ProvenanceError(f"import identity mismatch for {name}")
    for name, expected in manifest["installed_wheel_identities"].items():
        observed = installed_distribution_identity(name)
        if observed["installed_payload_sha256"] != expected["installed_payload_sha256"]:
            raise ProvenanceError(f"installed payload mismatch for {name}")
        if observed["native_binaries"] != expected["native_binaries"]:
            raise ProvenanceError(f"native MLX/wheel identity mismatch for {name}")
    if relevant_environment() != manifest["environment"] or list(sys.path) != manifest["sys_path"]:
        raise ProvenanceError("environment or sys.path differs from content-addressed manifest")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command_name", required=True)
    model = sub.add_parser("capture-model")
    model.add_argument("--model-root", type=Path, required=True); model.add_argument("--hf-manifest", type=Path, required=True)
    model.add_argument("--download-status-dir", type=Path, required=True); model.add_argument("--output-dir", type=Path, required=True)
    model.add_argument("--full-rehash", action="store_true")
    env = sub.add_parser("capture-environment")
    env.add_argument("--runtime", type=Path, required=True); env.add_argument("--reference-checkout", type=Path, required=True)
    env.add_argument("--venv", type=Path, required=True); env.add_argument("--lock", type=Path, required=True)
    env.add_argument("--model-manifest", type=Path, required=True); env.add_argument("--output-dir", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--runtime", type=Path, required=True); verify.add_argument("--environment-manifest", type=Path, required=True)
    verify.add_argument("--model-manifest", type=Path, required=True); verify.add_argument("--model-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command_name == "capture-model":
        value = capture_model_manifest(args.model_root, args.hf_manifest.resolve(strict=True), args.download_status_dir.resolve(strict=True), full_rehash=args.full_rehash)
        path = write_content_addressed(args.output_dir, "model", value); print(path); print(canonical_sha256(value))
    elif args.command_name == "capture-environment":
        model_ref = manifest_reference(args.model_manifest)
        value = capture_environment_manifest(args.runtime, args.reference_checkout, args.venv, args.lock.resolve(strict=True), model_ref)
        path = write_content_addressed(args.output_dir, "environment", value); print(path); print(canonical_sha256(value))
    else:
        environment = json.loads(args.environment_manifest.read_text()); model_value = json.loads(args.model_manifest.read_text())
        if manifest_reference(args.model_manifest)["canonical_sha256"] != environment["model_manifest"]["canonical_sha256"]:
            raise ProvenanceError("environment/model manifest hash mismatch")
        verify_model_root(model_value, args.model_root); verify_checkpoint_stats(model_value, args.model_root)
        verify_runtime_manifest(environment, args.runtime)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
