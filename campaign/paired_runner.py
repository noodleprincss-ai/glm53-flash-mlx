"""Sequential parent/candidate paired-order process orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import sys
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from campaign.locking import CampaignLock
from campaign.provenance import sha256_file


PressureFn = Callable[[], dict[str, Any]]
SleepFn = Callable[[float], None]
HARD_COUNTERS = ("swap_used_bytes", "pageouts", "compressor_pages")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def pressure() -> dict[str, Any]:
    from campaign.h0_benchmark import pressure_snapshot
    return pressure_snapshot()


def descendants(pid: int) -> list[int]:
    output = subprocess.run(["pgrep", "-P", str(pid)], text=True, capture_output=True).stdout
    direct = [int(row) for row in output.split() if row.isdigit()]
    return direct + [child for item in direct for child in descendants(item)]


def wait_teardown(pid: int, known_children: list[int], timeout: float) -> dict[str, Any]:
    """Wait for the benchmark process tree and return evidence instead of raising."""
    deadline = time.monotonic() + timeout
    watched = sorted(set([pid, *known_children]))
    alive: list[int] = watched
    while time.monotonic() < deadline:
        alive = [item for item in watched
                 if subprocess.run(["kill", "-0", str(item)], capture_output=True).returncode == 0]
        if not alive:
            break
        time.sleep(.25)
    return {"process_gone": not alive, "watched_pids": watched, "alive_pids": alive,
            "timeout_seconds": timeout}


def _positive_every_interval(samples: list[dict[str, Any]], key: str) -> bool:
    return len(samples) >= 4 and all(
        samples[index + 1][key] > samples[index][key]
        for index in range(len(samples) - 1)
    )


def assess_cooldown(
    samples: list[dict[str, Any]],
    *,
    process_gone: bool,
    max_swap_growth: int,
    max_pageout_growth: int,
    max_compressor_growth: int,
    min_free_percent: int,
    host_rss_ceiling_bytes: int,
) -> dict[str, Any]:
    """Classify cooldown telemetry without treating global page-ins as pressure.

    macOS page-ins are a monotonic, system-wide activity counter.  They are
    retained, including their interval trend, but can only corroborate a hard
    signal.  Teardown, swap/pageout/compressor, memory-pressure free percentage,
    and host RSS remain fail-closed.
    """
    reasons: list[str] = []
    valid = [sample for sample in samples if "capture_error" not in sample]
    if not process_gone:
        reasons.append("benchmark process tree did not tear down")
    if len(valid) != len(samples) or len(valid) < 2:
        reasons.append("cooldown pressure sampling incomplete")

    deltas: dict[str, int] = {}
    interval_deltas: dict[str, list[int]] = {}
    sustained: dict[str, bool] = {}
    keys = (*HARD_COUNTERS, "pageins", "combined_rss_bytes", "other_rss_bytes",
            "memory_free_percent")
    if len(valid) >= 2:
        for key in keys:
            if all(key in sample for sample in valid):
                deltas[key] = int(valid[-1][key]) - int(valid[0][key])
                interval_deltas[key] = [
                    int(valid[index + 1][key]) - int(valid[index][key])
                    for index in range(len(valid) - 1)
                ]
                sustained[key] = _positive_every_interval(valid, key)

        limits = {"swap_used_bytes": max_swap_growth, "pageouts": max_pageout_growth,
                  "compressor_pages": max_compressor_growth}
        for key, limit in limits.items():
            if deltas.get(key, 0) > limit:
                reasons.append(f"cooldown {key} growth exceeded limit: {deltas[key]} > {limit}")
            if sustained.get(key, False):
                reasons.append(f"sustained cooldown growth for {key}")

        low_free = [sample.get("memory_free_percent", 100) < min_free_percent for sample in valid]
        if low_free and all(low_free):
            reasons.append(
                f"sustained system memory pressure: free percentage below {min_free_percent}%"
            )

        for key in ("combined_rss_bytes", "other_rss_bytes"):
            over = [sample.get(key, 0) > host_rss_ceiling_bytes for sample in valid]
            if over and (over[-1] or all(over)):
                reasons.append(f"cooldown {key} exceeded host ceiling {host_rss_ceiling_bytes}")

    return {
        "samples": samples,
        "sample_count": len(samples),
        "deltas": deltas,
        "interval_deltas": interval_deltas,
        "sustained_growth": sustained,
        "pageins_role": "telemetry_and_corroboration_only",
        "pageins_corroborating": deltas.get("pageins", 0) > 0,
        "hard_failure": bool(reasons),
        "failure_reasons": reasons,
    }


def cooldown(
    samples_required: int,
    interval: float,
    *,
    process_gone: bool,
    max_swap_growth: int,
    max_pageout_growth: int,
    max_compressor_growth: int,
    min_free_percent: int,
    host_rss_ceiling_bytes: int,
    pressure_fn: PressureFn = pressure,
    sleep_fn: SleepFn = time.sleep,
) -> dict[str, Any]:
    """Collect every requested sample, retaining capture failures as evidence."""
    samples: list[dict[str, Any]] = []
    for index in range(samples_required + 1):
        if index:
            sleep_fn(interval)
        try:
            samples.append(pressure_fn())
        except BaseException as exc:
            samples.append({"captured_at_utc": utc_now(),
                            "capture_error": f"{type(exc).__name__}: {exc}"})
    return assess_cooldown(
        samples,
        process_gone=process_gone,
        max_swap_growth=max_swap_growth,
        max_pageout_growth=max_pageout_growth,
        max_compressor_growth=max_compressor_growth,
        min_free_percent=min_free_percent,
        host_rss_ceiling_bytes=host_rss_ceiling_bytes,
    )


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def persist_result(output: Path, emergency_dir: Path, result: dict[str, Any],
                   writer: Callable[[Path, dict[str, Any]], None] = atomic_write_json) -> BaseException | None:
    """Write the result atomically, with an independent emergency evidence path."""
    try:
        writer(output, result)
        return None
    except BaseException as exc:
        result["result_write_error"] = f"{type(exc).__name__}: {exc}"
        emergency = emergency_dir / f"{result['run_id']}-pid{os.getpid()}.json"
        result["emergency_result_path"] = str(emergency)
        try:
            atomic_write_json(emergency, result)
        except BaseException as emergency_exc:
            result["emergency_result_write_error"] = (
                f"{type(emergency_exc).__name__}: {emergency_exc}"
            )
            # The caller's captured stderr/log is the final evidence channel.
            sys.stderr.write(json.dumps({"unpersisted_orchestrator_result": result}, sort_keys=True) + "\n")
        return exc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--teardown-timeout", type=float, default=30)
    parser.add_argument("--cooldown-samples", type=int, default=3)
    parser.add_argument("--cooldown-interval", type=float, default=5)
    parser.add_argument("--max-swap-growth-mb", type=float, default=64)
    parser.add_argument("--max-pagein-growth", type=int, default=8192,
                        help="deprecated telemetry threshold; page-ins never fail alone")
    parser.add_argument("--max-pageout-growth", type=int, default=8192)
    parser.add_argument("--max-compressor-growth", type=int, default=8192)
    parser.add_argument("--min-free-percent", type=int, default=10)
    parser.add_argument("--host-rss-fraction", type=float, default=.90)
    parser.add_argument("--emergency-output-dir", type=Path)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    if spec.get("schema_version") != 1 or spec.get("orders") != [["parent", "candidate"], ["candidate", "parent"]]:
        raise RuntimeError("paired spec must declare parent→candidate and candidate→parent")
    if spec.get("counted_repetitions_per_variant_order", 0) < 1:
        raise RuntimeError("paired spec repetition count must be explicit")

    initial = pressure()
    physical_memory = int(initial["physical_memory_bytes"])
    host_rss_ceiling = int(physical_memory * args.host_rss_fraction)
    result: dict[str, Any] = {
        "schema_version": 2, "run_id": args.run_id, "status": "running",
        "started_at_utc": utc_now(), "spec_path": str(args.spec.resolve()),
        "spec_sha256": sha256_file(args.spec), "raw_runs": [],
        "cooldown_semantics": {
            "pageins": "telemetry_and_corroboration_only",
            "hard_signals": ["process_teardown", "host_rss_ceiling", "memory_free_percent",
                             "swap_growth", "pageout_growth", "compressor_growth"],
            "host_rss_ceiling_bytes": host_rss_ceiling,
            "min_free_percent": args.min_free_percent,
        },
    }
    error: BaseException | None = None
    lock_path = args.campaign_root / "locks" / "paired-orchestrator.lock"
    try:
        with CampaignLock(lock_path, args.run_id):
            try:
                for order_index, order in enumerate(spec["orders"], 1):
                    for repetition in range(1, spec["counted_repetitions_per_variant_order"] + 1):
                        for variant in order:
                            command = spec["variants"][variant]["command"]
                            if not isinstance(command, list) or not command:
                                raise RuntimeError("variant command must be an argv array")
                            row: dict[str, Any] = {
                                "order_index": order_index, "order": order,
                                "repetition": repetition, "variant": variant,
                                "command": command, "started_at_utc": utc_now(), "status": "running",
                                "result_path": spec["variants"][variant].get("result_path"),
                            }
                            result["raw_runs"].append(row)
                            try:
                                row["pressure_before"] = pressure()
                                proc = subprocess.Popen(
                                    command,
                                    cwd=spec["variants"][variant].get("neutral_cwd"),
                                    env=spec["variants"][variant].get("environment"),
                                )
                                row["pid"] = proc.pid
                                time.sleep(.05)
                                children = descendants(proc.pid)
                                row["known_children"] = children
                                row["exit_status"] = proc.wait()
                                teardown = wait_teardown(proc.pid, children, args.teardown_timeout)
                                row["teardown"] = teardown
                                assessment = cooldown(
                                    args.cooldown_samples,
                                    args.cooldown_interval,
                                    process_gone=teardown["process_gone"],
                                    max_swap_growth=int(args.max_swap_growth_mb * 1024**2),
                                    max_pageout_growth=args.max_pageout_growth,
                                    max_compressor_growth=args.max_compressor_growth,
                                    min_free_percent=args.min_free_percent,
                                    host_rss_ceiling_bytes=host_rss_ceiling,
                                )
                                row["cooldown_samples"] = assessment["samples"]
                                row["cooldown_assessment"] = {
                                    key: value for key, value in assessment.items() if key != "samples"
                                }
                                if row["result_path"]:
                                    path = Path(row["result_path"])
                                    row["result_sha256"] = sha256_file(path) if path.exists() else None
                                if assessment["hard_failure"]:
                                    raise RuntimeError("; ".join(assessment["failure_reasons"]))
                                if row["exit_status"] != 0:
                                    raise RuntimeError(f"{variant} exited {row['exit_status']}")
                                row["status"] = "complete"
                            except BaseException as exc:
                                row["status"] = "failed"
                                row["failure_reason"] = f"{type(exc).__name__}: {exc}"
                                raise
                            finally:
                                row["finished_at_utc"] = utc_now()
                result["status"] = "complete"
            except BaseException as exc:
                error = exc
                result["status"] = "failed"
                result["error"] = f"{type(exc).__name__}: {exc}"
                raise
    except BaseException as exc:
        if error is None:
            error = exc
            result["status"] = "failed"
            result["error"] = f"{type(exc).__name__}: {exc}"
        elif exc is not error:
            result["lock_cleanup_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["finished_at_utc"] = utc_now()

    emergency_dir = args.emergency_output_dir or args.campaign_root / "results" / "orchestrator-emergency"
    write_error = persist_result(args.output, emergency_dir, result)
    if write_error is not None and error is None:
        error = write_error
    if error is not None:
        raise error
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
