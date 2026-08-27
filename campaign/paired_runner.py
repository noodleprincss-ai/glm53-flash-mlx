"""Sequential parent/candidate paired-order process orchestrator."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from campaign.locking import CampaignLock
from campaign.provenance import sha256_file


def utc_now() -> str: return datetime.now(timezone.utc).isoformat()


def pressure() -> dict[str, Any]:
    from campaign.h0_benchmark import pressure_snapshot
    return pressure_snapshot()


def descendants(pid: int) -> list[int]:
    output = subprocess.run(["pgrep", "-P", str(pid)], text=True, capture_output=True).stdout
    direct = [int(row) for row in output.split() if row.isdigit()]
    return direct + [child for item in direct for child in descendants(item)]


def wait_teardown(pid: int, known_children: list[int], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    watched = [pid, *known_children]
    while time.monotonic() < deadline:
        alive = [item for item in watched if subprocess.run(["kill", "-0", str(item)], capture_output=True).returncode == 0]
        if not alive: return
        time.sleep(.25)
    raise RuntimeError(f"process teardown did not stabilize: {alive}")


def cooldown(samples_required: int, interval: float, max_swap_growth: int,
             max_pagein_growth: int, max_compressor_growth: int) -> list[dict[str, Any]]:
    samples = [pressure()]
    while len(samples) < samples_required + 1:
        time.sleep(interval); samples.append(pressure())
    base = samples[0]; final = samples[-1]
    deltas = {key: final[key] - base[key] for key in ("swap_used_bytes", "pageins", "compressor_pages")}
    limits = {"swap_used_bytes": max_swap_growth, "pageins": max_pagein_growth,
              "compressor_pages": max_compressor_growth}
    if any(deltas[key] > limits[key] for key in limits):
        raise RuntimeError(f"cooldown pressure trend did not settle: {deltas}")
    # Growth in every consecutive interval is sustained rather than a transient tick.
    for key in limits:
        if len(samples) >= 4 and all(samples[i + 1][key] > samples[i][key] for i in range(len(samples) - 1)):
            raise RuntimeError(f"sustained cooldown growth for {key}")
    return samples


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True); parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--run-id", required=True)
    parser.add_argument("--teardown-timeout", type=float, default=30); parser.add_argument("--cooldown-samples", type=int, default=3)
    parser.add_argument("--cooldown-interval", type=float, default=5); parser.add_argument("--max-swap-growth-mb", type=float, default=64)
    parser.add_argument("--max-pagein-growth", type=int, default=8192); parser.add_argument("--max-compressor-growth", type=int, default=8192)
    args = parser.parse_args(); spec = json.loads(args.spec.read_text())
    if spec.get("schema_version") != 1 or spec.get("orders") != [["parent", "candidate"], ["candidate", "parent"]]:
        raise RuntimeError("paired spec must declare parent→candidate and candidate→parent")
    if spec.get("counted_repetitions_per_variant_order", 0) < 1:
        raise RuntimeError("paired spec repetition count must be explicit")
    result: dict[str, Any] = {"schema_version": 1, "run_id": args.run_id, "status": "running",
                              "started_at_utc": utc_now(), "spec_path": str(args.spec.resolve()),
                              "spec_sha256": sha256_file(args.spec), "raw_runs": []}
    error: BaseException | None = None
    with CampaignLock(args.campaign_root / "locks" / "paired-orchestrator.lock", args.run_id):
        try:
            for order_index, order in enumerate(spec["orders"], 1):
                for repetition in range(1, spec["counted_repetitions_per_variant_order"] + 1):
                    for variant in order:
                        command = spec["variants"][variant]["command"]
                        if not isinstance(command, list) or not command: raise RuntimeError("variant command must be an argv array")
                        started = utc_now(); before = pressure()
                        proc = subprocess.Popen(command, cwd=spec["variants"][variant].get("neutral_cwd"),
                                                env=spec["variants"][variant].get("environment"))
                        time.sleep(.05); children = descendants(proc.pid); exit_status = proc.wait()
                        wait_teardown(proc.pid, children, args.teardown_timeout)
                        cool = cooldown(args.cooldown_samples, args.cooldown_interval,
                                        int(args.max_swap_growth_mb * 1024**2), args.max_pagein_growth,
                                        args.max_compressor_growth)
                        row = {"order_index": order_index, "order": order, "repetition": repetition,
                               "variant": variant, "command": command, "started_at_utc": started,
                               "finished_at_utc": utc_now(), "exit_status": exit_status,
                               "pressure_before": before, "cooldown_samples": cool,
                               "result_path": spec["variants"][variant].get("result_path")}
                        if row["result_path"]:
                            path = Path(row["result_path"]); row["result_sha256"] = sha256_file(path) if path.exists() else None
                        result["raw_runs"].append(row)
                        if exit_status != 0: raise RuntimeError(f"{variant} exited {exit_status}")
            result["status"] = "complete"; result["finished_at_utc"] = utc_now()
        except BaseException as exc:
            error = exc; result["status"] = "failed"; result["error"] = f"{type(exc).__name__}: {exc}"; result["finished_at_utc"] = utc_now()
        finally:
            args.output.parent.mkdir(parents=True, exist_ok=True); temporary = args.output.with_suffix(f".tmp-{os.getpid()}")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n"); os.replace(temporary, args.output)
    if error: raise error
    return 0


if __name__ == "__main__": raise SystemExit(main())
