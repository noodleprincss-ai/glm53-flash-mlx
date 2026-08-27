"""Acceptance-capable H0 measurement with strict provenance and resource gates."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import re
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from campaign.instrumented_prefill import instrumented_prefill
from campaign.locking import CampaignLock
from campaign.provenance import (
    canonical_sha256, manifest_reference, sha256_file, verify_checkpoint_stats,
    verify_model_root, verify_runtime_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def command(*args: str, allow_failure: bool = False) -> str:
    result = subprocess.run(args, text=True, capture_output=True)
    if result.returncode and not allow_failure:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(args)}: {result.stderr.strip()}")
    return result.stdout.strip()


def listeners() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for port in (8000, 8001, 8062):
        raw = command("lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc", allow_failure=True)
        rows: list[dict[str, Any]] = []; current: dict[str, Any] = {}
        for line in raw.splitlines():
            if line.startswith("p"):
                if current: rows.append(current)
                current = {"pid": int(line[1:])}
            elif line.startswith("c"): current["command"] = line[1:]
        if current: rows.append(current)
        result[str(port)] = rows
    return result


def parse_vm_stat(text: str) -> dict[str, int]:
    rows: dict[str, int] = {}
    for line in text.splitlines()[1:]:
        if ":" not in line: continue
        key, value = line.split(":", 1)
        digits = re.sub(r"[^0-9]", "", value)
        if digits: rows[key.strip().lower().replace(" ", "_")] = int(digits)
    return rows


def swap_used_bytes(text: str) -> int:
    match = re.search(r"used = ([0-9.]+)([MG])", text)
    if not match: raise RuntimeError(f"cannot parse vm.swapusage: {text}")
    return int(float(match.group(1)) * (1024 ** (2 if match.group(2) == "M" else 3)))


def pressure_snapshot() -> dict[str, Any]:
    swap = command("sysctl", "vm.swapusage")
    vm_text = command("vm_stat")
    vm = parse_vm_stat(vm_text)
    return {
        "captured_at_utc": utc_now(), "swapusage": swap, "swap_used_bytes": swap_used_bytes(swap),
        "pageins": vm.get("pageins", 0), "pageouts": vm.get("pageouts", 0),
        "compressor_pages": vm.get("pages_occupied_by_compressor", 0),
        "vm_stat": vm_text,
    }


def process_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in command("ps", "-axo", "pid=,ppid=,rss=,comm=").splitlines():
        fields = line.strip().split(None, 3)
        if len(fields) == 4:
            rows.append({"pid": int(fields[0]), "ppid": int(fields[1]),
                         "rss_bytes": int(fields[2]) * 1024, "command": fields[3]})
    return rows


def power_snapshot() -> dict[str, Any]:
    custom = command("pmset", "-g", "custom", allow_failure=True)
    battery = command("pmset", "-g", "batt", allow_failure=True)
    low_power = None
    match = re.search(r"lowpowermode\s+(\d+)", custom)
    if match: low_power = int(match.group(1))
    return {"pmset_custom": custom, "pmset_battery": battery, "low_power_mode": low_power}


def system_snapshot() -> dict[str, Any]:
    rows = process_rows(); self_row = next((row for row in rows if row["pid"] == os.getpid()), None)
    others = [row for row in rows if row["pid"] != os.getpid()]
    return {
        "captured_at_utc": utc_now(), "listeners": listeners(), "power": power_snapshot(),
        "self_rss_bytes": 0 if self_row is None else self_row["rss_bytes"],
        "other_rss_bytes": sum(row["rss_bytes"] for row in others),
        "largest_other_processes": sorted(others, key=lambda row: row["rss_bytes"], reverse=True)[:12],
        "pressure": pressure_snapshot(),
    }


class ResourceSampler:
    def __init__(self, physical_bytes: int, interval: float):
        self.physical_bytes = physical_bytes; self.interval = interval
        self.samples: list[dict[str, Any]] = []; self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="glm53-resource-sampler", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                rows = process_rows(); own = next((row["rss_bytes"] for row in rows if row["pid"] == os.getpid()), 0)
                other = sum(row["rss_bytes"] for row in rows if row["pid"] != os.getpid())
                self.samples.append({"monotonic": time.monotonic(), "self_rss_bytes": own,
                                     "other_rss_bytes": other, "combined_rss_bytes": own + other})
            except Exception as exc:
                self.samples.append({"monotonic": time.monotonic(), "sampling_error": f"{type(exc).__name__}: {exc}"})
            self._stop.wait(self.interval)

    def __enter__(self) -> "ResourceSampler": self._thread.start(); return self
    def __exit__(self, *args: Any) -> None: self._stop.set(); self._thread.join(timeout=max(2.0, self.interval * 4))

    def summary(self) -> dict[str, Any]:
        good = [row for row in self.samples if "combined_rss_bytes" in row]
        peak = max(good, key=lambda row: row["combined_rss_bytes"]) if good else {}
        ceiling = int(self.physical_bytes * 0.90)
        return {"sample_interval_seconds": self.interval, "sample_count": len(good), "peak": peak,
                "host_residency_ceiling_bytes": ceiling,
                "host_residency_gate_passed": bool(good) and peak["combined_rss_bytes"] < ceiling,
                "raw_samples": self.samples}


def topk(logprobs: Any, k: int = 10) -> dict[str, Any]:
    import mlx.core as mx
    indices = mx.argpartition(-logprobs, kth=k - 1)[:k]; values = logprobs[indices]; mx.eval(indices, values)
    pairs = sorted(zip(indices.tolist(), values.tolist()), key=lambda pair: pair[1], reverse=True)
    return {"indices": [int(i) for i, _ in pairs], "logprobs": [float(v) for _, v in pairs]}


def prompt_ids(tokenizer: Any, length: int) -> tuple[Any, list[int], str]:
    import mlx.core as mx
    seed = ("Benchmarking long-context language model inference requires careful separation "
            "of prompt processing from autoregressive decoding. This deterministic prose "
            "supplies representative text tokens. ")
    base = tokenizer.encode(seed, add_special_tokens=False)
    ids = (base * ((length + len(base) - 1) // len(base)))[:length]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return mx.array([ids]), ids, digest


def consume_lookahead_window(generator: Iterator[Any], timed_decode: int,
                             synchronize: Any, clock: Any = time.perf_counter) -> dict[str, Any]:
    discarded_yields = 8
    discarded: list[int] = []
    first_token = None; first_logprobs = None; ttft_start = clock()
    for index in range(discarded_yields):
        token, logprobs = next(generator)
        if index == 0: first_token, first_logprobs = int(token), logprobs; ttft_stop = clock()
        discarded.append(int(token))
    synchronize()  # drain primed token 8
    decode_start = clock(); timed_tokens: list[int] = []; last_logprobs = None
    for _ in range(timed_decode):
        token, last_logprobs = next(generator); timed_tokens.append(int(token))
    synchronize()  # flush model step 136 before the stop boundary
    decode_stop = clock()
    lookahead_token, _ = next(generator)
    if hasattr(generator, "close"): generator.close()
    synchronize()
    return {"ttft_seconds": ttft_stop - ttft_start, "first_token_id": first_token,
            "first_logprobs": first_logprobs, "discarded_yield_indices": [0, 7],
            "discarded_token_ids": discarded, "timed_yield_indices": [8, 8 + timed_decode - 1],
            "timed_model_step_indices": [9, 8 + timed_decode], "timed_token_ids": timed_tokens,
            "timed_decode_steps": timed_decode, "decode_seconds": decode_stop - decode_start,
            "lookahead_token_index": 8 + timed_decode, "lookahead_token_id_excluded": int(lookahead_token),
            "last_logprobs": last_logprobs}


def public_metrics(model: Any, input_ids: Any, prefill_step_size: int, timed_decode: int = 128) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm.generate import generate_step
    generator = generate_step(input_ids, model, None, None, max_tokens=8 + timed_decode + 1,
                              temperature=0.0, prefill_step_size=prefill_step_size)
    row = consume_lookahead_window(generator, timed_decode, mx.synchronize)
    first_logprobs = row.pop("first_logprobs"); last_logprobs = row.pop("last_logprobs")
    row.update({"ttft_derived_prompt_rate": int(input_ids.size) / row["ttft_seconds"],
                "first_token_topk": topk(first_logprobs),
                "decode_tokens_per_second": timed_decode / row["decode_seconds"],
                "decode_ms_per_token": 1000 * row["decode_seconds"] / timed_decode,
                "final_timed_topk": topk(last_logprobs),
                "attached_context_tokens": int(input_ids.size) + 8})
    return row


def one_regime(model: Any, tokenizer: Any, length: int, step: int, phase: str,
               mlx_preferred: int, mlx_hard: int) -> dict[str, Any]:
    import mlx.core as mx
    input_ids, ids, prompt_sha = prompt_ids(tokenizer, length)
    mx.clear_cache(); gc.collect(); mx.reset_peak_memory()
    public = public_metrics(model, input_ids, step); public_peak = mx.get_peak_memory()
    mx.clear_cache(); gc.collect(); mx.reset_peak_memory()
    pure = instrumented_prefill(input_ids, model, None, None, temperature=0.0,
                                prefill_step_size=step, continuation_tokens=1)
    pure_peak = mx.get_peak_memory(); peak = max(public_peak, pure_peak)
    if peak > mlx_hard: raise RuntimeError(f"MLX hard peak exceeded: {peak} > {mlx_hard}")
    if pure.first_token_id != public["first_token_id"]: raise RuntimeError("instrumented/public first token mismatch")
    return {"phase": phase, "requested_prompt_tokens": length, "effective_prompt_tokens": int(input_ids.size),
            "prompt_token_ids": ids, "prompt_token_sha256": prompt_sha, "public": public,
            "pure_prefill": {"seconds": pure.prefill_seconds,
                "tokens_per_second": pure.prompt_rate_tokens_per_second, "first_token_id": pure.first_token_id,
                "final_prompt_topk": topk(pure.final_prompt_logprobs), "source_path": pure.source_path,
                "source_sha256": pure.source_sha256, "chunk_timings": [row.__dict__ for row in pure.chunk_timings]},
            "memory": {"public_peak_bytes": public_peak, "pure_prefill_peak_bytes": pure_peak,
                "maximum_mlx_peak_bytes": peak, "preferred_ceiling_bytes": mlx_preferred,
                "hard_ceiling_bytes": mlx_hard, "preferred_gate_passed": peak <= mlx_preferred,
                "hard_gate_passed": peak <= mlx_hard, "active_bytes": mx.get_active_memory(),
                "cache_bytes": mx.get_cache_memory()}}


def pressure_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int]:
    return {key: after[key] - before[key] for key in ("swap_used_bytes", "pageins", "pageouts", "compressor_pages")}


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n"); os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True); parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True); parser.add_argument("--environment-manifest", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-id", required=True); parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--repetitions", type=int, default=0,
                        help="counted repetitions; default 0 prevents accidental acceptance-like claims")
    parser.add_argument("--run-class", choices=("smoke", "screen", "acceptance", "confirmation", "reproduction"), default="smoke")
    parser.add_argument("--prefill-step-size", type=int, default=512); parser.add_argument("--wired-limit-gb", type=float, default=220.0)
    parser.add_argument("--expected-peak-gb", type=float, default=190.0); parser.add_argument("--rss-sample-seconds", type=float, default=1.0)
    parser.add_argument("--mlx-preferred-gb", type=float, default=205.0); parser.add_argument("--mlx-hard-gb", type=float, default=210.0)
    parser.add_argument("--max-swap-growth-mb", type=float, default=256.0); parser.add_argument("--max-pagein-growth", type=int, default=131072)
    parser.add_argument("--max-compressor-growth", type=int, default=65536)
    args = parser.parse_args()
    minimums = {"smoke": 1, "screen": 3, "acceptance": 5, "confirmation": 5, "reproduction": 1}
    if args.repetitions == 0: raise RuntimeError("--repetitions must be explicit; zero counted runs cannot support a claim")
    if args.run_class in {"screen", "acceptance", "confirmation"} and args.repetitions < minimums[args.run_class]:
        raise RuntimeError(f"{args.run_class} requires at least {minimums[args.run_class]} counted repetitions")
    campaign_root = args.campaign_root.resolve(strict=True); runtime = args.runtime.resolve(strict=True)
    environment_path = args.environment_manifest.resolve(strict=True); model_path = args.model_manifest.resolve(strict=True)
    environment = json.loads(environment_path.read_text()); model_manifest = json.loads(model_path.read_text())
    if manifest_reference(model_path) != environment["model_manifest"]: raise RuntimeError("environment/model content hash mismatch")
    model_root = verify_model_root(model_manifest, args.model_root); verify_checkpoint_stats(model_manifest, model_root)
    verify_runtime_manifest(environment, runtime)
    before = system_snapshot()
    if before["listeners"]["8001"]: raise RuntimeError("port 8001 must remain disabled")
    physical = environment["host"]["physical_memory_bytes"]
    projected = int(args.expected_peak_gb * 1e9) + before["other_rss_bytes"]
    if projected > int(physical * 0.90): raise RuntimeError("projected residency exceeds 90% physical-memory ceiling")
    data: dict[str, Any] = {
        "schema_version": 2, "status": "running", "run_id": args.run_id,
        "run_class": args.run_class, "acceptance_eligible": args.run_class in {"acceptance", "confirmation"},
        "started_at_utc": utc_now(), "hostname": platform.node(), "runtime": environment["runtime"],
        "model_root": str(model_root), "environment_manifest": manifest_reference(environment_path),
        "model_manifest": manifest_reference(model_path), "provenance_tuple_sha256": canonical_sha256({
            "runtime": environment["runtime"], "environment": manifest_reference(environment_path),
            "model": manifest_reference(model_path)}),
        "prefill_step_size": args.prefill_step_size, "counted_repetitions": args.repetitions,
        "gates": {"mlx_preferred_bytes": int(args.mlx_preferred_gb * 1e9),
                  "mlx_hard_bytes": int(args.mlx_hard_gb * 1e9),
                  "host_residency_bytes": int(physical * .90),
                  "max_swap_growth_bytes": int(args.max_swap_growth_mb * 1024**2),
                  "max_pagein_growth": args.max_pagein_growth,
                  "max_compressor_growth": args.max_compressor_growth},
        "before": before, "warmups": [], "runs": [],
    }
    error: BaseException | None = None; sampler: ResourceSampler | None = None
    with CampaignLock(campaign_root / "locks" / "full-model.lock", args.run_id):
        try:
            import mlx.core as mx
            from glm53_flash_mlx.load import load
            mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
            with ResourceSampler(physical, args.rss_sample_seconds) as sampler:
                load_start = time.perf_counter(); model, processor = load(str(model_root), lazy=True)
                data["model_load_lazy_seconds"] = time.perf_counter() - load_start
                tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
                for length in args.lengths:
                    data["warmups"].append(one_regime(model, tokenizer, length, args.prefill_step_size,
                        "exact_shape_warmup", data["gates"]["mlx_preferred_bytes"], data["gates"]["mlx_hard_bytes"]))
                    counted_pressure_before = pressure_snapshot()
                    for repetition in range(1, args.repetitions + 1):
                        row = one_regime(model, tokenizer, length, args.prefill_step_size, "counted",
                            data["gates"]["mlx_preferred_bytes"], data["gates"]["mlx_hard_bytes"])
                        row["repetition"] = repetition; data["runs"].append(row)
                    counted_pressure_after = pressure_snapshot(); delta = pressure_delta(counted_pressure_before, counted_pressure_after)
                    data.setdefault("counted_pressure", []).append({"length": length, "before": counted_pressure_before,
                        "after": counted_pressure_after, "delta": delta})
                    if delta["swap_used_bytes"] > data["gates"]["max_swap_growth_bytes"]: raise RuntimeError("swap growth gate exceeded")
                    if delta["pageins"] > args.max_pagein_growth: raise RuntimeError("page-in growth gate exceeded")
                    if delta["compressor_pages"] > args.max_compressor_growth: raise RuntimeError("compressor growth gate exceeded")
            data["resource_sampling"] = sampler.summary(); sampler = None
            if not data["resource_sampling"]["host_residency_gate_passed"]: raise RuntimeError("90% host-residency gate exceeded")
            data["after"] = system_snapshot()
            if data["after"]["listeners"] != before["listeners"]: raise RuntimeError("protected listener identity changed")
            if data["after"]["power"] != before["power"]: raise RuntimeError("power mode changed during benchmark")
            data["status"] = "complete"; data["finished_at_utc"] = utc_now()
        except BaseException as exc:
            error = exc; data["status"] = "failed"; data["error"] = f"{type(exc).__name__}: {exc}"
            data["finished_at_utc"] = utc_now(); data["after"] = system_snapshot()
            if sampler is not None: data["resource_sampling"] = sampler.summary()
        finally:
            # Lock release is controlled by the outer context and is attempted even
            # if this independent result write fails.
            try: atomic_write_json(args.output, data)
            except BaseException as write_exc:
                if error is None: error = write_exc
                else: data["result_write_error"] = f"{type(write_exc).__name__}: {write_exc}"
    if error is not None: raise error
    print(json.dumps({"result": str(args.output.resolve()), "sha256": sha256_file(args.output),
                      "status": data["status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
