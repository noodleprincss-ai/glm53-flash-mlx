"""H0 TTFT, pure-prefill, and lookahead-aware decode measurement."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from campaign.instrumented_prefill import instrumented_prefill
from campaign.provenance import verify_checkpoint_stats, verify_runtime_manifest


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()


def listeners() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for port in (8000, 8001, 8062):
        proc = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-Fpc"],
            text=True,
            capture_output=True,
        )
        rows: list[dict[str, Any]] = []
        current: dict[str, Any] = {}
        for line in proc.stdout.splitlines():
            if line.startswith("p"):
                if current:
                    rows.append(current)
                current = {"pid": int(line[1:])}
            elif line.startswith("c"):
                current["command"] = line[1:]
        if current:
            rows.append(current)
        result[str(port)] = rows
    return result


def system_snapshot() -> dict[str, Any]:
    ps = command("ps", "-axo", "pid=,rss=,comm=")
    processes = []
    for line in ps.splitlines():
        fields = line.strip().split(None, 2)
        if len(fields) == 3:
            processes.append(
                {"pid": int(fields[0]), "rss_bytes": int(fields[1]) * 1024, "command": fields[2]}
            )
    others = [row for row in processes if row["pid"] != os.getpid()]
    return {
        "listeners": listeners(),
        "self_rss_bytes": next((row["rss_bytes"] for row in processes if row["pid"] == os.getpid()), 0),
        "other_rss_bytes": sum(row["rss_bytes"] for row in others),
        "largest_other_processes": sorted(others, key=lambda row: row["rss_bytes"], reverse=True)[:12],
        "swapusage": command("sysctl", "vm.swapusage"),
        "vm_stat": command("vm_stat"),
    }


def topk(logprobs: Any, k: int = 10) -> dict[str, Any]:
    import mlx.core as mx
    indices = mx.argpartition(-logprobs, kth=k - 1)[:k]
    values = logprobs[indices]
    mx.eval(indices, values)
    pairs = sorted(zip(indices.tolist(), values.tolist()), key=lambda pair: pair[1], reverse=True)
    return {"indices": [int(pair[0]) for pair in pairs], "logprobs": [float(pair[1]) for pair in pairs]}


def prompt_ids(tokenizer: Any, length: int) -> tuple[Any, list[int], str]:
    import mlx.core as mx
    seed = (
        "Benchmarking long-context language model inference requires careful separation "
        "of prompt processing from autoregressive decoding. This deterministic prose "
        "supplies representative text tokens. "
    )
    base = tokenizer.encode(seed, add_special_tokens=False)
    ids = (base * ((length + len(base) - 1) // len(base)))[:length]
    digest = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    return mx.array([ids]), ids, digest


def public_metrics(model: Any, input_ids: Any, prefill_step_size: int, timed_decode: int = 128) -> dict[str, Any]:
    import mlx.core as mx
    from mlx_vlm.generate import generate_step

    discarded_yields = 8
    max_tokens = discarded_yields + timed_decode + 1
    generator = generate_step(
        input_ids,
        model,
        None,
        None,
        max_tokens=max_tokens,
        temperature=0.0,
        prefill_step_size=prefill_step_size,
    )
    ttft_start = time.perf_counter()
    first_token, first_logprobs = next(generator)
    ttft_stop = time.perf_counter()
    discarded = [int(first_token)]
    for _ in range(discarded_yields - 1):
        token, _ = next(generator)
        discarded.append(int(token))

    # Drain the already-issued token-8 lookahead before starting.
    mx.synchronize()
    decode_start = time.perf_counter()
    timed_tokens: list[int] = []
    last_logprobs = None
    for _ in range(timed_decode):
        token, last_logprobs = next(generator)
        timed_tokens.append(int(token))
    # Include the step that generated lookahead token 136 before stopping.
    mx.synchronize()
    decode_stop = time.perf_counter()

    # Retrieve, record, and exclude the already-computed lookahead token.  The
    # public iterator queues one further step before yielding it; that happens
    # after the timed boundary and is synchronized only for clean shutdown.
    lookahead_token, _ = next(generator)
    generator.close()
    mx.synchronize()
    decode_seconds = decode_stop - decode_start
    return {
        "ttft_seconds": ttft_stop - ttft_start,
        "ttft_derived_prompt_rate": int(input_ids.size) / (ttft_stop - ttft_start),
        "first_token_id": int(first_token),
        "first_token_topk": topk(first_logprobs),
        "discarded_yield_indices": [0, discarded_yields - 1],
        "discarded_token_ids": discarded,
        "timed_yield_indices": [discarded_yields, discarded_yields + timed_decode - 1],
        "timed_model_step_indices": [discarded_yields + 1, discarded_yields + timed_decode],
        "timed_token_ids": timed_tokens,
        "timed_decode_steps": timed_decode,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": timed_decode / decode_seconds,
        "decode_ms_per_token": 1000.0 * decode_seconds / timed_decode,
        "lookahead_token_index": discarded_yields + timed_decode,
        "lookahead_token_id_excluded": int(lookahead_token),
        "final_timed_topk": topk(last_logprobs),
        "attached_context_tokens": int(input_ids.size) + discarded_yields,
    }


def one_regime(model: Any, tokenizer: Any, length: int, prefill_step_size: int, phase: str) -> dict[str, Any]:
    import mlx.core as mx

    input_ids, ids, prompt_sha = prompt_ids(tokenizer, length)
    mx.clear_cache()
    gc.collect()
    mx.reset_peak_memory()
    public = public_metrics(model, input_ids, prefill_step_size)
    public_peak = mx.get_peak_memory()

    mx.clear_cache()
    gc.collect()
    mx.reset_peak_memory()
    pure = instrumented_prefill(
        input_ids,
        model,
        None,
        None,
        temperature=0.0,
        prefill_step_size=prefill_step_size,
        continuation_tokens=1,
    )
    pure_peak = mx.get_peak_memory()
    if pure.first_token_id != public["first_token_id"]:
        raise RuntimeError("instrumented/public first token mismatch")
    return {
        "phase": phase,
        "requested_prompt_tokens": length,
        "effective_prompt_tokens": int(input_ids.size),
        "prompt_token_ids": ids,
        "prompt_token_sha256": prompt_sha,
        "public": public,
        "pure_prefill": {
            "seconds": pure.prefill_seconds,
            "tokens_per_second": pure.prompt_rate_tokens_per_second,
            "first_token_id": pure.first_token_id,
            "final_prompt_topk": topk(pure.final_prompt_logprobs),
            "source_path": pure.source_path,
            "source_sha256": pure.source_sha256,
            "chunk_timings": [row.__dict__ for row in pure.chunk_timings],
        },
        "memory": {
            "public_peak_bytes": public_peak,
            "pure_prefill_peak_bytes": pure_peak,
            "active_bytes": mx.get_active_memory(),
            "cache_bytes": mx.get_cache_memory(),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument("--wired-limit-gb", type=float, default=220.0)
    parser.add_argument("--expected-peak-gb", type=float, default=190.0)
    args = parser.parse_args()

    campaign_root = args.campaign_root.resolve(strict=True)
    runtime = args.runtime.resolve(strict=True)
    model_root = args.model_root.resolve(strict=True)
    provenance = json.loads(args.provenance.read_text())
    verify_checkpoint_stats(provenance)
    verify_runtime_manifest(provenance, runtime)

    before = system_snapshot()
    if before["listeners"]["8001"]:
        raise RuntimeError("port 8001 must remain disabled")
    physical = provenance["host"]["physical_memory_bytes"]
    projected = int(args.expected_peak_gb * 1e9) + before["other_rss_bytes"]
    if projected > int(physical * 0.90):
        raise RuntimeError(
            f"projected residency {projected} exceeds 90% ceiling {int(physical * 0.90)}"
        )

    lock = campaign_root / "locks" / "full-model.lock"
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise RuntimeError(f"campaign lock already held: {lock}") from exc

    data: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "runtime": provenance["runtime"],
        "model_root": str(model_root),
        "provenance_manifest": str(args.provenance.resolve()),
        "prefill_step_size": args.prefill_step_size,
        "before": before,
        "warmups": [],
        "runs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        import mlx.core as mx
        from glm53_flash_mlx.load import load

        mx.set_wired_limit(int(args.wired_limit_gb * 1e9))
        load_start = time.perf_counter()
        model, processor = load(str(model_root), lazy=True)
        data["model_load_lazy_seconds"] = time.perf_counter() - load_start
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        for length in args.lengths:
            data["warmups"].append(
                one_regime(model, tokenizer, length, args.prefill_step_size, "exact_shape_warmup")
            )
            for repetition in range(1, args.repetitions + 1):
                row = one_regime(model, tokenizer, length, args.prefill_step_size, "counted_h0_control")
                row["repetition"] = repetition
                data["runs"].append(row)
        data["status"] = "complete"
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["after"] = system_snapshot()
        if data["after"]["listeners"] != before["listeners"]:
            raise RuntimeError("protected listener identity changed during benchmark")
        if data["after"]["listeners"]["8001"]:
            raise RuntimeError("port 8001 became enabled")
    except Exception as exc:
        data["status"] = "failed"
        data["error"] = f"{type(exc).__name__}: {exc}"
        data["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        data["after"] = system_snapshot()
        raise
    finally:
        args.output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
        lock.rmdir()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
