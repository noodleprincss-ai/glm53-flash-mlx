"""Strict schema validation for the authorized next-candidate contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from campaign.provenance import canonical_sha256, manifest_reference, sha256_file


class ContractError(RuntimeError):
    """Raised for an undeclared or malformed candidate experiment."""


TOP_LEVEL = {
    "allowed_files", "correctness", "declared_parent", "environment", "evidence_class",
    "experiment", "feature_contract", "hypothesis", "implementation_sha", "kind",
    "optimization_patch_present", "owner", "parent_selection_required", "result_protocol",
    "results", "runtime_branch", "runtime_worktree", "schema_version", "slug", "status",
    "stop_conditions", "task_store_modified",
}


def require_exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ContractError(f"{label} keys differ: missing={sorted(expected-set(value))}, extra={sorted(set(value)-expected)}")


def validate_next_candidate(value: dict[str, Any]) -> str:
    require_exact_keys(value, TOP_LEVEL, "candidate")
    if value["schema_version"] != 2 or value["slug"] != "chunk-sweep" or value["kind"] != "config_only":
        raise ContractError("only the schema-v2 chunk-sweep config candidate is authorized")
    if value["implementation_sha"] is not None or value["optimization_patch_present"] or value["results"]:
        raise ContractError("candidate patch/results must remain absent before the experiment")
    if value["status"] != "planned" or value["parent_selection_required"] or value["task_store_modified"]:
        raise ContractError("invalid pre-experiment status")
    parent_keys = {"configuration_sha256", "environment_manifest", "fixture_set_sha256", "mlx_branch", "mlx_sha", "mlx_tree",
                   "model_manifest", "runtime_branch", "runtime_sha", "runtime_tree", "state_id",
                   "state_manifest_canonical_sha256", "state_manifest_path", "venv_path", "wheel_payload_sha256"}
    require_exact_keys(value["declared_parent"], parent_keys, "declared_parent")
    for name in ("environment_manifest", "model_manifest"):
        ref = value["declared_parent"][name]; require_exact_keys(ref, {"canonical_sha256", "file_sha256", "path"}, name)
        observed = manifest_reference(Path(ref["path"]))
        if observed != ref:
            raise ContractError(f"{name} content identity mismatch")
    exp_keys = {"acceptance", "candidate_values", "confirmation", "control_value", "focused_screen", "memory_gates",
                "metrics", "order_protocol", "smoke", "warmup"}
    require_exact_keys(value["experiment"], exp_keys, "experiment")
    exp = value["experiment"]
    if exp["candidate_values"] != [256, 768, 1024, 1536, 2048] or exp["control_value"] != 512:
        raise ContractError("chunk values/control changed")
    if exp["order_protocol"] != ["parent_then_candidate", "candidate_then_parent"]:
        raise ContractError("both sequential process orders are mandatory")
    if exp["warmup"]["uncounted_per_exact_variant_shape"] < 1:
        raise ContractError("exact-shape warmup is mandatory")
    if exp["smoke"]["counted_repetitions_per_variant"] != 1 or exp["smoke"]["acceptance_eligible"]:
        raise ContractError("smoke must be exactly one non-acceptance repetition")
    if exp["focused_screen"]["counted_repetitions_per_variant_order"] < 3:
        raise ContractError("focused screen needs at least three repetitions")
    if exp["acceptance"]["counted_repetitions_per_variant_order"] < 5 or exp["confirmation"]["counted_repetitions_per_variant_order"] < 5:
        raise ContractError("acceptance and confirmation need at least five repetitions/order")
    if exp["metrics"]["primary_threshold"] != "geomean_prefill_only_gain_2k_8k_16k>=0.05":
        raise ContractError("primary threshold differs from architecture")
    correctness_keys = {"component_atol", "component_rtol", "fallback", "full_model_greedy_tokens", "logit_contract", "parity_command", "prompt_hashes"}
    require_exact_keys(value["correctness"], correctness_keys, "correctness")
    if value["correctness"]["component_atol"] != 0 or value["correctness"]["component_rtol"] != 0:
        raise ContractError("configuration-only chunking must be exact")
    feature_keys = {"behavior", "fallback_prefill_step_size", "feature_flag", "unsupported_or_invalid", "user_override"}
    require_exact_keys(value["feature_contract"], feature_keys, "feature_contract")
    if value["feature_contract"]["fallback_prefill_step_size"] != 512:
        raise ContractError("fallback must remain 512")
    environment_keys = {"cleared_environment", "imports", "lock_sha256", "retained_environment", "sys_path_policy", "venv_path"}
    require_exact_keys(value["environment"], environment_keys, "environment")
    result_keys = {"candidate_manifest_hash", "environment_model_hashes", "exit_status", "full_parent_tuple", "raw_run_directory",
                   "required_summaries", "test_log_hashes"}
    require_exact_keys(value["result_protocol"], result_keys, "result_protocol")
    if not value["allowed_files"] or any(not isinstance(row, str) for row in value["allowed_files"]):
        raise ContractError("allowed_files must be a non-empty string list")
    if not value["stop_conditions"]:
        raise ContractError("stop conditions are required")
    return canonical_sha256(value)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("manifest", type=Path); args = parser.parse_args()
    value = json.loads(args.manifest.read_text()); print(validate_next_candidate(value)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
