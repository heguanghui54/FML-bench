"""Evidence-preserving post-hoc policy reassessment for adaptive FML runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from agents.adaptive_pipeline.agent import AdaptivePipelineAgent
from agents.llm import create_client, get_response_from_llm

from .adaptive_fml_executor import validate_post_execution_policy_evaluations
from .strategy_operator import compile_policy_assessment_operator


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _descriptor(path: Path, artifact_type: str, content_summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path),
        "artifact_type": artifact_type,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
        "content_summary": content_summary,
    }


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    value = json.loads(stripped)
    if not isinstance(value, dict):
        raise ValueError("Reviewer response must be a JSON object")
    return value


def build_reassessment_evidence(summary_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    summary = _read_json(summary_path)
    run_dir = summary_path.parent
    adaptive = (summary.get("metadata") or {}).get("adaptive_pipeline") or {}
    contract = (adaptive.get("decision_contracts") or [])[-1]
    audit = (adaptive.get("reachability_audits") or [])[-1]
    step = (summary.get("val_steps") or [])[-1]
    val_result = step.get("val_result") or {}

    contract_path = run_dir / "adaptive_contract_step_0001.json"
    audit_path = run_dir / "reachability_step_0001.json"
    snapshot_path = run_dir / "step_snapshots" / "step_0001_code.json"
    artifacts = [
        _descriptor(contract_path, "decision_contract", {
            "contract_sha256": contract.get("contract_sha256"),
            "selected_policy_by_role": (contract.get("decision") or {}).get("selected_policy_by_role"),
            "planner_action": (contract.get("planner_output") or {}).get("action"),
            "planner_rationale": (contract.get("planner_output") or {}).get("rationale"),
            "planned_policy_use": (contract.get("planner_output") or {}).get("planned_policy_use"),
        }),
        _descriptor(audit_path, "reachability_audit", {
            key: audit.get(key)
            for key in ("passed", "changed_files", "new_functions", "uncalled_new_functions", "syntax_errors", "errors")
        }),
        _descriptor(snapshot_path, "code_snapshot", {
            "snapshot_file_paths": sorted(_read_json(snapshot_path)),
            "changed_files": audit.get("changed_files"),
            "reachability_passed": audit.get("passed"),
        }),
    ]
    if val_result.get("success") and val_result.get("primary_metric") is not None:
        candidates = [
            path for path in run_dir.rglob("val_info.json")
            if "final_test" not in str(path)
        ]
        if not candidates:
            raise ValueError("Successful adaptive summary has no validation artifact")
        val_path = min(candidates, key=lambda path: path.stat().st_mtime_ns)
        artifacts.append(_descriptor(val_path, "validation_result", {
            "success": True,
            "primary_metric": val_result.get("primary_metric"),
            "filtered_results": val_result.get("filtered_results"),
        }))
        outcome = "VALID_CANDIDATE"
    else:
        failure = AdaptivePipelineAgent._classify_validation_failure(val_result.get("error"))
        failure_path = run_dir / "posthoc_validation_failure.json"
        failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        artifacts.append(_descriptor(failure_path, "validation_failure", failure))
        outcome = failure["kind"]
    return summary, artifacts, outcome


def reassess(
    *,
    summary_path: Path,
    graph_path: Path,
    plan_root: Path,
    out_path: Path,
    model: str,
    provider: str,
) -> dict[str, Any]:
    summary, artifacts, outcome = build_reassessment_evidence(summary_path)
    graph = _read_json(graph_path)
    task_id = str(summary.get("benchmark"))
    plan = _read_json(plan_root / f"{task_id}.json")
    adaptive = summary["metadata"]["adaptive_pipeline"]
    contract = adaptive["decision_contracts"][-1]
    decision = contract["decision"]
    audit = adaptive["reachability_audits"][-1]
    step = summary["val_steps"][-1]
    val_result = step.get("val_result") or {}
    node = next(row for row in plan["nodes"] if row["node_id"] == "hypothesis-frontier")
    assessment = compile_policy_assessment_operator(
        graph=graph,
        selected_policy_by_role=decision["selected_policy_by_role"],
        node=node,
        planned_policy_use=(contract.get("planner_output") or {}).get("planned_policy_use") or [],
        outcome=outcome,
        evidence_artifacts=artifacts,
        state_before={"validation_metric": summary.get("baseline_primary_metric")},
        next_state={
            "validation_success": bool(val_result.get("success")),
            "validation_metric": val_result.get("primary_metric"),
            "reachability_gate_passed": audit.get("passed"),
        },
        resource_telemetry={
            "tokens_consumed": (summary.get("token_usage") or {}).get("total_tokens"),
            "wall_clock_seconds": summary.get("total_duration_seconds"),
        },
    )
    client, resolved_model = create_client(model, provider)
    response, _, usage = get_response_from_llm(
        assessment["instruction"],
        client=client,
        model=resolved_model,
        system_message=(
            "You are a post-execution ML research auditor. Return only JSON with policy_evaluations. "
            "The supplied content_summary is a compact extraction of the referenced immutable hashed artifact. "
            "Separate planned, executed, and effect credit. Mark a role FAILED when it was applied but its "
            "role-specific gate or predicted effect was contradicted. Mark NOT_APPLIED when its planned artifact "
            "or procedure was never executed. Use learning intent NONE for task-specific, duplicate, inconclusive, "
            "or single-run rules; never use a protected-test metric for credit or learning."
        ),
    )
    reviewed = _parse_json_object(response)
    assessment["policy_evaluations"] = validate_post_execution_policy_evaluations(
        payload=reviewed,
        selected_policy_by_role=decision["selected_policy_by_role"],
        evidence_artifacts=artifacts,
    )
    assessment["review_status"] = "VALIDATED_POSTHOC_ARTIFACT_REASSESSMENT"
    assessment["evidence_artifacts"] = artifacts
    payload = {
        "schema_version": "adaptive-policy-posthoc-reassessment-v1",
        "source_summary_path": str(summary_path),
        "source_summary_sha256": _sha256(summary_path),
        "protected_final_evidence_included": False,
        "usage": usage,
        "assessment": assessment,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--plan-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--provider", default="CodexCLI")
    args = parser.parse_args()
    payload = reassess(
        summary_path=args.summary, graph_path=args.graph, plan_root=args.plan_root,
        out_path=args.out, model=args.model, provider=args.provider,
    )
    print(json.dumps({
        "out": str(args.out),
        "outcome": payload["assessment"]["outcome"],
        "review_status": payload["assessment"]["review_status"],
    }, indent=2))


if __name__ == "__main__":
    main()
