"""Import completed legacy FML summaries as non-crediting observation episodes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .knowledge_graph import append_episode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summary_to_observational_episode(
    summary_path: Path,
    *,
    graph_snapshot_sha256: str,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Preserve bundle-level evidence without inventing atomic policy adoption."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    task_id = str(summary.get("benchmark"))
    agent_id = str(summary.get("agent"))
    task = next((row for row in catalog.get("tasks", []) if row.get("task_id") == task_id), {})
    run_id = str(summary.get("workspace_label") or summary_path.parent.name)
    summary_hash = _sha256(summary_path)
    steps = summary.get("val_steps") or []
    validation_evidence = []
    for step in steps:
        validation_evidence.append({
            "metric_name": task.get("canonical_metric", "validation_primary_metric"),
            "observability": "ONLINE_VISIBLE",
            "split": "validation",
            "step_id": step.get("step_id"),
            "value": step.get("primary_metric"),
            "success": (step.get("val_result") or {}).get("success"),
        })
    test_result = summary.get("test_result") or {}
    test_success = test_result.get("success") is True
    best_val = summary.get("best_val_metric")
    outcome = "OBSERVATIONAL_COMPLETE" if test_success and best_val is not None else "OBSERVATIONAL_INVALID_OR_FALLBACK"
    token_usage = summary.get("token_usage") or {}
    return {
        "schema_version": "ml-scientist-adaptive-episode-v2",
        "episode_id": f"fml-observation:{run_id}:{summary_hash[:12]}",
        "run_id": run_id,
        "task_id": task_id,
        "node_id": "baseline-reproduction",
        "node_type": "experiment",
        "decision_id": f"decision:fml-observation:{summary_hash[:16]}",
        "selected_strategy_id": f"strategy:{agent_id}",
        "selected_policy_ids": [],
        "selected_policy_by_role": {},
        "visible_evidence": validation_evidence,
        "research_context": {
            "research_request": "Frozen FML pilot baseline observation",
            "task_id": task_id,
            "task_domain": task.get("domain"),
            "task_description": task.get("description"),
            "node_id": "baseline-reproduction",
            "node_type": "experiment",
        },
        "state_before": {
            "node_status": "READY",
            "baseline_primary_metric": summary.get("baseline_primary_metric"),
            "candidate_budget_steps": (summary.get("agent_params") or {}).get("max_steps"),
        },
        "action": {
            "kind": "execute_frozen_baseline_pipeline_bundle",
            "agent_id": agent_id,
            "strategy_id": f"strategy:{agent_id}",
            "atomic_policy_trace_available": False,
            "credit_assignment": "BUNDLE_LEVEL_OBSERVATION_ONLY",
        },
        "outcome": outcome,
        "next_state": {
            "node_status": "COMPLETE",
            "route": {"action": "ARCHIVE_OBSERVATION", "used_protected_metric": False},
            "observations": {
                "best_validation_metric": best_val,
                "protected_test_executed_after_search": bool(test_result),
                "protected_test_success": test_success,
                "protected_test_metric_redacted_from_routing_state": True,
            },
        },
        "budget_before": {"candidate_steps": (summary.get("agent_params") or {}).get("max_steps")},
        "budget_after": {"candidate_steps_consumed": summary.get("total_steps")},
        "resource_telemetry": {
            "tokens_consumed": token_usage.get("total_tokens"),
            "prompt_tokens": token_usage.get("prompt_tokens"),
            "completion_tokens": token_usage.get("completion_tokens"),
            "wall_clock_seconds": summary.get("total_duration_seconds"),
            "ideas_generated": summary.get("total_ideas"),
        },
        "evidence_artifacts": [{
            "path": str(summary_path.resolve()),
            "sha256": summary_hash,
            "artifact_type": "fml_summary",
            "contains_protected_final_metric": bool(test_result),
        }],
        "policy_evaluations": [],
        "learning_signals": [],
        "learning_signal_status": "MISSING_LEGACY_ATOMIC_TRACE_NOT_INFERRED",
        "skill_validation_evaluations": [],
        "snapshot_sha256": graph_snapshot_sha256,
        "protected_metric_used_for_routing": False,
        "protected_metric_used_for_skill_evolution": False,
        "maturity": "observation",
        "attribution_scope": "baseline_pipeline_bundle_only",
    }


def summary_to_adaptive_episode(
    summary_path: Path,
    *,
    graph_snapshot_sha256: str,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    """Convert the new executor's explicit post-outcome trace into an episode."""
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    adaptive = (summary.get("metadata") or {}).get("adaptive_pipeline") or {}
    contracts = adaptive.get("decision_contracts") or []
    assessments = adaptive.get("post_execution_assessments") or []
    if not contracts or not assessments:
        raise ValueError("Adaptive summary is missing decision or post-execution assessment traces")
    contract = contracts[-1]
    assessment = assessments[-1]
    posthoc_path = summary_path.parent / "posthoc_policy_assessment.json"
    if posthoc_path.exists():
        posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
        if posthoc.get("source_summary_sha256") == _sha256(summary_path):
            assessment = posthoc.get("assessment") or assessment
    decision = contract.get("decision") or {}
    task_id = str(summary.get("benchmark"))
    task = next((row for row in catalog.get("tasks", []) if row.get("task_id") == task_id), {})
    run_id = str(summary.get("workspace_label") or summary_path.parent.name)
    summary_hash = _sha256(summary_path)
    steps = summary.get("val_steps") or []
    visible_evidence = [{
        "metric_name": task.get("canonical_metric", "validation_primary_metric"),
        "observability": "ONLINE_VISIBLE",
        "split": "validation",
        "step_id": step.get("step_id"),
        "value": step.get("primary_metric"),
        "success": (step.get("val_result") or {}).get("success"),
    } for step in steps]
    policy_evaluations = assessment.get("policy_evaluations") or []
    learning_signals = [
        row["learning_signal"] for row in policy_evaluations
        if (row.get("learning_signal") or {}).get("intent") in {"CREATE", "PATCH", "NONE"}
    ]
    artifact_rows = [{
        "path": str(summary_path.resolve()), "sha256": summary_hash,
        "artifact_type": "adaptive_fml_summary", "contains_protected_final_metric": bool(summary.get("test_result")),
    }]
    for key in ("artifact_path",):
        if assessment.get(key):
            artifact_rows.append({"path": assessment[key], "artifact_type": "policy_assessment"})
    for audit in adaptive.get("reachability_audits") or []:
        if audit.get("artifact_path"):
            artifact_rows.append({"path": audit["artifact_path"], "artifact_type": "reachability_audit"})
    if posthoc_path.exists() and assessment.get("review_status") == "VALIDATED_POSTHOC_ARTIFACT_REASSESSMENT":
        artifact_rows.append({
            "path": str(posthoc_path), "sha256": _sha256(posthoc_path),
            "artifact_type": "posthoc_policy_assessment",
        })
    best_val = summary.get("best_val_metric")
    assessment_outcome = assessment.get("outcome")
    outcome = (
        "PASSED_GATE" if best_val is not None else
        "FAILED_GATE" if assessment_outcome == "CONSTRAINT_FAILED" else
        "INVALID_EXECUTION"
    )
    return {
        "schema_version": "ml-scientist-adaptive-episode-v2",
        "episode_id": f"adaptive-fml:{run_id}:{summary_hash[:12]}",
        "run_id": run_id,
        "task_id": task_id,
        "node_id": "hypothesis-frontier",
        "node_type": "method_search",
        "decision_id": decision.get("decision_id", f"decision:adaptive-fml:{summary_hash[:16]}"),
        "selected_policy_ids": list(decision.get("selected_policy_ids") or []),
        "selected_policy_by_role": decision.get("selected_policy_by_role") or {},
        "visible_evidence": visible_evidence,
        "research_context": {
            "research_request": "Adaptive FML executor arm",
            "task_id": task_id,
            "task_domain": task.get("domain"),
            "task_description": task.get("description"),
            "node_id": "hypothesis-frontier",
            "node_type": "method_search",
        },
        "state_before": {
            "node_status": "READY",
            "baseline_primary_metric": summary.get("baseline_primary_metric"),
            "budget": contract.get("operator", {}).get("budget", {}),
        },
        "action": {
            "decision_id": decision.get("decision_id"),
            "kind": "adaptive_atomic_policy_composition",
            "selected_policy_by_role": decision.get("selected_policy_by_role") or {},
            "planner_output": contract.get("planner_output") or {},
        },
        "outcome": outcome,
        "next_state": {
            "node_status": "COMPLETE" if outcome == "PASSED_GATE" else "FAILED_GATE",
            "route": {"action": "ADVANCE" if outcome == "PASSED_GATE" else "SELECT_ALTERNATIVE"},
            "observations": {
                "best_validation_metric": best_val,
                "assessment_outcome": assessment_outcome,
                "protected_test_metric_redacted_from_routing_state": True,
            },
        },
        "budget_before": contract.get("operator", {}).get("budget", {}),
        "budget_after": {"candidate_steps_consumed": summary.get("total_steps")},
        "resource_telemetry": {
            "tokens_consumed": (summary.get("token_usage") or {}).get("total_tokens"),
            "wall_clock_seconds": summary.get("total_duration_seconds"),
            "ideas_generated": summary.get("total_ideas"),
        },
        "evidence_artifacts": artifact_rows,
        "policy_evaluations": policy_evaluations,
        "learning_signals": learning_signals,
        "learning_signal_status": "EXPLICIT_POST_EXECUTION_TRACE",
        "skill_validation_evaluations": assessment.get("skill_validation_evaluations") or [],
        "snapshot_sha256": adaptive.get("graph_content_sha256") or graph_snapshot_sha256,
        "protected_metric_used_for_routing": False,
        "protected_metric_used_for_skill_evolution": False,
        "maturity": "observation",
        "attribution_scope": "explicit_atomic_policy_trace_only",
    }


def import_completed_summaries(
    *,
    results_root: Path,
    episodes_path: Path,
    graph_path: Path,
    catalog: dict[str, Any],
) -> dict[str, Any]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    snapshot = str(graph.get("content_sha256") or _sha256(graph_path))
    existing_ids: set[str] = set()
    if episodes_path.exists():
        for line in episodes_path.read_text(encoding="utf-8").splitlines():
            try:
                existing_ids.add(str(json.loads(line).get("episode_id")))
            except (json.JSONDecodeError, AttributeError):
                continue
    imported: list[str] = []
    skipped: list[str] = []
    for summary_path in sorted(results_root.rglob("summary.json")) if results_root.exists() else []:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        is_adaptive = summary.get("agent") == "adaptive_pipeline"
        episode = (
            summary_to_adaptive_episode(
                summary_path, graph_snapshot_sha256=snapshot, catalog=catalog,
            )
            if is_adaptive else
            summary_to_observational_episode(
                summary_path, graph_snapshot_sha256=snapshot, catalog=catalog,
            )
        )
        if episode["episode_id"] in existing_ids:
            skipped.append(episode["episode_id"])
            continue
        append_episode(episodes_path, episode)
        existing_ids.add(episode["episode_id"])
        imported.append(episode["episode_id"])
    return {
        "schema_version": "fml-observational-episode-import-v1",
        "results_root": str(results_root),
        "episodes_path": str(episodes_path),
        "imported_count": len(imported),
        "skipped_existing_count": len(skipped),
        "total_episode_count": len(existing_ids),
        "imported_episode_ids": imported,
        "policy_credit_inferred": False,
        "learning_signal_inferred": False,
        "next_arm_only": True,
    }
