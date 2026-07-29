"""Control-plane adapter for a separately labeled adaptive FML execution arm."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from typing import Any

from .adaptive_controller import (
    assess_negative_memory_candidate,
    retrieve_negative_memory,
    retrieve_policy_candidates,
    select_policy_composition,
)
from .strategy_operator import compile_policy_operator


class AdaptiveExecutionGateError(ValueError):
    """Raised before validation when an adaptive proposal violates a hard gate."""


def validate_post_execution_policy_evaluations(
    *,
    payload: dict[str, Any],
    selected_policy_by_role: dict[str, str],
    evidence_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Reject incomplete, mismatched, or artifact-free post-outcome credit."""
    rows = payload.get("policy_evaluations")
    if not isinstance(rows, list):
        raise AdaptiveExecutionGateError("Assessment must contain policy_evaluations")
    artifact_refs = {
        str(value)
        for artifact in evidence_artifacts
        for value in (artifact.get("path"), artifact.get("sha256"))
        if value
    }
    by_role: dict[str, dict[str, Any]] = {}
    allowed_status = {"PASSED", "FAILED", "NOT_APPLIED", "INCONCLUSIVE"}
    for row in rows:
        if not isinstance(row, dict):
            raise AdaptiveExecutionGateError("Every policy evaluation must be an object")
        role = row.get("role")
        if role not in selected_policy_by_role or role in by_role:
            raise AdaptiveExecutionGateError(f"Unexpected or duplicate policy role: {role}")
        if row.get("policy_id") != selected_policy_by_role[role]:
            raise AdaptiveExecutionGateError(f"Policy mismatch for role: {role}")
        if not isinstance(row.get("applied"), bool) or row.get("status") not in allowed_status:
            raise AdaptiveExecutionGateError(f"Invalid applied/status fields for role: {role}")
        refs = row.get("evidence_artifact_refs") or []
        if row.get("applied") and (not refs or any(str(ref) not in artifact_refs for ref in refs)):
            raise AdaptiveExecutionGateError(f"Applied policy lacks a real supplied artifact: {role}")
        if row.get("applied") and not row.get("gate_checks"):
            raise AdaptiveExecutionGateError(f"Applied policy lacks gate checks: {role}")
        signal = row.get("learning_signal") or {"intent": "NONE"}
        intent = signal.get("intent")
        if intent not in {"CREATE", "PATCH", "NONE"}:
            raise AdaptiveExecutionGateError(f"Invalid learning intent for role: {role}")
        if intent in {"CREATE", "PATCH"}:
            required = {
                "branch", "reusable_pattern", "trigger", "procedure", "eligible_stages",
                "expected_effect", "acceptance_gates", "rollback_condition",
            }
            missing = sorted(required - set(signal))
            if missing:
                raise AdaptiveExecutionGateError(f"Incomplete {intent} signal for {role}: {missing}")
        normalized = dict(row)
        normalized["learning_signal"] = signal
        by_role[str(role)] = normalized
    missing_roles = sorted(set(selected_policy_by_role) - set(by_role))
    if missing_roles:
        raise AdaptiveExecutionGateError(f"Assessment omitted selected roles: {missing_roles}")
    return [by_role[role] for role in selected_policy_by_role]


def _function_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _called_names(source: str) -> set[str]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names


def audit_modified_code_reachability(
    before: dict[str, str],
    after: dict[str, str],
) -> dict[str, Any]:
    """Catch the known failure mode: a new implementation is never called.

    This is deliberately a conservative static gate. It proves a new function
    has at least one call site in the allowed target-file set; it does not claim
    full dynamic call-path coverage. A later runtime marker remains required for
    confirmatory attribution.
    """
    changed = sorted(path for path in after if before.get(path) != after.get(path))
    syntax_errors: list[str] = []
    for path in changed:
        if Path(path).suffix.lower() != ".py":
            continue
        try:
            ast.parse(after[path])
        except SyntaxError as exc:
            syntax_errors.append(f"{path}:{exc.lineno}:{exc.msg}")
    previous_functions = set().union(*(_function_names(text) for text in before.values())) if before else set()
    current_functions = set().union(*(_function_names(text) for text in after.values())) if after else set()
    new_functions = sorted(current_functions - previous_functions)
    calls = set().union(*(_called_names(text) for text in after.values())) if after else set()
    uncalled_new_functions = sorted(name for name in new_functions if name not in calls)
    errors = []
    if not changed:
        errors.append("NO_TARGET_FILE_CHANGED")
    if syntax_errors:
        errors.append("PYTHON_SYNTAX_ERROR")
    if uncalled_new_functions:
        errors.append("NEW_FUNCTION_HAS_NO_TARGET_FILE_CALL_SITE")
    return {
        "schema_version": "adaptive-code-reachability-audit-v1",
        "passed": not errors,
        "claim_boundary": "Static target-file call-site evidence only; confirmatory attribution still requires a runtime marker.",
        "changed_files": changed,
        "new_functions": new_functions,
        "called_names": sorted(calls),
        "uncalled_new_functions": uncalled_new_functions,
        "syntax_errors": syntax_errors,
        "errors": errors,
    }


def enforce_resource_budget(
    *,
    tokens_consumed: int,
    wall_clock_seconds: float,
    candidate_steps_consumed: int,
    budget: dict[str, int | float],
) -> dict[str, Any]:
    checks = {
        "token_budget": not budget.get("token_budget") or tokens_consumed < int(budget["token_budget"]),
        "wall_clock_budget": not budget.get("wall_clock_seconds") or wall_clock_seconds < float(budget["wall_clock_seconds"]),
        "candidate_step_budget": candidate_steps_consumed < int(budget.get("candidate_steps", 1)),
    }
    return {
        "schema_version": "adaptive-resource-budget-gate-v1",
        "passed": all(checks.values()),
        "checks": checks,
        "consumed": {
            "tokens": tokens_consumed,
            "wall_clock_seconds": wall_clock_seconds,
            "candidate_steps": candidate_steps_consumed,
        },
        "budget": budget,
        "route": "CONTINUE" if all(checks.values()) else "STOP_BUDGET_EXHAUSTED",
    }


def build_adaptive_fml_operator(
    *,
    plan: dict[str, Any],
    graph: dict[str, Any],
    run_id: str,
    node_id: str,
    visible_evidence: list[dict[str, Any]],
    budget_before: dict[str, int | float],
    excluded_compositions: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Retrieve, compose, and compile node-local policies for one FML step."""
    nodes = {row["node_id"]: row for row in plan.get("nodes", [])}
    if node_id not in nodes:
        raise AdaptiveExecutionGateError(f"Unknown plan node: {node_id}")
    node = nodes[node_id]
    required_roles = list(node.get("metadata", {}).get("policy_roles", []))
    if not required_roles:
        raise AdaptiveExecutionGateError(f"Node has no atomic policy roles: {node_id}")
    context = plan.get("research_context", {})
    task_context = " ".join(
        str(value) for value in (
            plan.get("task_id"), context.get("task_domain"), context.get("task_description"),
            plan.get("research_request", ""),
        ) if value
    )
    candidates = retrieve_policy_candidates(
        graph, task_context=task_context, stage=node["node_type"], limit=128,
    )
    negative_memory = retrieve_negative_memory(
        graph,
        task_id=str(plan.get("task_id")),
        limit=8,
    )
    allowed = node.get("metadata", {}).get("policy_candidates_by_role", {})
    for role in required_roles:
        allowed_ids = set(allowed.get(role, []))
        candidates[role] = [row for row in candidates.get(role, []) if row["policy_id"] in allowed_ids]
    decision = select_policy_composition(
        run_id=run_id,
        node_id=node_id,
        node_type=node["node_type"],
        stage=int(node.get("stage", 4)),
        snapshot_sha256=str(graph.get("content_sha256")),
        candidates_by_role=candidates,
        required_roles=required_roles,
        evidence=visible_evidence,
        budget_before=budget_before,
        excluded_compositions=excluded_compositions,
    )
    operator = compile_policy_operator(
        graph=graph,
        selected_policy_by_role=decision["selected_policy_by_role"],
        node=node,
        task_context={
            "task_id": plan.get("task_id"),
            "research_request": plan.get("research_request", ""),
            **context,
            "negative_memory": negative_memory,
            "negative_memory_rule": (
                "Do not repeat a matching failed intervention. An override requires an explicit "
                "counterfactual_distinction that identifies a material mechanism or context difference."
            ),
        },
        visible_evidence=decision["visible_evidence"],
        budget=budget_before,
    )
    payload = {
        "schema_version": "adaptive-fml-executor-contract-v1",
        "run_id": run_id,
        "task_id": plan.get("task_id"),
        "node_id": node_id,
        "decision": decision,
        "operator": operator,
        "hard_gates": {
            "protected_final_feedback": False,
            "target_file_scope": True,
            "static_reachability_before_validation": True,
            "runtime_marker_for_confirmatory_attribution": True,
            "resource_budget_before_each_call": True,
            "explicit_policy_trace_after_execution": True,
        },
        "output_label": "adaptive_pipeline",
        "baseline_alias_forbidden": True,
        "negative_memory": negative_memory,
    }
    payload["contract_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return payload


def write_executor_contract(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
