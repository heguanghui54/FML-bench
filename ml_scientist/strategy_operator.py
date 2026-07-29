"""Compile frozen strategy or atomic-policy selections into Codex contracts."""

from __future__ import annotations

import json
from typing import Any

from .policy_model import validate_policy_composition


NODE_OBJECTIVES = {
    "literature_search": "Retrieve source-verifiable primary literature; never invent citations or paper contents.",
    "method_search": "Generate and compare falsifiable ML hypotheses; do not modify code.",
    "baseline_reproduction": "Reproduce the untouched baseline and diagnose any contract mismatch.",
    "code_modification": "Propose one reachable, minimal code change within the allowed files.",
    "experiment_design": "Freeze candidates, controls, activation targets, metrics, seeds, reproducibility, and resource budgets before execution.",
    "experiment": "Run or specify the frozen validation experiment without protected-test feedback.",
    "replication": "Replicate the selected candidate under the same protocol and independent seed.",
    "ablation": "Remove or vary one mechanism while holding the remaining protocol fixed.",
    "statistical_analysis": "Analyze complete matched units with uncertainty and multiplicity control.",
    "paper_outline": "Organize the claim contract without introducing result claims.",
    "paper_methods": "Describe only source-grounded methods and frozen protocol details.",
    "paper_writing": "Write claims that resolve to the current evidence bundle and hashes.",
    "paper_integrity": "Audit every claim, number, citation, and evidence version.",
    "paper_peer_review": "Adversarially review methodology, statistics, reproducibility, and overclaims.",
    "governance": "Validate and freeze the requested contract without changing experimental state.",
    "contract": "Validate and freeze the requested contract without changing experimental state.",
}


def compile_stage_operator(
    *,
    graph: dict[str, Any],
    strategy_id: str,
    node: dict[str, Any],
    task_context: dict[str, Any],
    visible_evidence: list[dict[str, Any]],
    budget: dict[str, float | int],
) -> dict[str, Any]:
    if any(row.get("observability") == "PROTECTED_FINAL_ONLY" for row in visible_evidence):
        raise ValueError("Protected-final evidence cannot enter an operator prompt")
    graph_nodes = {row["node_id"]: row for row in graph.get("nodes", [])}
    strategy = graph_nodes.get(strategy_id)
    if strategy is None:
        raise ValueError(f"Strategy is missing from frozen graph: {strategy_id}")
    attributes = strategy.get("attributes", {})
    node_type = node["node_type"]
    objective = NODE_OBJECTIVES.get(node_type, f"Complete the {node_type} node under its frozen contract.")
    policy = {
        "strategy_family": attributes.get("strategy_family", strategy["label"]),
        "proposal_operator": attributes.get("proposal_operator", "produce a bounded candidate"),
        "selection_policy": attributes.get("selection_policy", "apply the node hard gates"),
        "debug_policy": attributes.get("debug_policy", "return a typed failure instead of inventing success"),
        "memory_policy": attributes.get("memory_policy", "use only the supplied frozen context"),
        "diversity_mechanism": attributes.get("diversity_mechanism", "none unless required by the node"),
    }
    instruction = "\n\n".join(
        [
            f"## Node\n{node['node_id']} ({node_type})\n{objective}",
            "## Selected strategy ideology\n" + "\n".join(f"- {key}: {value}" for key, value in policy.items()),
            "## Task context\n" + json.dumps(task_context, indent=2, ensure_ascii=False, sort_keys=True),
            "## Visible evidence\n" + json.dumps(visible_evidence, indent=2, ensure_ascii=False, sort_keys=True),
            "## Budget\n" + json.dumps(budget, indent=2, ensure_ascii=False, sort_keys=True),
            "## Hard gates\n" + "\n".join(f"- {gate}" for gate in node["inner_loop"]["hard_gates"]),
            "## Required output\nReturn a JSON object with keys: action, rationale, proposed_artifacts, expected_metric_effect, risks, gate_checks, and stop_condition. Do not claim execution or results that are not present in the visible evidence.",
        ]
    )
    return {
        "schema_version": "ml-scientist-stage-operator-v1",
        "node_id": node["node_id"],
        "node_type": node_type,
        "strategy_id": strategy_id,
        "strategy_maturity": strategy.get("maturity", "observation"),
        "strategy_policy": policy,
        "visible_evidence_count": len(visible_evidence),
        "budget": budget,
        "protected_evidence_included": False,
        "instruction": instruction,
        "codex_cli_mode": "text_only_for_planning_or_CodeEditor_for_declared_target_files",
        "execution_requires_explicit_executor_call": True,
    }


def compile_policy_operator(
    *,
    graph: dict[str, Any],
    selected_policy_by_role: dict[str, str],
    node: dict[str, Any],
    task_context: dict[str, Any],
    visible_evidence: list[dict[str, Any]],
    budget: dict[str, float | int],
) -> dict[str, Any]:
    if any(row.get("observability") == "PROTECTED_FINAL_ONLY" for row in visible_evidence):
        raise ValueError("Protected-final evidence cannot enter an operator prompt")
    graph_nodes = {row["node_id"]: row for row in graph.get("nodes", [])}
    selected = []
    for role, policy_id in selected_policy_by_role.items():
        policy = graph_nodes.get(policy_id)
        if policy is None or policy.get("node_type") not in {"Policy", "Skill"}:
            raise ValueError(f"Atomic policy is missing from frozen graph: {policy_id}")
        attributes = policy.get("attributes", {})
        if attributes.get("policy_role") != role:
            raise ValueError(f"Policy role mismatch for {policy_id}: expected {role}")
        if policy.get("node_type") == "Skill" and not attributes.get("active"):
            raise ValueError(f"Inactive skill cannot enter a policy composition: {policy_id}")
        selected.append(policy)
    node_type = node["node_type"]
    required_roles = list(node.get("metadata", {}).get("policy_roles", selected_policy_by_role))
    compatibility = validate_policy_composition(
        selected,
        node_type=node_type,
        required_roles=required_roles,
    )
    if not compatibility["passed"]:
        raise ValueError("Invalid policy composition: " + "; ".join(compatibility["errors"]))
    objective = NODE_OBJECTIVES.get(node_type, f"Complete the {node_type} node under its frozen contract.")
    policies = {
        role: {
            "policy_id": selected_policy_by_role[role],
            "instruction": graph_nodes[selected_policy_by_role[role]]["attributes"].get("instruction"),
            "source_strategy_id": graph_nodes[selected_policy_by_role[role]]["attributes"].get("source_strategy_id"),
            "requires_capabilities": graph_nodes[selected_policy_by_role[role]]["attributes"].get("requires_capabilities", []),
            "provides_capabilities": graph_nodes[selected_policy_by_role[role]]["attributes"].get("provides_capabilities", []),
        }
        for role in required_roles
    }
    policy_lines = []
    for role in required_roles:
        row = policies[role]
        policy_lines.append(f"- {role} [{row['policy_id']}]: {row['instruction']}")
    planned_use_template = [
        {
            "policy_id": selected_policy_by_role[role],
            "role": role,
            "planned_use": "specific way this policy will influence the action",
            "planned_gate_checks": ["check that can be evaluated after execution"],
        }
        for role in required_roles
    ]
    instruction = "\n\n".join(
        [
            f"## Node\n{node['node_id']} ({node_type})\n{objective}",
            "## Explicit atomic policy composition\n" + "\n".join(policy_lines),
            "## Compatibility contract\n" + json.dumps(compatibility, indent=2, ensure_ascii=False, sort_keys=True),
            "## Task context\n" + json.dumps(task_context, indent=2, ensure_ascii=False, sort_keys=True),
            "## Visible evidence\n" + json.dumps(visible_evidence, indent=2, ensure_ascii=False, sort_keys=True),
            "## Budget\n" + json.dumps(budget, indent=2, ensure_ascii=False, sort_keys=True),
            "## Hard gates\n" + "\n".join(f"- {gate}" for gate in node["inner_loop"]["hard_gates"]),
            "## Planned policy use\nBefore execution, declare how each selected policy is intended to affect the action and which later artifact can verify that use. This is not credit assignment and must not contain PASSED/FAILED status or CREATE/PATCH learning signals.\n"
            + json.dumps(planned_use_template, indent=2, ensure_ascii=False),
            "## Required output\nReturn a JSON object with keys: action, rationale, proposed_artifacts, expected_metric_effect, risks, gate_checks, planned_policy_use, and stop_condition. Do not claim execution, measured telemetry, policy success, or learning signals before real outcome artifacts exist.",
        ]
    )
    return {
        "schema_version": "ml-scientist-atomic-policy-operator-v2",
        "node_id": node["node_id"],
        "node_type": node_type,
        "selected_policy_by_role": selected_policy_by_role,
        "selected_policy_ids": [selected_policy_by_role[role] for role in required_roles],
        "policy_contracts": policies,
        "compatibility": compatibility,
        "policy_trace_required": True,
        "trajectory_learning_signal_required": True,
        "policy_trace_phase": "POST_EXECUTION_ONLY",
        "pre_execution_credit_assignment_allowed": False,
        "learning_intents": ["CREATE", "PATCH", "NONE"],
        "visible_evidence_count": len(visible_evidence),
        "budget": budget,
        "protected_evidence_included": False,
        "instruction": instruction,
        "codex_cli_mode": "text_only_for_planning_or_CodeEditor_for_declared_target_files",
        "execution_requires_explicit_executor_call": True,
    }


def compile_policy_assessment_operator(
    *,
    graph: dict[str, Any],
    selected_policy_by_role: dict[str, str],
    node: dict[str, Any],
    planned_policy_use: list[dict[str, Any]],
    outcome: str,
    evidence_artifacts: list[dict[str, Any]],
    state_before: dict[str, Any],
    next_state: dict[str, Any],
    resource_telemetry: dict[str, Any],
) -> dict[str, Any]:
    """Compile the post-execution credit and skill-learning assessment."""
    if any(row.get("observability") == "PROTECTED_FINAL_ONLY" for row in next_state.get("visible_evidence", [])):
        raise ValueError("Protected-final evidence cannot enter search-skill assessment")
    graph_nodes = {row["node_id"]: row for row in graph.get("nodes", [])}
    for role, policy_id in selected_policy_by_role.items():
        policy = graph_nodes.get(policy_id)
        if policy is None or policy.get("attributes", {}).get("policy_role") != role:
            raise ValueError(f"Invalid selected policy for assessment: {role}={policy_id}")
    trace_template = [
        {
            "policy_id": policy_id,
            "role": role,
            "applied": "boolean supported by planned use and artifact",
            "status": "PASSED|FAILED|NOT_APPLIED|INCONCLUSIVE",
            "evidence_artifact_refs": ["existing artifact path or hash"],
            "gate_checks": ["observed role-specific check"],
            "learning_signal": {
                "intent": "CREATE|PATCH|NONE",
                "branch": "general|task_specific|action",
                "reusable_pattern": "required only for CREATE or PATCH",
                "trigger": {"measurable state condition": "observed value"},
                "procedure": ["ordered reusable step"],
                "eligible_stages": [node["node_type"]],
                "expected_effect": {"visible metric": "direction"},
                "expected_cost": "measured or bounded",
                "acceptance_gates": ["future held-out measurable gate"],
                "rollback_condition": "comparable held-out evidence contradicts the rule"
            },
        }
        for role, policy_id in selected_policy_by_role.items()
    ]
    instruction = "\n\n".join([
        "## Assessment phase\nExecution is complete. Assign credit only when the planned use is visible in a real artifact. A node-level success does not imply every selected policy passed.",
        "## Planned policy use\n" + json.dumps(planned_policy_use, indent=2, ensure_ascii=False, sort_keys=True),
        "## Outcome\n" + json.dumps({"outcome": outcome}, ensure_ascii=False),
        "## State transition\n" + json.dumps({"state_before": state_before, "next_state": next_state}, indent=2, ensure_ascii=False, sort_keys=True),
        "## Measured resources\n" + json.dumps(resource_telemetry, indent=2, ensure_ascii=False, sort_keys=True),
        "## Existing evidence artifacts\n" + json.dumps(evidence_artifacts, indent=2, ensure_ascii=False, sort_keys=True),
        "## Required trace\nReturn exactly one row per selected policy. CREATE/PATCH is allowed only for a reusable transition-level procedure with a measurable trigger and later held-out gate. Otherwise emit NONE.\n" + json.dumps(trace_template, indent=2, ensure_ascii=False),
    ])
    return {
        "schema_version": "ml-scientist-policy-assessment-operator-v1",
        "node_id": node["node_id"],
        "node_type": node["node_type"],
        "selected_policy_by_role": selected_policy_by_role,
        "outcome": outcome,
        "evidence_artifact_count": len(evidence_artifacts),
        "protected_final_evidence_included": False,
        "credit_assignment": "EXPLICIT_POST_EXECUTION_ARTIFACT_TRACE_ONLY",
        "learning_signal_phase": "POST_EXECUTION_ONLY",
        "instruction": instruction,
    }
