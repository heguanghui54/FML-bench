"""Atomic policy ontology and compatibility contracts for adaptive research nodes."""

from __future__ import annotations

from typing import Any, Iterable


POLICY_ATTRIBUTE_BY_ROLE = {
    "state": "state_representation",
    "proposal": "proposal_operator",
    "selection": "selection_policy",
    "debug": "debug_policy",
    "memory": "memory_policy",
    "diversity": "diversity_mechanism",
}

POLICY_ROLES_BY_NODE_TYPE = {
    "method_search": ("state", "proposal", "selection", "memory", "diversity"),
    "code_modification": ("state", "proposal", "selection", "debug", "memory", "diversity"),
    "experiment": ("state", "proposal", "selection", "debug", "memory"),
    "replication": ("state", "selection", "debug", "memory"),
    "ablation": ("state", "proposal", "selection", "memory"),
}

POLICY_ELIGIBLE_STAGES = {
    role: tuple(
        node_type
        for node_type, roles in POLICY_ROLES_BY_NODE_TYPE.items()
        if role in roles
    )
    for role in POLICY_ATTRIBUTE_BY_ROLE
}

POLICY_EXPECTED_COST = {
    "state": 0.04,
    "proposal": 0.24,
    "selection": 0.16,
    "debug": 0.14,
    "memory": 0.10,
    "diversity": 0.18,
}

STATE_CAPABILITY_BY_AGENT = {
    "adaptivesearch": "state:branch_frontier",
    "ai_scientist_v2": "state:solution_tree",
    "aide": "state:solution_tree",
    "aira_mcts": "state:mcts_tree",
    "autoresearch": "state:single_incumbent",
    "openevolve": "state:population_archive",
    "theaiscientist": "state:idea_workspaces",
}

BASE_NODE_CAPABILITIES = {
    "node_contract",
    "node_gate_signal",
    "artifact_store",
}


def policy_id(agent_id: str, role: str) -> str:
    return f"policy:{agent_id}:{role}"


def policy_roles_for_node(node_type: str) -> tuple[str, ...]:
    return POLICY_ROLES_BY_NODE_TYPE.get(node_type, ())


def _requires(agent_id: str, role: str) -> set[str]:
    state = STATE_CAPABILITY_BY_AGENT[agent_id]
    requirements = {
        "state": {"node_contract"},
        "proposal": {"candidate_state", "node_contract"},
        "selection": {"candidate_pool", "node_gate_signal"},
        "debug": {"candidate_artifact"},
        "memory": {"candidate_state", "artifact_store"},
        "diversity": {"candidate_pool"},
    }[role]
    coupled: dict[str, set[str]] = {
        "adaptivesearch": {"proposal", "selection", "memory", "diversity"},
        "ai_scientist_v2": {"proposal", "selection", "debug", "memory"},
        "aide": {"proposal", "selection", "debug", "memory"},
        "aira_mcts": {"proposal", "selection", "debug", "memory", "diversity"},
        "autoresearch": {"selection", "memory"},
        "openevolve": {"proposal", "selection", "memory", "diversity"},
        "theaiscientist": {"selection", "memory"},
    }
    if role in coupled.get(agent_id, set()):
        requirements.add(state)
    if role == "diversity" and agent_id in {"autoresearch", "theaiscientist"}:
        requirements.add("history")
    return requirements


def _provides(agent_id: str, role: str) -> set[str]:
    state = STATE_CAPABILITY_BY_AGENT[agent_id]
    provisions = {
        "state": {"candidate_state", state},
        "proposal": {"candidate_pool", "candidate_artifact"},
        "selection": {"selected_candidate"},
        "debug": {"debug_trace"},
        "memory": {"history"},
        "diversity": {"diversity_signal"},
    }[role]
    return provisions


def build_agent_policy_specs(agent: dict[str, Any]) -> list[dict[str, Any]]:
    """Decompose a source agent into policy nodes without claiming local success."""
    agent_id = agent["agent_id"]
    source_strategy_id = f"strategy:{agent_id}"
    specs = []
    for role, attribute in POLICY_ATTRIBUTE_BY_ROLE.items():
        specs.append(
            {
                "policy_id": policy_id(agent_id, role),
                "policy_role": role,
                "instruction": agent[attribute],
                "source_agent_id": agent_id,
                "source_strategy_id": source_strategy_id,
                "strategy_family": agent["strategy_family"],
                "eligible_stages": list(POLICY_ELIGIBLE_STAGES[role]),
                "requires_capabilities": sorted(_requires(agent_id, role)),
                "provides_capabilities": sorted(_provides(agent_id, role)),
                "conflicts_with": [],
                "expected_cost": POLICY_EXPECTED_COST[role],
                "diversity_contribution": 1.0 if role == "diversity" else 0.0,
                "credit_assignment": "explicit_policy_trace_only",
                "selectable_unit": True,
            }
        )
    return specs


def validate_policy_composition(
    policies: Iterable[dict[str, Any]],
    *,
    node_type: str,
    required_roles: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Validate stage eligibility, unique roles, capabilities, and conflicts."""
    rows = list(policies)
    expected_roles = tuple(required_roles or policy_roles_for_node(node_type))
    roles = [row.get("attributes", {}).get("policy_role") for row in rows]
    errors: list[str] = []
    if len(roles) != len(set(roles)):
        errors.append("duplicate policy role")
    missing_roles = sorted(set(expected_roles) - set(roles))
    unexpected_roles = sorted(set(roles) - set(expected_roles))
    if missing_roles:
        errors.append(f"missing policy roles: {missing_roles}")
    if unexpected_roles:
        errors.append(f"unexpected policy roles: {unexpected_roles}")
    selected_ids = {row.get("node_id") or row.get("policy_id") for row in rows}
    provided = set(BASE_NODE_CAPABILITIES)
    required: set[str] = set()
    for row in rows:
        attributes = row.get("attributes", row)
        eligible = set(attributes.get("eligible_stages", []))
        if eligible and node_type not in eligible:
            errors.append(f"policy is not eligible for {node_type}: {row.get('node_id') or row.get('policy_id')}")
        provided.update(attributes.get("provides_capabilities", []))
        required.update(attributes.get("requires_capabilities", []))
        conflicts = selected_ids & set(attributes.get("conflicts_with", []))
        if conflicts:
            errors.append(f"policy conflict: {sorted(conflicts)}")
    missing_capabilities = sorted(required - provided)
    if missing_capabilities:
        errors.append(f"missing capabilities: {missing_capabilities}")
    return {
        "passed": not errors,
        "errors": errors,
        "node_type": node_type,
        "required_roles": list(expected_roles),
        "provided_capabilities": sorted(provided),
        "required_capabilities": sorted(required),
    }
