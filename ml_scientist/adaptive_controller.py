"""Evidence-aware Stage 4-6 controller contracts and deterministic routing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from itertools import product
from typing import Any
import re

from .contracts import ControllerDecision, PolicyCompositionDecision
from .knowledge_graph import query_knowledge_graph
from .policy_model import validate_policy_composition


MATURITY_SCORE = {
    "repeated_success": 3.0,
    "provisional_success": 2.0,
    "verified": 1.5,
    "observation": 0.0,
    "superseded": -2.0,
    "contradiction": -5.0,
}


def _memory_tokens(value: Any) -> set[str]:
    text = json.dumps(value, sort_keys=True, ensure_ascii=False) if not isinstance(value, str) else value
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 2}


def _selected_action_signature(value: Any) -> Any:
    """Exclude unselected alternatives and rationale from intervention identity."""
    current = value
    for _ in range(4):
        if not isinstance(current, dict):
            break
        planner = current.get("planner_output")
        if isinstance(planner, dict) and planner.get("action") is not None:
            current = planner["action"]
            continue
        nested = current.get("action")
        if isinstance(nested, dict) and (
            "decision_id" in current or "planner_output" in current or "selected_policy_by_role" in current
        ):
            current = nested
            continue
        break
    if not isinstance(current, dict):
        return current
    identity_keys = (
        "hypothesis_id",
        "candidate_id",
        "type",
        "kind",
        "target",
        "modification",
        "minimal_modification",
        "single_testable_variant",
    )
    selected = {key: current[key] for key in identity_keys if current.get(key) is not None}
    return selected or current


def retrieve_negative_memory(
    graph: dict[str, Any],
    *,
    task_id: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Return failed prior interventions for the same task, never protected metrics."""
    nodes = {row.get("node_id"): row for row in graph.get("nodes", [])}
    failure_ids = {
        node_id for node_id, row in nodes.items() if row.get("node_type") == "Failure"
    }
    failed_episode_ids = {
        row.get("source") for row in graph.get("edges", [])
        if row.get("target") in failure_ids
    }
    results: list[dict[str, Any]] = []
    for episode_id in sorted(failed_episode_ids):
        row = nodes.get(episode_id) or {}
        attributes = row.get("attributes", {})
        episode_task = attributes.get("task_id") or (attributes.get("research_context") or {}).get("task_id")
        if episode_task != task_id:
            continue
        results.append({
            "episode_id": episode_id,
            "task_id": episode_task,
            "node_id": attributes.get("node_id"),
            "outcome": attributes.get("outcome"),
            "action": attributes.get("action"),
            "state_before": attributes.get("state_before"),
            "next_state": attributes.get("next_state"),
            "resource_telemetry": attributes.get("resource_telemetry"),
            "planner_default": "EXCLUDE_UNLESS_ARTIFACT_BACKED_DISTINCTION",
        })
    return results[:limit]


def assess_negative_memory_candidate(
    action: Any,
    negative_memories: list[dict[str, Any]],
    *,
    counterfactual_distinction: Any = None,
) -> dict[str, Any]:
    action_tokens = _memory_tokens(_selected_action_signature(action))
    matches = []
    for row in negative_memories:
        memory_tokens = _memory_tokens(_selected_action_signature(row.get("action")))
        if not action_tokens or not memory_tokens:
            continue
        containment = len(action_tokens & memory_tokens) / len(action_tokens)
        key_overlap = sorted(
            token for token in action_tokens & memory_tokens
            if "label" in token or "smooth" in token or "mixup" in token or "crossfit" in token
        )
        if containment >= 0.55 or key_overlap:
            matches.append({
                "episode_id": row["episode_id"],
                "containment": containment,
                "key_overlap": key_overlap,
            })
    distinction_text = str(counterfactual_distinction or "").strip()
    passed = not matches or bool(distinction_text)
    return {
        "schema_version": "ml-scientist-negative-memory-gate-v1",
        "passed": passed,
        "matches": matches,
        "counterfactual_distinction": distinction_text or None,
        "route": "ALLOW" if not matches else "ALLOW_JUSTIFIED_OVERRIDE" if passed else "REJECT_NEGATIVE_MEMORY_DUPLICATE",
    }


def _stable_id(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()[:20]


def filter_visible_evidence(evidence: list[dict[str, Any]], *, stage_complete: bool) -> list[dict[str, Any]]:
    visible: list[dict[str, Any]] = []
    for row in evidence:
        observability = row.get("observability")
        if observability == "PROTECTED_FINAL_ONLY":
            continue
        if observability in {"PAPER_REVIEW_ONLY", "OPERATOR_ONLY"}:
            continue
        if observability == "POST_STAGE_VISIBLE" and not stage_complete:
            continue
        visible.append(row)
    return visible


def retrieve_strategy_candidates(
    graph: dict[str, Any],
    *,
    task_context: str,
    stage: str,
    limit: int = 8,
) -> list[dict[str, Any]]:
    rows = query_knowledge_graph(graph, task_context, node_types={"Strategy", "Skill"}, stage=stage, limit=limit)
    candidates = []
    for row in rows:
        if row.get("maturity") in {"contradiction", "superseded"}:
            continue
        attributes = row.get("attributes", {})
        candidates.append(
            {
                "strategy_id": row["node_id"],
                "label": row["label"],
                "maturity": row.get("maturity", "observation"),
                "retrieval_score": row.get("score", 0.0),
                "successful_contexts": attributes.get("successful_contexts_by_stage", {}).get(stage, 0),
                "failure_count": attributes.get("failure_count_by_stage", {}).get(stage, 0),
                "expected_cost": attributes.get("expected_cost", 0.5),
                "diversity_contribution": attributes.get("diversity_contribution", 0.0),
                "contextual_utility": attributes.get("utility", {}).get("global", 0.5),
            }
        )
    return candidates


def retrieve_policy_candidates(
    graph: dict[str, Any],
    *,
    task_context: str,
    stage: str,
    limit: int = 96,
) -> dict[str, list[dict[str, Any]]]:
    rows = query_knowledge_graph(graph, task_context, node_types={"Policy", "Skill"}, stage=stage, limit=limit)
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("maturity") in {"contradiction", "superseded"}:
            continue
        attributes = row.get("attributes", {})
        if row.get("node_type") == "Skill" and not attributes.get("active"):
            continue
        role = attributes.get("policy_role")
        if not role:
            continue
        candidates.setdefault(role, []).append(
            {
                "policy_id": row["node_id"],
                "label": row["label"],
                "role": role,
                "maturity": row.get("maturity", "observation"),
                "retrieval_score": row.get("score", 0.0),
                "successful_contexts": attributes.get("successful_contexts_by_stage", {}).get(stage, 0),
                "failure_count": attributes.get("failure_count_by_stage", {}).get(stage, 0),
                "expected_cost": attributes.get("expected_cost", 0.1),
                "diversity_contribution": attributes.get("diversity_contribution", 0.0),
                "contextual_utility": attributes.get("utility", {}).get("by_context", {}).get(
                    f"{task_context}::{stage}", attributes.get("utility", {}).get("global", 0.5)
                ),
                "attributes": attributes,
            }
        )
    return candidates


def rank_strategy_candidates(
    candidates: list[dict[str, Any]],
    *,
    remaining_budget_fraction: float,
    diversity_is_low: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        cost = float(candidate.get("expected_cost", 0.5))
        if remaining_budget_fraction < 0.25 and cost > remaining_budget_fraction:
            continue
        score_components = {
            "retrieval": float(candidate.get("retrieval_score", 0.0)),
            "maturity": MATURITY_SCORE.get(candidate.get("maturity", "observation"), 0.0),
            "successful_contexts": min(float(candidate.get("successful_contexts", 0)), 3.0) * 0.5,
            "failure_penalty": -min(float(candidate.get("failure_count", 0)), 4.0) * 0.75,
            "diversity_bonus": float(candidate.get("diversity_contribution", 0.0)) if diversity_is_low else 0.0,
            "contextual_utility": (float(candidate.get("contextual_utility", 0.5)) - 0.5) * 2.0,
            "cost_penalty": -cost * (1.5 if remaining_budget_fraction < 0.5 else 0.5),
        }
        ranked.append({**candidate, "score_components": score_components, "controller_score": sum(score_components.values())})
    ranked.sort(key=lambda row: (-row["controller_score"], row["strategy_id"]))
    return ranked


def rank_policy_candidates(
    candidates: list[dict[str, Any]],
    *,
    remaining_budget_fraction: float,
    diversity_is_low: bool,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate in candidates:
        cost = float(candidate.get("expected_cost", 0.1))
        if remaining_budget_fraction < 0.25 and cost > remaining_budget_fraction:
            continue
        score_components = {
            "retrieval": float(candidate.get("retrieval_score", 0.0)),
            "maturity": MATURITY_SCORE.get(candidate.get("maturity", "observation"), 0.0),
            "successful_contexts": min(float(candidate.get("successful_contexts", 0)), 3.0) * 0.5,
            "failure_penalty": -min(float(candidate.get("failure_count", 0)), 4.0) * 0.75,
            "diversity_bonus": float(candidate.get("diversity_contribution", 0.0)) if diversity_is_low else 0.0,
            "contextual_utility": (float(candidate.get("contextual_utility", 0.5)) - 0.5) * 2.0,
            "cost_penalty": -cost * (1.5 if remaining_budget_fraction < 0.5 else 0.5),
        }
        ranked.append({**candidate, "score_components": score_components, "controller_score": sum(score_components.values())})
    ranked.sort(key=lambda row: (-row["controller_score"], row["policy_id"]))
    return ranked


def select_policy_composition(
    *,
    run_id: str,
    node_id: str,
    node_type: str,
    stage: int,
    snapshot_sha256: str,
    candidates_by_role: dict[str, list[dict[str, Any]]],
    required_roles: list[str],
    evidence: list[dict[str, Any]],
    budget_before: dict[str, float | int],
    stage_complete: bool = False,
    excluded_compositions: set[tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    """Select one compatible atomic policy per role and preserve role-level credit."""
    visible = filter_visible_evidence(evidence, stage_complete=stage_complete)
    remaining = float(budget_before.get("remaining_fraction", 1.0))
    diversity_is_low = any(
        row.get("metric_name") in {"Exploration Spread", "Exploration Uniqueness", "Exploration Reach"}
        and row.get("status") == "low"
        for row in visible
    )
    ranked_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in required_roles:
        ranked = rank_policy_candidates(
            candidates_by_role.get(role, []),
            remaining_budget_fraction=remaining,
            diversity_is_low=diversity_is_low,
        )
        if not ranked:
            raise ValueError(f"No eligible policy remains for required role: {role}")
        ranked_by_role[role] = ranked[:8]

    excluded = excluded_compositions or set()
    compatible: list[dict[str, Any]] = []
    for combination in product(*(ranked_by_role[role] for role in required_roles)):
        composition_key = tuple(row["policy_id"] for row in combination)
        if composition_key in excluded:
            continue
        graph_rows = [
            {"node_id": row["policy_id"], "attributes": row.get("attributes", {})}
            for row in combination
        ]
        compatibility = validate_policy_composition(
            graph_rows,
            node_type=node_type,
            required_roles=required_roles,
        )
        if not compatibility["passed"]:
            continue
        source_strategies = {
            row.get("attributes", {}).get("source_strategy_id")
            for row in combination
            if row.get("attributes", {}).get("source_strategy_id")
        }
        # A source bundle is provenance, not the selection unit. Reward its
        # known compatibility only after every selected role has successful
        # evidence in the current stage; otherwise a cold-start tie silently
        # reconstructs an entire baseline pipeline and defeats atomic mixing.
        bundle_evidence_eligible = (
            len(source_strategies) == 1
            and all(float(row.get("successful_contexts", 0)) > 0 for row in combination)
        )
        known_bundle_bonus = 0.15 if bundle_evidence_eligible else 0.0
        score = sum(float(row["controller_score"]) for row in combination) + known_bundle_bonus
        compatible.append(
            {
                "composition_key": composition_key,
                "selected_policy_by_role": {
                    role: row["policy_id"] for role, row in zip(required_roles, combination)
                },
                "controller_score": score,
                "known_bundle_bonus": known_bundle_bonus,
                "bundle_evidence_eligible": bundle_evidence_eligible,
                "source_strategy_ids": sorted(source_strategies),
                "compatibility": compatibility,
            }
        )
    if not compatible:
        raise ValueError("No compatible policy composition remains after capability and budget checks")
    compatible.sort(key=lambda row: (-row["controller_score"], row["composition_key"]))
    selected = compatible[0]
    selected_by_role = selected["selected_policy_by_role"]
    selected_ids = tuple(selected_by_role[role] for role in required_roles)
    base = {
        "run_id": run_id,
        "node_id": node_id,
        "stage": stage,
        "snapshot_sha256": snapshot_sha256,
        "selected_policy_by_role": selected_by_role,
        "visible_evidence": visible,
    }
    rationale = (
        f"composition_score={selected['controller_score']:.4g}",
        f"known_bundle_bonus={selected['known_bundle_bonus']:.4g}",
        "capability_contract=passed",
        "policy_credit=explicit_trace_required",
    )
    decision = PolicyCompositionDecision(
        decision_id="decision:" + _stable_id(base),
        run_id=run_id,
        node_id=node_id,
        stage=stage,
        snapshot_sha256=snapshot_sha256,
        visible_evidence=tuple(visible),
        candidate_policy_ids_by_role={
            role: tuple(row["policy_id"] for row in ranked_by_role[role])
            for role in required_roles
        },
        selected_policy_by_role=selected_by_role,
        selected_policy_ids=selected_ids,
        rationale=rationale,
        budget_before=budget_before,
        protected_metric_visible=False,
    )
    payload = decision.to_dict()
    payload["schema_version"] = "ml-scientist-policy-composition-decision-v1"
    payload["decision_kind"] = "atomic_policy_composition"
    payload["compatibility"] = selected["compatibility"]
    payload["source_strategy_ids"] = selected["source_strategy_ids"]
    payload["ranked_policy_candidates_by_role"] = ranked_by_role
    payload["compatible_composition_count"] = len(compatible)
    return payload


def select_strategy(
    *,
    run_id: str,
    node_id: str,
    stage: int,
    snapshot_sha256: str,
    candidates: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
    budget_before: dict[str, float | int],
    stage_complete: bool = False,
) -> dict[str, Any]:
    visible = filter_visible_evidence(evidence, stage_complete=stage_complete)
    remaining = float(budget_before.get("remaining_fraction", 1.0))
    diversity_is_low = any(row.get("metric_name") in {"Exploration Spread", "Exploration Uniqueness", "Exploration Reach"} and row.get("status") == "low" for row in visible)
    ranked = rank_strategy_candidates(candidates, remaining_budget_fraction=remaining, diversity_is_low=diversity_is_low)
    if not ranked:
        raise ValueError("No eligible strategy remains after evidence and budget filters")
    selected = ranked[0]
    rationale = tuple(
        f"{name}={value:.4g}" for name, value in selected["score_components"].items() if abs(value) > 1e-12
    ) or ("deterministic_tie_break",)
    base = {
        "run_id": run_id, "node_id": node_id, "stage": stage,
        "snapshot_sha256": snapshot_sha256, "candidate_strategy_ids": [row["strategy_id"] for row in ranked],
        "selected_strategy_id": selected["strategy_id"], "visible_evidence": visible,
    }
    decision = ControllerDecision(
        decision_id="decision:" + _stable_id(base),
        run_id=run_id,
        node_id=node_id,
        stage=stage,
        snapshot_sha256=snapshot_sha256,
        visible_evidence=tuple(visible),
        candidate_strategy_ids=tuple(row["strategy_id"] for row in ranked),
        selected_strategy_id=selected["strategy_id"],
        rationale=rationale,
        budget_before=budget_before,
        protected_metric_visible=False,
    )
    payload = asdict(decision)
    payload["ranked_candidates"] = ranked
    return payload


def route_after_evaluation(
    outcome: str,
    *,
    retry_count: int,
    retry_limit: int,
    budget_remaining: bool,
) -> dict[str, Any]:
    if outcome in {"PASSED_GATE", "VALID_CANDIDATE", "COMPLETE"}:
        action = "ADVANCE"
    elif not budget_remaining:
        action = "STOP_BUDGET"
    elif outcome == "INVALID_EXECUTION" and retry_count < retry_limit:
        action = "DEBUG_RETRY"
    elif outcome in {"VALID_REGRESSION", "STOCHASTIC_UNCERTAIN"} and retry_count < retry_limit:
        action = "SELECT_ALTERNATIVE_STRATEGY"
    elif outcome in {"FAILED_GATE", "VALID_NONPROMOTABLE"}:
        action = "BACKTRACK_OR_AMEND"
    else:
        action = "STOP_DOCUMENT_LIMITATION"
    return {
        "outcome": outcome,
        "action": action,
        "retry_count": retry_count,
        "retry_limit": retry_limit,
        "budget_remaining": budget_remaining,
    }


AMENDMENT_TARGETS = {
    "wording": "final-paper-writing",
    "citation": "final-paper-writing",
    "claim_evidence": "final-evidence-bundle",
    "statistics": "confirmatory-analysis",
    "experimental_design": "visible-validation",
    "insufficient_replication": "replication",
    "implementation": "code-modification",
    "metric_semantics": "process-evaluation",
}


def build_review_amendment(
    *,
    issue_id: str,
    issue_type: str,
    source_paper_node: str,
    protected_test_exposed: bool,
) -> dict[str, Any]:
    if issue_type not in AMENDMENT_TARGETS:
        raise ValueError(f"Unknown review issue type: {issue_type}")
    target = AMENDMENT_TARGETS[issue_type]
    empirical = target in {"code-modification", "visible-validation", "replication", "process-evaluation", "confirmatory-analysis"}
    return {
        "schema_version": "ml-scientist-amendment-v1",
        "issue_id": issue_id,
        "issue_type": issue_type,
        "source_node": source_paper_node,
        "target_node": target,
        "creates_new_graph_version": empirical,
        "invalidates_descendants": empirical or target in {"final-evidence-bundle", "confirmatory-analysis"},
        "requires_new_hidden_evaluation": bool(empirical and protected_test_exposed),
        "post_hoc_only_if_no_new_hidden_evaluation": bool(empirical and protected_test_exposed),
    }
