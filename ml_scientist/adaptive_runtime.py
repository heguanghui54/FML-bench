"""Persisted Stage 4-6 runtime for graph-bound conditional research plans.

This module schedules and records decisions. It deliberately does not launch an
experiment by itself; executors submit outcomes so a paused campaign cannot be
restarted merely by inspecting or initializing controller state.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adaptive_controller import (
    build_review_amendment,
    route_after_evaluation,
    select_policy_composition,
    select_strategy,
)
from .strategy_operator import compile_policy_operator, compile_stage_operator
from .governance import EvidenceGatedSkillRegistry
from .skill_evolution import SkillEvolutionEngine


PASS_OUTCOMES = {"PASSED_GATE", "VALID_CANDIDATE", "COMPLETE"}
FAIL_OUTCOMES = {"INVALID_EXECUTION", "VALID_REGRESSION", "STOCHASTIC_UNCERTAIN", "VALID_NONPROMOTABLE", "FAILED_GATE", "BLOCKED_BUDGET"}
LEARNING_INTENTS = {"CREATE", "PATCH", "NONE"}
SKILL_BRANCHES = {"general", "task_specific", "action"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class AdaptiveResearchRuntime:
    def __init__(
        self,
        *,
        plan: dict[str, Any],
        graph: dict[str, Any],
        run_id: str,
        state_path: Path,
        event_path: Path | None = None,
    ):
        expected = plan.get("knowledge_graph_snapshot", {}).get("content_sha256")
        actual = graph.get("content_sha256")
        if expected and expected != actual:
            raise ValueError("Plan and knowledge-graph snapshot do not match")
        self.plan = plan
        self.graph = graph
        self.run_id = run_id
        self.state_path = state_path
        self.event_path = event_path or state_path.with_name("adaptive_runtime_events.jsonl")
        self.nodes = {node["node_id"]: node for node in plan["nodes"]}
        self.outgoing: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for node in self.nodes.values():
            for dependency in node["depends_on"]:
                self.outgoing[dependency].append(node["node_id"])
        if state_path.is_file():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            if self.state.get("run_id") != run_id:
                raise ValueError("Persisted runtime state belongs to another run ID")
            if self.state.get("plan_sha256") != _sha_payload(plan) or self.state.get("graph_content_sha256") != actual:
                raise ValueError("Persisted runtime state belongs to another frozen plan or graph")
        else:
            self.state = {
                "schema_version": "ml-scientist-adaptive-runtime-v3",
                "run_id": run_id,
                "plan_sha256": _sha_payload(plan),
                "graph_content_sha256": actual,
                "experiment_execution_enabled": False,
                "node_status": {
                    node_id: ("READY" if not node["depends_on"] else "LOCKED_DEPENDENCIES")
                    for node_id, node in self.nodes.items()
                },
                "attempts": {node_id: 0 for node_id in self.nodes},
                "decisions": [],
                "outcomes": [],
                "amendments": [],
                "evidence_version": 1,
                "hidden_evaluation_required": False,
                "created_at": _now(),
                "updated_at": _now(),
            }
            self._save()

    def _save(self) -> None:
        self.state["updated_at"] = _now()
        _atomic_json(self.state_path, self.state)

    def _event(self, action: str, payload: dict[str, Any]) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"timestamp": _now(), "run_id": self.run_id, "action": action, "payload": payload}, ensure_ascii=False) + "\n")

    def ready_nodes(self) -> list[str]:
        return sorted(node_id for node_id, status in self.state["node_status"].items() if status == "READY")

    def _unlock_descendants(self) -> None:
        for node_id, node in self.nodes.items():
            if self.state["node_status"][node_id] != "LOCKED_DEPENDENCIES":
                continue
            if all(self.state["node_status"].get(dependency) == "COMPLETE" for dependency in node["depends_on"]):
                self.state["node_status"][node_id] = "READY"

    def choose_operator(
        self,
        node_id: str,
        *,
        evidence: list[dict[str, Any]],
        budget_before: dict[str, float | int],
        stage_complete: bool = False,
    ) -> dict[str, Any]:
        if self.state["node_status"].get(node_id) != "READY":
            raise ValueError(f"Node is not ready: {node_id}")
        node = self.nodes[node_id]
        if node.get("metadata", {}).get("controller_routing_allowed") is False:
            raise ValueError(f"Controller routing is forbidden for {node_id}")
        graph_nodes = {row["node_id"]: row for row in self.graph.get("nodes", [])}
        policy_candidates_by_role = node.get("metadata", {}).get("policy_candidates_by_role", {})
        if policy_candidates_by_role:
            candidates_by_role: dict[str, list[dict[str, Any]]] = {}
            for role in node["metadata"]["policy_roles"]:
                candidates_by_role[role] = []
                for policy_id in policy_candidates_by_role.get(role, []):
                    row = graph_nodes.get(policy_id)
                    if row is None:
                        continue
                    attributes = row.get("attributes", {})
                    if row.get("node_type") == "Skill" and not attributes.get("active"):
                        continue
                    candidates_by_role[role].append(
                        {
                            "policy_id": policy_id,
                            "label": row.get("label", policy_id),
                            "role": role,
                            "maturity": row.get("maturity", "observation"),
                            "retrieval_score": 0.0,
                            "successful_contexts": attributes.get("successful_contexts_by_stage", {}).get(
                                node["node_type"], 0
                            ),
                            "failure_count": attributes.get("failure_count_by_stage", {}).get(
                                node["node_type"], 0
                            ),
                            "expected_cost": attributes.get("expected_cost", 0.1),
                            "diversity_contribution": attributes.get("diversity_contribution", 0.0),
                            "contextual_utility": attributes.get("utility", {}).get("by_context", {}).get(
                                f"{self.plan['task_id']}::{node['node_type']}",
                                attributes.get("utility", {}).get("global", 0.5),
                            ),
                            "attributes": attributes,
                        }
                    )
            prior_outcomes = [row for row in self.state["outcomes"] if row.get("node_id") == node_id]
            must_change_composition = bool(
                prior_outcomes
                and prior_outcomes[-1].get("route", {}).get("action")
                in {"SELECT_ALTERNATIVE_STRATEGY", "SELECT_ALTERNATIVE_POLICY_COMPOSITION"}
            )
            excluded = (
                {
                    tuple(decision.get("selected_policy_by_role", {}).get(role, "") for role in node["metadata"]["policy_roles"])
                    for decision in self.state["decisions"]
                    if decision.get("node_id") == node_id and decision.get("selected_policy_ids")
                }
                if must_change_composition
                else set()
            )
            decision = select_policy_composition(
                run_id=self.run_id,
                node_id=node_id,
                node_type=node["node_type"],
                stage=4,
                snapshot_sha256=self.graph["content_sha256"],
                candidates_by_role=candidates_by_role,
                required_roles=list(node["metadata"]["policy_roles"]),
                evidence=evidence,
                budget_before=budget_before,
                stage_complete=stage_complete,
                excluded_compositions=excluded,
            )
            decision["operator_contract"] = compile_policy_operator(
                graph=self.graph,
                selected_policy_by_role=decision["selected_policy_by_role"],
                node=node,
                task_context={
                    "task_id": self.plan["task_id"],
                    "research_request": self.plan.get("research_request", ""),
                    "metric_bindings": node.get("metadata", {}).get("metric_bindings", []),
                },
                visible_evidence=decision["visible_evidence"],
                budget=budget_before,
            )
            event_action = "POLICY_COMPOSITION_SELECTED"
        else:
            decision = self._choose_legacy_strategy(
                node=node,
                graph_nodes=graph_nodes,
                evidence=evidence,
                budget_before=budget_before,
                stage_complete=stage_complete,
            )
            event_action = "STRATEGY_SELECTED"
        self.state["node_status"][node_id] = "SCHEDULED"
        self.state["decisions"].append(decision)
        self._event(event_action, decision)
        self._save()
        return decision

    def choose_strategy(
        self,
        node_id: str,
        *,
        evidence: list[dict[str, Any]],
        budget_before: dict[str, float | int],
        stage_complete: bool = False,
    ) -> dict[str, Any]:
        """Compatibility alias; research nodes now select policy compositions."""
        return self.choose_operator(
            node_id,
            evidence=evidence,
            budget_before=budget_before,
            stage_complete=stage_complete,
        )

    def _choose_legacy_strategy(
        self,
        *,
        node: dict[str, Any],
        graph_nodes: dict[str, dict[str, Any]],
        evidence: list[dict[str, Any]],
        budget_before: dict[str, float | int],
        stage_complete: bool,
    ) -> dict[str, Any]:
        candidates = []
        for strategy_id in node.get("metadata", {}).get("strategy_candidates", []):
            row = graph_nodes.get(strategy_id, {})
            attributes = row.get("attributes", {})
            candidates.append(
                {
                    "strategy_id": strategy_id,
                    "maturity": row.get("maturity", "observation"),
                    "retrieval_score": 0.0,
                    "successful_contexts": attributes.get("successful_contexts", 0),
                    "failure_count": attributes.get("failure_count", 0),
                    "expected_cost": attributes.get("expected_cost", 0.5),
                    "diversity_contribution": attributes.get("diversity_contribution", 0.0),
                    "contextual_utility": attributes.get("utility", {}).get("global", 0.5),
                }
            )
        decision = select_strategy(
            run_id=self.run_id,
            node_id=node["node_id"],
            stage=4,
            snapshot_sha256=self.graph["content_sha256"],
            candidates=candidates,
            evidence=evidence,
            budget_before=budget_before,
            stage_complete=stage_complete,
        )
        decision["operator_contract"] = compile_stage_operator(
            graph=self.graph,
            strategy_id=decision["selected_strategy_id"],
            node=node,
            task_context={
                "task_id": self.plan["task_id"],
                "research_request": self.plan.get("research_request", ""),
                "metric_bindings": node.get("metadata", {}).get("metric_bindings", []),
            },
            visible_evidence=decision["visible_evidence"],
            budget=budget_before,
        )
        return decision

    def record_outcome(
        self,
        node_id: str,
        *,
        outcome: str,
        evidence_artifacts: list[dict[str, Any]],
        budget_remaining: bool,
        policy_evaluations: list[dict[str, Any]] | None = None,
        budget_after: dict[str, float | int] | None = None,
        telemetry: dict[str, Any] | None = None,
        state_observations: dict[str, Any] | None = None,
        skill_validation_evaluations: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if outcome not in PASS_OUTCOMES | FAIL_OUTCOMES:
            raise ValueError(f"Unknown runtime outcome: {outcome}")
        if self.state["node_status"].get(node_id) not in {"READY", "SCHEDULED", "RUNNING"}:
            raise ValueError(f"Cannot record outcome for node in {self.state['node_status'].get(node_id)}")
        decisions = [row for row in self.state["decisions"] if row.get("node_id") == node_id]
        latest_decision = decisions[-1] if decisions else None
        normalized_policy_evaluations = self._validate_policy_evaluations(
            latest_decision,
            policy_evaluations,
        )
        self.state["attempts"][node_id] += 1
        retry_limit = int(self.nodes[node_id]["inner_loop"]["budget"].get("debug_depth", self.nodes[node_id]["inner_loop"]["budget"].get("candidate_steps", 1)))
        route = route_after_evaluation(
            outcome,
            retry_count=self.state["attempts"][node_id],
            retry_limit=retry_limit,
            budget_remaining=budget_remaining,
        )
        if latest_decision and latest_decision.get("selected_policy_ids") and route["action"] == "SELECT_ALTERNATIVE_STRATEGY":
            route["action"] = "SELECT_ALTERNATIVE_POLICY_COMPOSITION"
        if outcome in PASS_OUTCOMES:
            self.state["node_status"][node_id] = "COMPLETE"
            if node_id == "protected-final-test":
                self.state["hidden_evaluation_required"] = False
            self._unlock_descendants()
        elif route["action"] in {
            "DEBUG_RETRY",
            "SELECT_ALTERNATIVE_STRATEGY",
            "SELECT_ALTERNATIVE_POLICY_COMPOSITION",
            "BACKTRACK_OR_AMEND",
        }:
            self.state["node_status"][node_id] = "READY"
        elif route["action"] == "STOP_BUDGET":
            self.state["node_status"][node_id] = "BLOCKED_BUDGET"
        else:
            self.state["node_status"][node_id] = outcome
        record = {
            "node_id": node_id,
            "decision_id": latest_decision.get("decision_id") if latest_decision else None,
            "outcome": outcome,
            "route": route,
            "evidence_artifacts": evidence_artifacts,
            "policy_evaluations": normalized_policy_evaluations,
            "budget_after": dict(budget_after or {}),
            "resource_telemetry": self._normalize_telemetry(
                latest_decision,
                budget_after=budget_after or {},
                telemetry=telemetry or {},
                state_observations=state_observations or {},
            ),
            "state_observations": dict(state_observations or {}),
            "skill_validation_evaluations": list(skill_validation_evaluations or []),
            "recorded_at": _now(),
        }
        self.state["outcomes"].append(record)
        self._event("OUTCOME_RECORDED", record)
        self._save()
        return record

    def _validate_policy_evaluations(
        self,
        decision: dict[str, Any] | None,
        evaluations: list[dict[str, Any]] | None,
    ) -> list[dict[str, Any]]:
        if not decision or not decision.get("selected_policy_ids"):
            return []
        if evaluations is None:
            raise ValueError("Atomic-policy decisions require one explicit policy evaluation per selected role")
        expected_by_role = decision["selected_policy_by_role"]
        seen_roles: set[str] = set()
        normalized: list[dict[str, Any]] = []
        allowed_statuses = {"PASSED", "FAILED", "NOT_APPLIED", "INCONCLUSIVE"}
        for evaluation in evaluations:
            role = evaluation.get("role")
            policy_id = evaluation.get("policy_id")
            if role not in expected_by_role:
                raise ValueError(f"Unexpected policy evaluation role: {role}")
            if role in seen_roles:
                raise ValueError(f"Duplicate policy evaluation role: {role}")
            if expected_by_role[role] != policy_id:
                raise ValueError(f"Policy evaluation does not match selected policy for role {role}")
            status = evaluation.get("status")
            if status not in allowed_statuses:
                raise ValueError(f"Unknown policy evaluation status: {status}")
            applied = bool(evaluation.get("applied"))
            evidence_refs = list(evaluation.get("evidence_artifact_refs", []))
            if applied and status == "NOT_APPLIED":
                raise ValueError(f"Applied policy cannot be NOT_APPLIED: {policy_id}")
            if not applied and status in {"PASSED", "FAILED"}:
                raise ValueError(f"Non-applied policy cannot receive success or failure credit: {policy_id}")
            if applied and status in {"PASSED", "FAILED"} and not evidence_refs:
                raise ValueError(f"Policy credit requires an evidence artifact reference: {policy_id}")
            seen_roles.add(role)
            row = {
                "policy_id": policy_id,
                "role": role,
                "applied": applied,
                "status": status,
                "evidence_artifact_refs": evidence_refs,
                "gate_checks": list(evaluation.get("gate_checks", [])),
            }
            learning_signal = evaluation.get("learning_signal")
            if learning_signal is not None:
                row["learning_signal"] = self._validate_learning_signal(
                    learning_signal,
                    role=role,
                    default_stage=self.nodes[decision["node_id"]]["node_type"],
                )
            normalized.append(row)
        missing = sorted(set(expected_by_role) - seen_roles)
        if missing:
            raise ValueError(f"Missing policy evaluations for roles: {missing}")
        return normalized

    @staticmethod
    def _validate_learning_signal(
        signal: dict[str, Any],
        *,
        role: str,
        default_stage: str,
    ) -> dict[str, Any]:
        if not isinstance(signal, dict):
            raise ValueError("A learning_signal must be an object")
        intent = str(signal.get("intent", "CREATE")).upper()
        if intent not in LEARNING_INTENTS:
            raise ValueError(f"Unknown learning intent: {intent}")
        if intent == "NONE":
            return {"intent": "NONE", "reason": str(signal.get("reason", "no reusable pattern"))}
        branch = signal.get("branch", "task_specific")
        if branch not in SKILL_BRANCHES:
            raise ValueError(f"Unknown skill branch: {branch}")
        pattern = str(signal.get("reusable_pattern", "")).strip()
        procedure = [str(step).strip() for step in signal.get("procedure", []) if str(step).strip()]
        trigger = signal.get("trigger", {})
        if not pattern or not procedure or not trigger:
            raise ValueError("CREATE/PATCH learning signals require reusable_pattern, trigger, and procedure")
        target = signal.get("target_skill_id")
        if intent == "PATCH" and not target:
            raise ValueError("PATCH learning signals require target_skill_id")
        return {
            "intent": intent,
            "target_skill_id": target,
            "branch": branch,
            "policy_role": role,
            "eligible_stages": list(signal.get("eligible_stages") or [default_stage]),
            "reusable_pattern": pattern,
            "trigger": trigger,
            "procedure": procedure,
            "expected_effect": signal.get("expected_effect", {}),
            "do_not_use_when": list(signal.get("do_not_use_when", [])),
            "requires_capabilities": list(signal.get("requires_capabilities", [])),
            "provides_capabilities": list(signal.get("provides_capabilities", [])),
            "expected_cost": float(signal.get("expected_cost", 0.1)),
            "acceptance_gates": list(signal.get("acceptance_gates", [])),
            "rollback_condition": str(signal.get("rollback_condition", "held-out no-regression gate fails")),
        }

    @staticmethod
    def _metric_value(rows: list[dict[str, Any]], names: set[str]) -> float | None:
        for row in rows:
            if row.get("metric_name") in names or row.get("name") in names:
                try:
                    return float(row["value"])
                except (KeyError, TypeError, ValueError):
                    continue
        return None

    def _normalize_telemetry(
        self,
        decision: dict[str, Any] | None,
        *,
        budget_after: dict[str, float | int],
        telemetry: dict[str, Any],
        state_observations: dict[str, Any],
    ) -> dict[str, Any]:
        normalized = dict(telemetry)
        before = dict((decision or {}).get("budget_before", {}))
        for before_key, after_key in (
            ("tokens_remaining", "tokens_remaining"),
            ("token_budget_remaining", "token_budget_remaining"),
        ):
            if "tokens_consumed" not in normalized and before_key in before and after_key in budget_after:
                normalized["tokens_consumed"] = max(0, float(before[before_key]) - float(budget_after[after_key]))
        visible = list((decision or {}).get("visible_evidence", []))
        after_metrics = state_observations.get("metrics", [])
        if not isinstance(after_metrics, list):
            after_metrics = []
        diversity_names = {"Exploration Spread", "Exploration Uniqueness", "Exploration Reach"}
        diversity_before = self._metric_value(visible, diversity_names)
        diversity_after = self._metric_value(after_metrics, diversity_names)
        if diversity_before is not None:
            normalized.setdefault("diversity_before", diversity_before)
        if diversity_after is not None:
            normalized.setdefault("diversity_after", diversity_after)
        if diversity_before is not None and diversity_after is not None:
            normalized.setdefault("diversity_delta", diversity_after - diversity_before)
        normalized.setdefault("measurement_status", "EXECUTOR_REPORTED" if telemetry else "PARTIALLY_DERIVED")
        return normalized

    def apply_review_issue(
        self,
        *,
        issue_id: str,
        issue_type: str,
        source_paper_node: str = "paper-peer-review",
        protected_test_exposed: bool,
    ) -> dict[str, Any]:
        amendment = build_review_amendment(
            issue_id=issue_id,
            issue_type=issue_type,
            source_paper_node=source_paper_node,
            protected_test_exposed=protected_test_exposed,
        )
        amendment["evidence_version_before"] = int(self.state.get("evidence_version", 1))
        if amendment["creates_new_graph_version"]:
            self.state["evidence_version"] = amendment["evidence_version_before"] + 1
        amendment["evidence_version_after"] = int(self.state.get("evidence_version", 1))
        if amendment["requires_new_hidden_evaluation"]:
            self.state["hidden_evaluation_required"] = True
            amendment["status"] = "REQUIRES_NEW_HIDDEN_EVALUATION"
            amendment["claim_eligibility"] = "POST_HOC_ONLY_UNTIL_NEW_HIDDEN_EVALUATION"
        else:
            amendment["status"] = "AMENDMENT_READY"
            amendment["claim_eligibility"] = "REQUIRES_REBUILT_DEPENDENCY_EVIDENCE"
        target = amendment["target_node"]
        if amendment["invalidates_descendants"]:
            queue = [target]
            seen: set[str] = set()
            superseded_nodes: list[str] = []
            while queue:
                current = queue.pop(0)
                if current in seen:
                    continue
                seen.add(current)
                if current != target:
                    superseded_nodes.append(current)
                    # The prior evidence version is superseded, but the node must
                    # return to a dependency-locked state so it can be rebuilt.
                    self.state["node_status"][current] = "LOCKED_DEPENDENCIES"
                queue.extend(self.outgoing.get(current, []))
            self.state["node_status"][target] = "READY"
            amendment["superseded_nodes"] = superseded_nodes
            amendment["superseded_outcome_indices"] = [
                index for index, row in enumerate(self.state.get("outcomes", []))
                if row.get("node_id") in set(superseded_nodes)
            ]
        self.state["amendments"].append(amendment)
        self._event("REVIEW_AMENDMENT_CREATED", amendment)
        self._save()
        return amendment

    def run_skill_evolution(
        self,
        *,
        episodes_path: Path,
        governance_root: Path,
        out_dir: Path,
        budget_before: dict[str, float | int] | None = None,
    ) -> dict[str, Any]:
        """Execute the Stage-7 post-arm lifecycle without launching experiments."""
        node_id = "skill-evolution"
        if node_id not in self.nodes:
            raise ValueError("The plan does not contain a skill-evolution node")
        if self.state["node_status"].get(node_id) != "READY":
            raise ValueError("The skill-evolution node is not dependency-ready")
        decision = self.choose_operator(
            node_id,
            evidence=[],
            budget_before=dict(budget_before or {"remaining_fraction": 1.0}),
            stage_complete=True,
        )
        registry = EvidenceGatedSkillRegistry(
            governance_root,
            self.graph.get("repository_commit", "unknown"),
        )
        report = SkillEvolutionEngine(
            registry=registry,
            out_dir=out_dir,
            knowledge_graph=self.graph,
        ).evolve(episodes_path)
        blocking_rejections = [
            row for row in report["rejected_episodes"]
            if row.get("reason", "").startswith(
                ("LEGACY_", "PROTECTED_", "MISSING_TRANSITION_FIELDS", "NON_DISTILLABLE_OUTCOME")
            )
        ]
        blocking_rejections.extend(
            row for row in report["write_actions"] + report["assessment_governance_actions"]
            if row.get("intent") == "REJECT" or row.get("action") == "REJECT"
        )
        outcome = "FAILED_GATE" if blocking_rejections else "PASSED_GATE"
        evidence_artifacts = [
            {"path": str((out_dir / name).resolve())}
            for name in (
                "trajectory_buffer_report.json", "skill_distillation_report.json",
                "utility_assessment_report.json", "skill_governance_report.json",
            )
        ]
        outcome_record = self.record_outcome(
            node_id,
            outcome=outcome,
            evidence_artifacts=evidence_artifacts,
            budget_remaining=True,
            state_observations={
                "accepted_episode_count": len(report["accepted_episodes"]),
                "rejected_episode_count": len(report["rejected_episodes"]),
                "write_action_count": len(report["write_actions"]),
                "governance_action_count": len(report["assessment_governance_actions"]),
                "blocking_rejections": blocking_rejections,
            },
        )
        return {
            "schema_version": "ml-scientist-runtime-skill-evolution-result-v1",
            "decision_id": decision["decision_id"],
            "outcome_record": outcome_record,
            "lifecycle_report": report,
            "experiment_execution_started": False,
        }

    def export_episode(self, node_id: str) -> dict[str, Any]:
        decisions = [row for row in self.state["decisions"] if row["node_id"] == node_id]
        outcomes = [row for row in self.state["outcomes"] if row["node_id"] == node_id]
        if not decisions or not outcomes:
            raise ValueError("An episode requires both a decision and an outcome")
        latest_decision = decisions[-1]
        latest_outcome = outcomes[-1]
        research_context = {
            "research_request": self.plan.get("research_request", ""),
            "task_id": self.plan["task_id"],
            "task_domain": self.plan.get("research_context", {}).get("task_domain"),
            "task_description": self.plan.get("research_context", {}).get("task_description"),
            "node_id": node_id,
            "node_type": self.nodes[node_id]["node_type"],
        }
        learning_signals = [
            evaluation["learning_signal"]
            for evaluation in latest_outcome.get("policy_evaluations", [])
            if evaluation.get("learning_signal")
        ]
        return {
            "schema_version": "ml-scientist-adaptive-episode-v2",
            "episode_id": f"{self.run_id}:{node_id}:{len(outcomes)}",
            "run_id": self.run_id,
            "task_id": self.plan["task_id"],
            "node_id": node_id,
            "node_type": self.nodes[node_id]["node_type"],
            "decision_id": latest_decision["decision_id"],
            "selected_strategy_id": latest_decision.get("selected_strategy_id"),
            "selected_policy_ids": latest_decision.get("selected_policy_ids", []),
            "selected_policy_by_role": latest_decision.get("selected_policy_by_role", {}),
            "visible_evidence": latest_decision["visible_evidence"],
            "research_context": research_context,
            "state_before": {
                "node_status": "READY",
                "visible_evidence": latest_decision["visible_evidence"],
                "budget": latest_decision.get("budget_before", {}),
            },
            "action": {
                "decision_id": latest_decision["decision_id"],
                "selected_strategy_id": latest_decision.get("selected_strategy_id"),
                "selected_policy_by_role": latest_decision.get("selected_policy_by_role", {}),
                "operator_schema_version": latest_decision.get("operator_contract", {}).get("schema_version"),
            },
            "outcome": latest_outcome["outcome"],
            "next_state": {
                "node_status": self.state["node_status"][node_id],
                "route": latest_outcome.get("route", {}),
                "ready_nodes": self.ready_nodes(),
                "observations": latest_outcome.get("state_observations", {}),
            },
            "budget_before": latest_decision.get("budget_before", {}),
            "budget_after": latest_outcome.get("budget_after", {}),
            "resource_telemetry": latest_outcome.get("resource_telemetry", {}),
            "evidence_artifacts": latest_outcome["evidence_artifacts"],
            "policy_evaluations": latest_outcome.get("policy_evaluations", []),
            "learning_signals": learning_signals,
            "skill_validation_evaluations": latest_outcome.get("skill_validation_evaluations", []),
            "snapshot_sha256": self.graph["content_sha256"],
            "protected_metric_used_for_routing": False,
            "maturity": "observation",
        }
