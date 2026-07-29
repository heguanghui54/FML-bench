"""Post-run Read-Write-Assess-Govern skill evolution for ML research.

The engine consumes enriched, completed episodes only. It never launches an
experiment and never makes writes visible to the graph snapshot that authored
them. New skills remain observations until a later held-out downstream episode
validates them.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .governance import EvidenceGateError, EvidenceGatedSkillRegistry


DISTILLABLE_OUTCOMES = {
    "PASSED_GATE", "VALID_CANDIDATE", "COMPLETE", "VALID_REGRESSION",
    "STOCHASTIC_UNCERTAIN", "VALID_NONPROMOTABLE", "FAILED_GATE",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_payload(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _tokens(value: Any) -> set[str]:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
    return {token for token in re.findall(r"[a-z0-9_]+", text.lower()) if len(token) > 1}


def _similarity(left: Any, right: Any) -> float:
    left_tokens, right_tokens = _tokens(left), _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _project_skill_ref(value: str) -> tuple[str, int | None]:
    match = re.fullmatch(r"skill:(.+):v(\d+)", value)
    if match:
        return match.group(1), int(match.group(2))
    return value.removeprefix("skill:"), None


class SkillEvolutionEngine:
    """Distill skills and apply later held-out utility/governance evidence."""

    def __init__(
        self,
        *,
        registry: EvidenceGatedSkillRegistry,
        out_dir: Path,
        knowledge_graph: dict[str, Any] | None = None,
    ):
        self.registry = registry
        self.out_dir = out_dir
        self.knowledge_graph = knowledge_graph or {"nodes": [], "edges": []}

    def _load_episodes(self, path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        episodes: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError as exc:
                rejected.append({"line": line_number, "reason": "INVALID_JSON", "detail": str(exc)})
                continue
            episode["_source_line"] = line_number
            episodes.append(episode)
        return episodes, rejected

    @staticmethod
    def _gate_episode(episode: dict[str, Any]) -> tuple[bool, str]:
        if episode.get("schema_version") != "ml-scientist-adaptive-episode-v2":
            return False, "LEGACY_OR_INCOMPLETE_TRANSITION_SCHEMA"
        if episode.get("protected_metric_used_for_routing") or episode.get("protected_metric_used_for_skill_evolution"):
            return False, "PROTECTED_METRIC_LEAKAGE"
        if any(row.get("observability") == "PROTECTED_FINAL_ONLY" for row in episode.get("visible_evidence", [])):
            return False, "PROTECTED_METRIC_VISIBLE"
        required = {"research_context", "state_before", "action", "next_state", "resource_telemetry"}
        missing = sorted(key for key in required if key not in episode)
        if missing:
            return False, "MISSING_TRANSITION_FIELDS:" + ",".join(missing)
        if episode.get("outcome") == "INVALID_EXECUTION":
            return False, "INFRASTRUCTURE_OR_INVALID_EXECUTION"
        if episode.get("outcome") not in DISTILLABLE_OUTCOMES:
            return False, "NON_DISTILLABLE_OUTCOME"
        signals = [
            row.get("learning_signal")
            for row in episode.get("policy_evaluations", [])
            if row.get("learning_signal", {}).get("intent") in {"CREATE", "PATCH"}
            and row.get("applied")
            and row.get("status") in {"PASSED", "FAILED"}
            and row.get("evidence_artifact_refs")
        ]
        validations = episode.get("skill_validation_evaluations", [])
        if not signals and not validations:
            return False, "NO_REUSABLE_SIGNAL_OR_SKILL_VALIDATION"
        return True, "INFORMATIVE_TRANSITION"

    def _registry_rows(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [
            (skill, version)
            for skills in self.registry.registry.get("channels", {}).values()
            for skill in skills
            for version in skill.get("versions", [])
        ]

    def _graph_rows(self, node_types: set[str]) -> list[dict[str, Any]]:
        return [
            row for row in self.knowledge_graph.get("nodes", [])
            if row.get("node_type") in node_types
        ]

    def _failed_episode_ids(self) -> set[str]:
        failure_ids = {
            row.get("node_id") for row in self._graph_rows({"Failure"})
        }
        return {
            row.get("source") for row in self.knowledge_graph.get("edges", [])
            if row.get("target") in failure_ids
        }

    @staticmethod
    def _containment(left: Any, right: Any) -> float:
        left_tokens, right_tokens = _tokens(left), _tokens(right)
        if not left_tokens:
            return 0.0
        return len(left_tokens & right_tokens) / len(left_tokens)

    def _novelty_route(
        self,
        signal: dict[str, Any],
        *,
        episode: dict[str, Any] | None = None,
    ) -> tuple[str, str | None, float]:
        if episode and episode.get("outcome") in {"FAILED_GATE", "VALID_REGRESSION", "STOCHASTIC_UNCERTAIN"}:
            return "NEGATIVE_MEMORY", episode.get("episode_id"), 1.0
        failed_episode_ids = self._failed_episode_ids()
        for row in self._graph_rows({"Episode"}):
            if row.get("node_id") not in failed_episode_ids:
                continue
            searchable = {
                "label": row.get("label"),
                "attributes": row.get("attributes", {}),
            }
            score = self._containment(signal, searchable)
            if score >= 0.60:
                return "NEGATIVE_MEMORY", row.get("node_id"), score

        fingerprint = self._fingerprint(signal)
        best: tuple[float, str | None] = (0.0, None)
        for skill, version in self._registry_rows():
            if version.get("fingerprint") == fingerprint:
                return "DUPLICATE", skill["skill_id"], 1.0
            if version.get("policy_role") != signal.get("policy_role"):
                continue
            if version.get("skill_branch") != signal.get("branch"):
                continue
            score = _similarity(version.get("structured_skill") or version.get("rule", ""), signal)
            if score > best[0]:
                best = (score, skill["skill_id"])
        if signal.get("intent") == "PATCH":
            return "PATCH", str(signal["target_skill_id"]), best[0]
        if best[0] >= 0.82 and best[1]:
            return "PATCH", best[1], best[0]

        for row in self._graph_rows({"Policy"}):
            attributes = row.get("attributes", {})
            if attributes.get("policy_role") != signal.get("policy_role"):
                continue
            score = self._containment(signal, {
                "label": row.get("label"),
                "instruction": attributes.get("instruction"),
                "eligible_stages": attributes.get("eligible_stages", []),
            })
            if score >= 0.85:
                return "DUPLICATE", row.get("node_id"), score
        return "CREATE", None, best[0]

    @staticmethod
    def _fingerprint(signal: dict[str, Any]) -> str:
        semantic = {
            "branch": signal.get("branch"),
            "policy_role": signal.get("policy_role"),
            "eligible_stages": sorted(signal.get("eligible_stages", [])),
            "reusable_pattern": signal.get("reusable_pattern"),
            "trigger": signal.get("trigger"),
            "procedure": signal.get("procedure", []),
        }
        return _sha_payload(semantic)

    @staticmethod
    def _rule(signal: dict[str, Any]) -> str:
        trigger = json.dumps(signal.get("trigger", {}), ensure_ascii=False, sort_keys=True)
        steps = "; ".join(signal.get("procedure", []))
        expected = json.dumps(signal.get("expected_effect", {}), ensure_ascii=False, sort_keys=True)
        exclusions = "; ".join(signal.get("do_not_use_when", [])) or "none recorded"
        return f"When {trigger}, execute: {steps}. Expected effect: {expected}. Do not use when: {exclusions}."

    @staticmethod
    def _context_values(episode: dict[str, Any]) -> list[str]:
        context = episode.get("research_context", {})
        return list(dict.fromkeys(str(value) for value in (
            context.get("task_domain"), context.get("task_id"), context.get("research_request")
        ) if value))

    def _write_candidate_artifact(
        self,
        *,
        skill_id: str,
        episode: dict[str, Any],
        signal: dict[str, Any],
        intent: str,
        target_skill_id: str | None,
        similarity: float,
    ) -> Path:
        artifact = {
            "schema_version": "ml-scientist-distilled-skill-candidate-v1",
            "skill_id": skill_id,
            "intent": intent,
            "target_skill_id": target_skill_id,
            "source_episode_id": episode["episode_id"],
            "source_decision_id": episode.get("decision_id"),
            "research_context": episode.get("research_context", {}),
            "state_before": episode.get("state_before", {}),
            "next_state": episode.get("next_state", {}),
            "resource_telemetry": episode.get("resource_telemetry", {}),
            "outcome": episode.get("outcome"),
            "structured_skill": signal,
            "novelty_similarity": similarity,
            "authoring_evidence_is_not_validation": True,
            "created_at": _now(),
        }
        path = self.out_dir / "candidates" / f"{skill_id}-{self._fingerprint(signal)[:8]}.json"
        _atomic_json(path, artifact)
        return path

    def _distill(self, episode: dict[str, Any]) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for evaluation in episode.get("policy_evaluations", []):
            signal = evaluation.get("learning_signal") or {}
            if signal.get("intent") == "NONE":
                actions.append({"intent": "NONE", "reason": signal.get("reason", "no reusable pattern")})
                continue
            if not evaluation.get("applied") or evaluation.get("status") not in {"PASSED", "FAILED"}:
                continue
            if not evaluation.get("evidence_artifact_refs") or signal.get("intent") not in {"CREATE", "PATCH"}:
                continue
            signal = {**signal, "policy_role": evaluation["role"]}
            intent, target, similarity = self._novelty_route(signal, episode=episode)
            fingerprint = self._fingerprint(signal)
            if intent == "DUPLICATE":
                actions.append({
                    "intent": "DUPLICATE", "reason": "DUPLICATE_SKILL_OR_POLICY", "target_skill_id": target,
                    "fingerprint": fingerprint,
                })
                continue
            if intent == "NEGATIVE_MEMORY":
                slug = re.sub(r"[^a-z0-9]+", "-", f"negative-{evaluation['role']}").strip("-")[:36]
                negative_id = f"hs-ml-{slug}-{fingerprint[:8]}"
                artifact_path = self._write_candidate_artifact(
                    skill_id=negative_id,
                    episode=episode,
                    signal=signal,
                    intent=intent,
                    target_skill_id=target,
                    similarity=similarity,
                )
                actions.append({
                    "intent": "NEGATIVE_MEMORY",
                    "memory_id": negative_id,
                    "matching_failure_or_episode": target,
                    "active_skill": False,
                    "evidence": str(artifact_path.resolve()),
                    "fingerprint": fingerprint,
                    "planner_default": "EXCLUDE_UNLESS_ARTIFACT_BACKED_DISTINCTION",
                })
                continue
            stage = (signal.get("eligible_stages") or [episode.get("node_type", "research")])[0]
            slug = re.sub(r"[^a-z0-9]+", "-", f"{evaluation['role']}-{stage}").strip("-")[:36]
            skill_id = target or f"hs-ml-{slug}-{fingerprint[:8]}"
            artifact_path = self._write_candidate_artifact(
                skill_id=skill_id,
                episode=episode,
                signal=signal,
                intent=intent,
                target_skill_id=target,
                similarity=similarity,
            )
            distillation = {
                "method": "two_pass_transition_analysis_and_mutation",
                "analysis_source": "executor_policy_trace_learning_signal",
                "mutation_intent": intent,
                "novelty_similarity": similarity,
                "authoring_window": episode.get("run_id"),
            }
            if intent == "PATCH":
                normalized_target, _ = _project_skill_ref(str(target))
                self.registry.patch(
                    skill_id=normalized_target,
                    rule=self._rule(signal),
                    evidence_paths=[artifact_path],
                    structured_skill=signal,
                    source_episode_ids=[episode["episode_id"]],
                    distillation=distillation,
                    fingerprint=fingerprint,
                )
                skill_id = normalized_target
            else:
                self.registry.propose(
                    skill_id=skill_id,
                    channel="research_execution",
                    rule=self._rule(signal),
                    evidence_paths=[artifact_path],
                    acceptance_gates=signal.get("acceptance_gates") or ["held_out_downstream_no_regression"],
                    rollback_condition=signal.get("rollback_condition", "held-out no-regression gate fails"),
                    policy_role=evaluation["role"],
                    eligible_stages=signal.get("eligible_stages") or [episode.get("node_type", "research")],
                    requires_capabilities=signal.get("requires_capabilities", []),
                    provides_capabilities=signal.get("provides_capabilities", []),
                    expected_cost=float(signal.get("expected_cost", 0.1)),
                    skill_branch=signal.get("branch", "task_specific"),
                    research_contexts=self._context_values(episode),
                    structured_skill=signal,
                    source_episode_ids=[episode["episode_id"]],
                    distillation=distillation,
                    fingerprint=fingerprint,
                )
            actions.append({
                "intent": intent,
                "skill_id": skill_id,
                "maturity": "observation",
                "active": False,
                "evidence": str(artifact_path.resolve()),
                "fingerprint": fingerprint,
                "authoring_episode_cannot_validate": True,
            })
        return actions

    def _assess_and_govern(
        self,
        *,
        episode: dict[str, Any],
        episodes_path: Path,
        frozen_skill_ids: set[str],
    ) -> list[dict[str, Any]]:
        actions: list[dict[str, Any]] = []
        for evaluation in episode.get("skill_validation_evaluations", []):
            skill_id, parsed_version = _project_skill_ref(str(evaluation.get("skill_id", "")))
            version_number = evaluation.get("version", parsed_version)
            if skill_id not in frozen_skill_ids:
                actions.append({"skill_id": skill_id, "action": "REJECT", "reason": "NOT_IN_AUTHORING_FROZEN_SNAPSHOT"})
                continue
            if not evaluation.get("held_out_validation"):
                actions.append({"skill_id": skill_id, "action": "REJECT", "reason": "NOT_HELD_OUT"})
                continue
            try:
                reward = float(evaluation["reward"])
                baseline_reward = float(evaluation["baseline_reward"])
            except (KeyError, TypeError, ValueError):
                actions.append({"skill_id": skill_id, "action": "REJECT", "reason": "MISSING_NUMERIC_REWARD_OR_BASELINE"})
                continue
            context_id = str(evaluation.get("context_id") or (
                f"{episode.get('task_id', 'unknown')}::{episode.get('node_type', episode.get('node_id', 'unknown'))}"
            ))
            status = evaluation.get("status")
            applied = bool(evaluation.get("applied"))
            trace = {
                "role": evaluation.get("role"),
                "applied": applied,
                "status": status,
                "evidence_artifact_refs": list(evaluation.get("evidence_artifact_refs", [])),
            }
            try:
                self.registry.record_outcome(
                    skill_id=skill_id,
                    evidence_path=episodes_path,
                    context_id=context_id,
                    real_downstream_complete=bool(evaluation.get("real_downstream_complete")),
                    no_regression=bool(evaluation.get("no_regression")),
                    metrics={"policy_trace": trace, "reward": reward, "baseline_reward": baseline_reward},
                    held_out_validation=True,
                    source_episode_id=episode["episode_id"],
                    version_number=int(version_number) if version_number is not None else None,
                )
                utility = self.registry.assess_contextual_utility(
                    skill_id=skill_id,
                    context_key=context_id,
                    reward=reward,
                    baseline_reward=baseline_reward,
                    adopted=applied,
                    harmful=bool(evaluation.get("harmful")),
                    version_number=int(version_number) if version_number is not None else None,
                )
                if evaluation.get("harmful") and evaluation.get("comparable_evidence"):
                    self.registry.contradict(
                        skill_id,
                        episodes_path,
                        str(evaluation.get("reason", "Comparable held-out evidence found harm")),
                        version_number=int(version_number) if version_number is not None else None,
                    )
                    actions.append({"skill_id": skill_id, "action": "ROLLBACK", "utility": utility})
                    continue
                try:
                    maturity = self.registry.promote(
                        skill_id,
                        version_number=int(version_number) if version_number is not None else None,
                    )
                    actions.append({"skill_id": skill_id, "action": "PROMOTE", "maturity": maturity, "utility": utility})
                except EvidenceGateError as exc:
                    actions.append({"skill_id": skill_id, "action": "RETAIN", "reason": str(exc), "utility": utility})
            except (EvidenceGateError, KeyError) as exc:
                actions.append({"skill_id": skill_id, "action": "REJECT", "reason": str(exc)})
        return actions

    def evolve(self, episodes_path: Path) -> dict[str, Any]:
        """Process one closed learning window without exposing its writes mid-arm."""
        episodes_path = episodes_path.resolve()
        if not episodes_path.is_file():
            raise FileNotFoundError(episodes_path)
        frozen_skill_ids = {
            skill["skill_id"]
            for skills in self.registry.registry.get("channels", {}).values()
            for skill in skills
        }
        episodes, rejected = self._load_episodes(episodes_path)
        accepted: list[dict[str, Any]] = []
        write_actions: list[dict[str, Any]] = []
        assessment_actions: list[dict[str, Any]] = []
        for episode in episodes:
            passed, reason = self._gate_episode(episode)
            identity = {"episode_id": episode.get("episode_id"), "line": episode.get("_source_line")}
            if not passed:
                rejected.append({**identity, "reason": reason})
                continue
            accepted.append({**identity, "reason": reason})
            assessment_actions.extend(
                self._assess_and_govern(
                    episode=episode,
                    episodes_path=episodes_path,
                    frozen_skill_ids=frozen_skill_ids,
                )
            )
            try:
                write_actions.extend(self._distill(episode))
            except (EvidenceGateError, KeyError, ValueError) as exc:
                write_actions.append(
                    {"episode_id": episode.get("episode_id"), "intent": "REJECT", "reason": str(exc)}
                )
        snapshot = self.registry.snapshot(self.out_dir / "retrieval_snapshot_next_window.json")
        report = {
            "schema_version": "ml-scientist-skill-evolution-report-v1",
            "lifecycle": ["READ", "WRITE", "ASSESS", "GOVERN"],
            "episodes_path": str(episodes_path),
            "window_frozen_skill_ids": sorted(frozen_skill_ids),
            "accepted_episodes": accepted,
            "rejected_episodes": rejected,
            "write_actions": write_actions,
            "assessment_governance_actions": assessment_actions,
            "new_writes_visible_to_authoring_window": False,
            "protected_final_metrics_used": False,
            "next_window_snapshot": snapshot,
            "created_at": _now(),
        }
        _atomic_json(
            self.out_dir / "trajectory_buffer_report.json",
            {
                "schema_version": "ml-scientist-trajectory-buffer-report-v1",
                "accepted_episodes": accepted,
                "rejected_episodes": rejected,
                "gate": "informative completed transitions only; invalid execution and protected leakage excluded",
            },
        )
        _atomic_json(
            self.out_dir / "skill_distillation_report.json",
            {
                "schema_version": "ml-scientist-skill-distillation-report-v1",
                "write_actions": write_actions,
                "intents": ["CREATE", "PATCH", "DUPLICATE", "NEGATIVE_MEMORY", "NONE"],
                "authoring_evidence_is_validation": False,
            },
        )
        _atomic_json(
            self.out_dir / "utility_assessment_report.json",
            {
                "schema_version": "ml-scientist-utility-assessment-report-v1",
                "assessment_actions": assessment_actions,
                "utility_basis": "held-out context reward minus context baseline with explicit adoption trace",
            },
        )
        _atomic_json(
            self.out_dir / "skill_governance_report.json",
            {
                "schema_version": "ml-scientist-skill-governance-report-v1",
                "governance_actions": assessment_actions,
                "next_window_snapshot": snapshot,
                "mid_arm_mutation_allowed": False,
                "rollback_available": True,
            },
        )
        _atomic_json(self.out_dir / "skill_evolution_report.json", report)
        return report
