"""Evidence-gated episodic memory and four-channel skill evolution.

Nothing is promoted merely because it appears in a plan or passes a unit test.
The registry requires hashed downstream evidence, preserves contradictions, and
checks a captured base checksum before a project skill can replace an existing
global skill implementation.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .policy_model import POLICY_ATTRIBUTE_BY_ROLE


CHANNELS = (
    "research_execution",
    "research_review",
    "paper_writing",
    "paper_review",
)
MATURITY = ("observation", "provisional_success", "repeated_success", "contradiction")
SKILL_BRANCHES = ("general", "task_specific", "action")


class EvidenceGateError(ValueError):
    """Raised when a skill promotion would outrun its evidence."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha_payload(payload: Any) -> str:
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class EvidenceGatedSkillRegistry:
    def __init__(self, root: Path, repository_commit: str = "unknown"):
        self.root = root
        self.registry_path = root / "skill_registry.json"
        self.event_path = root / "skill_evolution_events.jsonl"
        if self.registry_path.exists():
            self.registry = json.loads(self.registry_path.read_text(encoding="utf-8"))
            self._migrate_schema()
        else:
            self.registry = {
                "schema_version": "fml-scientist-skill-registry-v2",
                "repository_commit": repository_commit,
                "channels": {channel: [] for channel in CHANNELS},
                "skill_branches": list(SKILL_BRANCHES),
                "context_reward_baselines": {},
                "lifecycle": ["READ", "WRITE", "ASSESS", "GOVERN"],
                "promotion_policy": {
                    "observation": "source, proxy, or local evidence only",
                    "provisional_success": "at least one held-out complete real downstream pass with no regression",
                    "repeated_success": "at least two held-out qualifying passes in materially distinct contexts",
                    "contradiction": "comparable evidence invalidates the rule; narrow or roll back",
                },
                "mid_arm_mutation_allowed": False,
            }
            self._save()

    def _migrate_schema(self) -> None:
        """Upgrade older empty or populated registries without promoting evidence."""
        self.registry["schema_version"] = "fml-scientist-skill-registry-v2"
        self.registry.setdefault("skill_branches", list(SKILL_BRANCHES))
        self.registry.setdefault("context_reward_baselines", {})
        self.registry.setdefault("lifecycle", ["READ", "WRITE", "ASSESS", "GOVERN"])
        self.registry.setdefault("mid_arm_mutation_allowed", False)
        policy = self.registry.setdefault("promotion_policy", {})
        policy.update(
            {
                "observation": "source, proxy, or local evidence only",
                "provisional_success": "at least one held-out complete real downstream pass with no regression",
                "repeated_success": "at least two held-out qualifying passes in materially distinct contexts",
                "contradiction": "comparable evidence invalidates the rule; narrow or roll back",
            }
        )
        for skills in self.registry.get("channels", {}).values():
            for skill in skills:
                for version in skill.get("versions", []):
                    version.setdefault("skill_branch", "task_specific")
                    version.setdefault("research_contexts", [])
                    version.setdefault("structured_skill", {})
                    version.setdefault("source_episode_ids", [])
                    version.setdefault("utility", {"global": 0.5, "by_context": {}, "adoption_count": 0})
                    version.setdefault("distillation", {"origin": "legacy_registry"})
                    version.setdefault("fingerprint", _sha_payload({"rule": version.get("rule"), "role": version.get("policy_role")}))

    def _save(self) -> None:
        _atomic_json(self.registry_path, self.registry)

    def _append_event(self, action: str, skill_id: str, payload: dict[str, Any]) -> None:
        self.event_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": "fml-scientist-skill-event-v1",
            "timestamp": _now(),
            "action": action,
            "skill_id": skill_id,
            "payload": payload,
            "registry_sha256_after": _sha_payload(self.registry),
        }
        with self.event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _find(self, skill_id: str) -> dict[str, Any]:
        hits = [skill for skills in self.registry["channels"].values() for skill in skills if skill["skill_id"] == skill_id]
        if len(hits) != 1:
            raise KeyError(f"Expected exactly one project skill {skill_id!r}, found {len(hits)}")
        return hits[0]

    @staticmethod
    def _version(skill: dict[str, Any], version_number: int | None = None) -> dict[str, Any]:
        if version_number is None:
            return skill["versions"][-1]
        hits = [version for version in skill["versions"] if int(version["version"]) == int(version_number)]
        if len(hits) != 1:
            raise KeyError(f"Expected skill version {version_number!r}, found {len(hits)}")
        return hits[0]

    def propose(
        self,
        *,
        skill_id: str,
        channel: str,
        rule: str,
        evidence_paths: list[Path],
        acceptance_gates: list[str],
        rollback_condition: str,
        global_base_path: Path | None = None,
        policy_role: str | None = None,
        eligible_stages: list[str] | None = None,
        requires_capabilities: list[str] | None = None,
        provides_capabilities: list[str] | None = None,
        expected_cost: float = 0.1,
        skill_branch: str = "task_specific",
        research_contexts: list[str] | None = None,
        structured_skill: dict[str, Any] | None = None,
        source_episode_ids: list[str] | None = None,
        distillation: dict[str, Any] | None = None,
        fingerprint: str | None = None,
    ) -> dict[str, Any]:
        if channel not in CHANNELS:
            raise EvidenceGateError(f"Unknown channel {channel!r}")
        if any(skill["skill_id"] == skill_id for skills in self.registry["channels"].values() for skill in skills):
            raise EvidenceGateError(f"Skill already exists: {skill_id}")
        if not evidence_paths:
            raise EvidenceGateError("A skill observation needs at least one evidence artifact")
        if skill_branch not in SKILL_BRANCHES:
            raise EvidenceGateError(f"Unknown skill branch {skill_branch!r}")
        if policy_role is not None and policy_role not in POLICY_ATTRIBUTE_BY_ROLE:
            raise EvidenceGateError(f"Unknown atomic policy role {policy_role!r}")
        if policy_role is not None and not eligible_stages:
            raise EvidenceGateError("An atomic-policy skill requires at least one eligible stage")
        evidence = []
        for path in evidence_paths:
            resolved = path.resolve()
            if not resolved.is_file():
                raise EvidenceGateError(f"Evidence artifact does not exist: {resolved}")
            evidence.append({"path": str(resolved), "sha256": _sha_file(resolved)})
        base = None
        if global_base_path is not None:
            resolved_base = global_base_path.resolve()
            if not resolved_base.is_file():
                raise EvidenceGateError(f"Global base skill does not exist: {resolved_base}")
            base = {"path": str(resolved_base), "sha256": _sha_file(resolved_base)}
        skill = {
            "skill_id": skill_id,
            "channel": channel,
            "versions": [
                {
                    "version": 1,
                    "rule": rule,
                    "maturity": "observation",
                    "evidence": evidence,
                    "outcomes": [],
                    "acceptance_gates": acceptance_gates,
                    "rollback_condition": rollback_condition,
                    "global_base": base,
                    "policy_role": policy_role,
                    "eligible_stages": list(eligible_stages or []),
                    "requires_capabilities": list(requires_capabilities or []),
                    "provides_capabilities": list(provides_capabilities or []),
                    "expected_cost": float(expected_cost),
                    "credit_assignment": "explicit_policy_trace_only" if policy_role else None,
                    "skill_branch": skill_branch,
                    "research_contexts": list(research_contexts or []),
                    "structured_skill": dict(structured_skill or {}),
                    "source_episode_ids": list(source_episode_ids or []),
                    "distillation": dict(distillation or {}),
                    "fingerprint": fingerprint or _sha_payload(
                        {"rule": rule, "policy_role": policy_role, "eligible_stages": eligible_stages or []}
                    ),
                    "utility": {"global": 0.5, "by_context": {}, "adoption_count": 0},
                    "status": "CANDIDATE_NOT_ACTIVE",
                }
            ],
            "active_version": None,
            "evolution_history": [],
        }
        self.registry["channels"][channel].append(skill)
        self._save()
        self._append_event("PROPOSE", skill_id, {"channel": channel, "version": 1, "maturity": "observation"})
        return skill

    def patch(
        self,
        *,
        skill_id: str,
        rule: str,
        evidence_paths: list[Path],
        structured_skill: dict[str, Any],
        source_episode_ids: list[str],
        distillation: dict[str, Any],
        fingerprint: str,
    ) -> dict[str, Any]:
        """Create a non-active candidate version; the prior active version remains frozen."""
        skill = self._find(skill_id)
        if not evidence_paths:
            raise EvidenceGateError("A skill patch needs at least one evidence artifact")
        evidence = []
        for path in evidence_paths:
            resolved = path.resolve()
            if not resolved.is_file():
                raise EvidenceGateError(f"Evidence artifact does not exist: {resolved}")
            evidence.append({"path": str(resolved), "sha256": _sha_file(resolved)})
        previous = skill["versions"][-1]
        version = {
            **{key: value for key, value in previous.items() if key not in {"version", "rule", "maturity", "evidence", "outcomes", "status", "structured_skill", "source_episode_ids", "distillation", "fingerprint", "utility"}},
            "version": int(previous["version"]) + 1,
            "rule": rule,
            "maturity": "observation",
            "evidence": evidence,
            "outcomes": [],
            "structured_skill": dict(structured_skill),
            "source_episode_ids": list(source_episode_ids),
            "distillation": dict(distillation),
            "fingerprint": fingerprint,
            "utility": {"global": 0.5, "by_context": {}, "adoption_count": 0},
            "status": "CANDIDATE_NOT_ACTIVE",
        }
        skill["versions"].append(version)
        skill["evolution_history"].append(
            {"from": previous["version"], "to_candidate_version": version["version"], "at": _now()}
        )
        self._save()
        self._append_event("PATCH_PROPOSE", skill_id, {"version": version["version"], "maturity": "observation"})
        return version

    def record_outcome(
        self,
        *,
        skill_id: str,
        evidence_path: Path,
        context_id: str,
        real_downstream_complete: bool,
        no_regression: bool,
        metrics: dict[str, Any],
        held_out_validation: bool = False,
        source_episode_id: str | None = None,
        version_number: int | None = None,
    ) -> dict[str, Any]:
        resolved = evidence_path.resolve()
        if not resolved.is_file():
            raise EvidenceGateError(f"Outcome evidence does not exist: {resolved}")
        skill = self._find(skill_id)
        version = self._version(skill, version_number)
        if source_episode_id and source_episode_id in set(version.get("source_episode_ids", [])):
            raise EvidenceGateError("A skill cannot validate itself on an authoring episode")
        if version.get("policy_role"):
            trace = metrics.get("policy_trace")
            if not isinstance(trace, dict):
                raise EvidenceGateError("Atomic-policy skill outcomes require an explicit policy_trace metric")
            if trace.get("role") != version["policy_role"]:
                raise EvidenceGateError("Atomic-policy skill trace role does not match the proposed skill")
            if not trace.get("applied"):
                raise EvidenceGateError("A non-applied policy cannot receive promotion evidence")
            if trace.get("status") not in {"PASSED", "FAILED"}:
                raise EvidenceGateError("Atomic-policy skill trace requires PASSED or FAILED status")
            if not trace.get("evidence_artifact_refs"):
                raise EvidenceGateError("Atomic-policy skill trace requires an evidence artifact reference")
            if no_regression and trace.get("status") != "PASSED":
                raise EvidenceGateError("No-regression policy evidence must have PASSED status")
        outcome = {
            "context_id": context_id,
            "real_downstream_complete": real_downstream_complete,
            "no_regression": no_regression,
            "metrics": metrics,
            "held_out_validation": bool(held_out_validation),
            "source_episode_id": source_episode_id,
            "evidence_path": str(resolved),
            "evidence_sha256": _sha_file(resolved),
            "recorded_at": _now(),
        }
        version["outcomes"].append(outcome)
        self._save()
        self._append_event("RECORD_OUTCOME", skill_id, outcome)
        return outcome

    def assess_contextual_utility(
        self,
        *,
        skill_id: str,
        context_key: str,
        reward: float,
        baseline_reward: float,
        adopted: bool,
        harmful: bool = False,
        learning_rate: float = 0.2,
        version_number: int | None = None,
    ) -> dict[str, Any]:
        skill = self._find(skill_id)
        version = self._version(skill, version_number)
        utility = version.setdefault("utility", {"global": 0.5, "by_context": {}, "adoption_count": 0})
        advantage = float(reward) - float(baseline_reward)
        contribution = 0.0 if not adopted else advantage
        if harmful:
            contribution -= 0.5 + max(0.0, -advantage)
        current = float(utility.get("by_context", {}).get(context_key, utility.get("global", 0.5)))
        updated = min(1.0, max(0.0, current + float(learning_rate) * contribution))
        utility.setdefault("by_context", {})[context_key] = updated
        utility["adoption_count"] = int(utility.get("adoption_count", 0)) + int(adopted)
        values = list(utility["by_context"].values())
        utility["global"] = sum(values) / len(values) if values else 0.5
        baselines = self.registry.setdefault("context_reward_baselines", {})
        prior = baselines.get(context_key)
        baselines[context_key] = float(baseline_reward) if prior is None else 0.8 * float(prior) + 0.2 * float(reward)
        record = {
            "context_key": context_key,
            "reward": float(reward),
            "baseline_reward": float(baseline_reward),
            "advantage": advantage,
            "contribution": contribution,
            "utility_after": updated,
            "adopted": bool(adopted),
            "harmful": bool(harmful),
        }
        self._save()
        self._append_event("ASSESS_UTILITY", skill_id, record)
        return record

    def promote(self, skill_id: str, version_number: int | None = None) -> str:
        skill = self._find(skill_id)
        version = self._version(skill, version_number)
        base = version.get("global_base")
        if base:
            path = Path(base["path"])
            if not path.is_file() or _sha_file(path) != base["sha256"]:
                raise EvidenceGateError("Global base checksum drifted; refuse silent patch")
        qualifying = [
            outcome for outcome in version["outcomes"]
            if outcome["real_downstream_complete"] and outcome["no_regression"] and outcome.get("held_out_validation")
        ]
        contexts = {outcome["context_id"] for outcome in qualifying}
        current = version["maturity"]
        if current == "observation":
            if len(qualifying) < 1:
                raise EvidenceGateError("Provisional promotion requires one held-out complete real downstream no-regression pass")
            target = "provisional_success"
        elif current == "provisional_success":
            if len(contexts) < 2:
                raise EvidenceGateError("Repeated promotion requires two materially distinct held-out successful contexts")
            target = "repeated_success"
        elif current == "repeated_success":
            return current
        else:
            raise EvidenceGateError("Contradicted skills must be versioned or narrowed before promotion")
        version["maturity"] = target
        version["status"] = "ACTIVE"
        previous_active = skill.get("active_version")
        if previous_active is not None and int(previous_active) != int(version["version"]):
            previous_version = self._version(skill, int(previous_active))
            previous_version["status"] = "SUPERSEDED_BY_ACTIVE_VERSION"
        skill["active_version"] = version["version"]
        skill["evolution_history"].append({"from": current, "to": target, "at": _now()})
        self._save()
        self._append_event("PROMOTE", skill_id, {"from": current, "to": target})
        return target

    def contradict(
        self,
        skill_id: str,
        evidence_path: Path,
        reason: str,
        version_number: int | None = None,
    ) -> None:
        resolved = evidence_path.resolve()
        if not resolved.is_file():
            raise EvidenceGateError(f"Contradiction evidence does not exist: {resolved}")
        skill = self._find(skill_id)
        version = self._version(skill, version_number)
        prior = version["maturity"]
        version["maturity"] = "contradiction"
        version["status"] = "ROLLED_BACK_CONTRADICTION"
        version["contradiction"] = {"reason": reason, "path": str(resolved), "sha256": _sha_file(resolved)}
        if skill.get("active_version") == version["version"]:
            prior_candidates = [
                row for row in skill["versions"]
                if int(row["version"]) < int(version["version"])
                and row.get("maturity") in {"provisional_success", "repeated_success"}
                and row.get("maturity") != "contradiction"
            ]
            fallback = prior_candidates[-1] if prior_candidates else None
            skill["active_version"] = fallback["version"] if fallback else None
            if fallback:
                fallback["status"] = "ACTIVE_ROLLBACK_FALLBACK"
        skill["evolution_history"].append({"from": prior, "to": "contradiction", "at": _now()})
        self._save()
        self._append_event("CONTRADICT_AND_ROLL_BACK", skill_id, {"from": prior, "reason": reason})

    def snapshot(self, path: Path) -> dict[str, Any]:
        snapshot = {
            "schema_version": "fml-scientist-retrieval-snapshot-v1",
            "created_at": _now(),
            "registry_path": self.registry_path.name,
            "registry_sha256": _sha_file(self.registry_path),
            "event_ledger_sha256": _sha_file(self.event_path) if self.event_path.exists() else None,
            "pending_writes_visible": False,
            "mid_arm_mutation_allowed": False,
            "lifecycle": list(self.registry.get("lifecycle", [])),
            "skill_branches": list(self.registry.get("skill_branches", [])),
            "active_versions": {
                skill["skill_id"]: skill["active_version"]
                for skills in self.registry["channels"].values()
                for skill in skills
                if skill["active_version"] is not None
            },
        }
        _atomic_json(path, snapshot)
        return snapshot


def initialize_governance_artifacts(catalog: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_path = out_dir / "memory_registry.json"
    if not memory_path.exists():
        memory = {
            "schema_version": "fml-scientist-memory-registry-v2-graph-view",
            "repository_commit": catalog["repository_commit"],
            "retrieval_policy": "only frozen, source-hashed entries are visible to a running experiment arm",
            "canonical_store": "../knowledge_graph/knowledge_graph.json",
            "canonical_store_status": "PENDING_MEMORY_BUILD_GRAPH",
            "legacy_compatibility_view": True,
            "legacy_entries_migrated": False,
            "mid_arm_mutation_allowed": False,
            "entries": [
                {
                    "memory_id": "fml_repository_inventory",
                    "kind": "source_audit",
                    "claim": f"The checked-out benchmark registers {catalog['counts']['agents']} agents, {catalog['counts']['tasks']} tasks, and {catalog['counts']['process_metrics']} process metrics.",
                    "evidence": "../catalog/fml_catalog.json",
                    "maturity": "observation",
                    "paper_claim_eligible": False,
                }
            ],
            "pending_entries": [],
        }
        _atomic_json(memory_path, memory)
    else:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        memory["schema_version"] = "fml-scientist-memory-registry-v2-graph-view"
        memory.setdefault("canonical_store", "../knowledge_graph/knowledge_graph.json")
        memory.setdefault("canonical_store_status", "PENDING_MEMORY_BUILD_GRAPH")
        memory.setdefault("legacy_compatibility_view", True)
        memory.setdefault("legacy_entries_migrated", False)
        memory["mid_arm_mutation_allowed"] = False
        _atomic_json(memory_path, memory)
    registry = EvidenceGatedSkillRegistry(out_dir, catalog["repository_commit"])
    registry._save()
    snapshot = registry.snapshot(out_dir / "retrieval_snapshot.json")
    evolution_contract = {
        "schema_version": "ml-scientist-skill-evolution-contract-v2",
        "reference_design": "SkeMex-inspired Read-Write-Assess-Govern lifecycle adapted for ML research",
        "lifecycle": {
            "READ": "retrieve active branch-, context-, stage-, role-, utility-, and cost-compatible skills from a frozen snapshot",
            "WRITE": "after the arm, gate informative transitions and distill CREATE, PATCH, or NONE candidates",
            "ASSESS": "use held-out downstream outcomes, explicit adoption traces, contextual advantage, cost, and no-regression evidence",
            "GOVERN": "promote, retain, patch, merge, reject, or roll back without mid-arm mutation",
        },
        "skill_branches": list(SKILL_BRANCHES),
        "required_transition_fields": [
            "research_context", "state_before", "action", "next_state", "resource_telemetry",
            "policy_evaluations", "evidence_artifacts",
        ],
        "distillation_gates": [
            "exclude_infrastructure_failures", "exclude_mechanical_repetition", "retain_informative_failures",
            "explicit_policy_adoption_trace", "clear_trigger", "ordered_procedure", "scope_and_stage_declared",
            "novelty_or_patch_target", "acceptance_gates", "rollback_condition",
        ],
        "promotion_gates": [
            "held_out_validation", "real_downstream_complete", "no_regression",
            "authoring_episode_cannot_validate_candidate", "distinct_contexts_for_repeated_success",
        ],
        "protected_final_metrics_may_evolve_search_skills": False,
        "mid_arm_mutation_allowed": False,
    }
    _atomic_json(out_dir / "skill_evolution_contract.json", evolution_contract)
    status = {
        "schema_version": "fml-scientist-governance-status-v2",
            "memory_registry": memory_path.name,
            "skill_registry": registry.registry_path.name,
        "retrieval_snapshot": "retrieval_snapshot.json",
        "canonical_memory": "../knowledge_graph/knowledge_graph.json",
        "memory_representation": "typed_provenance_knowledge_graph_with_legacy_registry_view",
        "channels": list(CHANNELS),
        "candidate_skill_count": sum(len(skills) for skills in registry.registry["channels"].values()),
        "active_skill_count": len(snapshot["active_versions"]),
        "skill_evolution_contract": "skill_evolution_contract.json",
        "autonomous_lifecycle_available": True,
        "note": "The lifecycle can synthesize observation candidates from completed enriched episodes; promotion still requires later held-out downstream evidence.",
    }
    _atomic_json(out_dir / "governance_status.json", status)
    return status
