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


CHANNELS = (
    "research_execution",
    "research_review",
    "paper_writing",
    "paper_review",
)
MATURITY = ("observation", "provisional_success", "repeated_success", "contradiction")


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
        else:
            self.registry = {
                "schema_version": "fml-scientist-skill-registry-v1",
                "repository_commit": repository_commit,
                "channels": {channel: [] for channel in CHANNELS},
                "promotion_policy": {
                    "observation": "source, proxy, or local evidence only",
                    "provisional_success": "at least one complete real downstream pass with no regression",
                    "repeated_success": "at least two qualifying passes in materially distinct contexts",
                    "contradiction": "comparable evidence invalidates the rule; narrow or roll back",
                },
                "mid_arm_mutation_allowed": False,
            }
            self._save()

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
    ) -> dict[str, Any]:
        if channel not in CHANNELS:
            raise EvidenceGateError(f"Unknown channel {channel!r}")
        if any(skill["skill_id"] == skill_id for skills in self.registry["channels"].values() for skill in skills):
            raise EvidenceGateError(f"Skill already exists: {skill_id}")
        if not evidence_paths:
            raise EvidenceGateError("A skill observation needs at least one evidence artifact")
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

    def record_outcome(
        self,
        *,
        skill_id: str,
        evidence_path: Path,
        context_id: str,
        real_downstream_complete: bool,
        no_regression: bool,
        metrics: dict[str, Any],
    ) -> dict[str, Any]:
        resolved = evidence_path.resolve()
        if not resolved.is_file():
            raise EvidenceGateError(f"Outcome evidence does not exist: {resolved}")
        skill = self._find(skill_id)
        version = skill["versions"][-1]
        outcome = {
            "context_id": context_id,
            "real_downstream_complete": real_downstream_complete,
            "no_regression": no_regression,
            "metrics": metrics,
            "evidence_path": str(resolved),
            "evidence_sha256": _sha_file(resolved),
            "recorded_at": _now(),
        }
        version["outcomes"].append(outcome)
        self._save()
        self._append_event("RECORD_OUTCOME", skill_id, outcome)
        return outcome

    def promote(self, skill_id: str) -> str:
        skill = self._find(skill_id)
        version = skill["versions"][-1]
        base = version.get("global_base")
        if base:
            path = Path(base["path"])
            if not path.is_file() or _sha_file(path) != base["sha256"]:
                raise EvidenceGateError("Global base checksum drifted; refuse silent patch")
        qualifying = [
            outcome for outcome in version["outcomes"]
            if outcome["real_downstream_complete"] and outcome["no_regression"]
        ]
        contexts = {outcome["context_id"] for outcome in qualifying}
        current = version["maturity"]
        if current == "observation":
            if len(qualifying) < 1:
                raise EvidenceGateError("Provisional promotion requires one complete real downstream no-regression pass")
            target = "provisional_success"
        elif current == "provisional_success":
            if len(contexts) < 2:
                raise EvidenceGateError("Repeated promotion requires two materially distinct successful contexts")
            target = "repeated_success"
        elif current == "repeated_success":
            return current
        else:
            raise EvidenceGateError("Contradicted skills must be versioned or narrowed before promotion")
        version["maturity"] = target
        version["status"] = "ACTIVE"
        skill["active_version"] = version["version"]
        skill["evolution_history"].append({"from": current, "to": target, "at": _now()})
        self._save()
        self._append_event("PROMOTE", skill_id, {"from": current, "to": target})
        return target

    def contradict(self, skill_id: str, evidence_path: Path, reason: str) -> None:
        resolved = evidence_path.resolve()
        if not resolved.is_file():
            raise EvidenceGateError(f"Contradiction evidence does not exist: {resolved}")
        skill = self._find(skill_id)
        version = skill["versions"][-1]
        prior = version["maturity"]
        version["maturity"] = "contradiction"
        version["status"] = "ROLLED_BACK_CONTRADICTION"
        version["contradiction"] = {"reason": reason, "path": str(resolved), "sha256": _sha_file(resolved)}
        skill["active_version"] = None
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
            "schema_version": "fml-scientist-memory-registry-v1",
            "repository_commit": catalog["repository_commit"],
            "retrieval_policy": "only frozen, source-hashed entries are visible to a running experiment arm",
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
    registry = EvidenceGatedSkillRegistry(out_dir, catalog["repository_commit"])
    snapshot = registry.snapshot(out_dir / "retrieval_snapshot.json")
    status = {
        "schema_version": "fml-scientist-governance-status-v1",
            "memory_registry": memory_path.name,
            "skill_registry": registry.registry_path.name,
            "retrieval_snapshot": "retrieval_snapshot.json",
        "channels": list(CHANNELS),
        "candidate_skill_count": sum(len(skills) for skills in registry.registry["channels"].values()),
        "active_skill_count": len(snapshot["active_versions"]),
        "note": "No skill is synthesized or promoted from inventory and unit-test evidence alone.",
    }
    _atomic_json(out_dir / "governance_status.json", status)
    return status
