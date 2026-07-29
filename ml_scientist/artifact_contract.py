"""Materialize planner promises as immutable, auditable executor artifacts."""

from __future__ import annotations

import difflib
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def materialize_planner_artifacts(
    *,
    run_dir: Path,
    step_id: int,
    planner_output: dict[str, Any],
    before: dict[str, str],
    after: dict[str, str],
    research_context: dict[str, Any],
    validation: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    artifact_dir = run_dir / "planner_artifacts" / f"step_{step_id:04d}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    changed_files = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
    if not changed_files:
        raise ValueError("Artifact contract requires at least one changed target file")

    patch_parts: list[str] = []
    for path in changed_files:
        patch_parts.extend(difflib.unified_diff(
            before.get(path, "").splitlines(keepends=True),
            after.get(path, "").splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        ))
    patch_path = artifact_dir / "candidate.patch"
    patch_path.write_text("".join(patch_parts), encoding="utf-8")
    if patch_path.stat().st_size == 0:
        raise ValueError("Artifact contract produced an empty candidate patch")

    action = planner_output.get("action")
    rationale = planner_output.get("rationale")
    hypothesis = {
        "schema_version": "adaptive-hypothesis-artifact-v1",
        "step_id": step_id,
        "action": action,
        "rationale": rationale,
        "expected_metric_effect": planner_output.get("expected_metric_effect"),
        "risks": planner_output.get("risks") or [],
        "gate_checks": planner_output.get("gate_checks") or [],
        "stop_condition": planner_output.get("stop_condition"),
        "research_context": research_context,
    }
    hypothesis_path = artifact_dir / "hypothesis.json"
    _write_json(hypothesis_path, hypothesis)

    alternatives = []
    if isinstance(action, dict) and isinstance(action.get("comparison"), list):
        alternatives = action["comparison"]
    elif isinstance(rationale, dict) and isinstance(rationale.get("candidate_comparison"), list):
        alternatives = rationale["candidate_comparison"]
    journal = {
        "schema_version": "adaptive-idea-workspace-journal-v1",
        "step_id": step_id,
        "research_context": research_context,
        "alternatives": alternatives,
        "selected_action": action,
        "planned_policy_use": planner_output.get("planned_policy_use") or [],
        "parent_content_sha256": {
            path: _sha_bytes(before[path].encode("utf-8")) for path in changed_files if path in before
        },
        "candidate_content_sha256": {
            path: _sha_bytes(after[path].encode("utf-8")) for path in changed_files if path in after
        },
        "validation": validation,
        "resources": resources,
    }
    journal_path = artifact_dir / "idea_workspace_notes.json"
    _write_json(journal_path, journal)

    selection = {
        "schema_version": "adaptive-selection-record-v1",
        "step_id": step_id,
        "alternatives": alternatives,
        "selected_action": action,
        "eligibility_status": (
            "PENDING_VALIDATION" if validation is None else
            "ELIGIBLE" if validation.get("success") else
            "FAILED_GATE"
        ),
        "validation": validation,
        "resources": resources,
    }
    selection_path = artifact_dir / "selection_record.json"
    _write_json(selection_path, selection)

    manifest_inputs = [patch_path, hypothesis_path, journal_path, selection_path]
    manifest = {
        "schema_version": "adaptive-artifact-manifest-v1",
        "step_id": step_id,
        "artifacts": [
            {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": _sha_bytes(path.read_bytes()),
            }
            for path in manifest_inputs
        ],
        "validation_appended": validation is not None,
    }
    manifest_path = artifact_dir / "artifact_manifest.json"
    _write_json(manifest_path, manifest)

    summaries = {
        patch_path: {"changed_files": changed_files, "nonempty": True},
        hypothesis_path: hypothesis,
        journal_path: journal,
        selection_path: selection,
        manifest_path: manifest,
    }
    artifact_types = {
        patch_path: "candidate_patch",
        hypothesis_path: "hypothesis_record",
        journal_path: "idea_workspace_journal",
        selection_path: "selection_record",
        manifest_path: "artifact_manifest",
    }
    return [
        {
            "path": str(path),
            "artifact_type": artifact_types[path],
            "sha256": _sha_bytes(path.read_bytes()),
            "size_bytes": path.stat().st_size,
            "content_summary": summaries[path],
        }
        for path in [*manifest_inputs, manifest_path]
    ]
