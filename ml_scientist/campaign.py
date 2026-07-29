"""Resumable execution of a frozen FML controlled-run matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CampaignError(ValueError):
    pass


GOVERNED_CAMPAIGNS = {
    "resource_matched_sensitivity": {
        "authorization_scope": "resource_matched_sensitivity_campaign",
        "protocol": "resource_matched_sensitivity_protocol.json",
    },
    "confirmatory_heldout_transfer": {
        "authorization_scope": "heldout_transfer_confirmatory_campaign",
        "protocol": "heldout_transfer_protocol.json",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run_matrix(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
            raise CampaignError("JSON run matrix must be a list of objects")
        rows = payload
    else:
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    required = {"run_id", "phase", "agent", "task", "trial", "model", "provider", "result_root", "command"}
    if not rows:
        raise CampaignError("Run matrix is empty")
    missing = required - set(rows[0])
    if missing:
        raise CampaignError(f"Run matrix is missing columns: {sorted(missing)}")
    run_ids = [row["run_id"] for row in rows]
    if len(run_ids) != len(set(run_ids)):
        raise CampaignError("Run IDs are not unique")
    if any(row["model"] == "SET_MODEL" or "SET_MODEL" in row["command"] for row in rows):
        raise CampaignError("Run matrix still contains SET_MODEL; freeze one model before execution")
    return rows


def _summary_paths(repo: Path, row: dict[str, Any]) -> list[Path]:
    base = repo / row["result_root"] / row["agent"] / row["task"]
    return sorted(base.rglob("summary.json")) if base.exists() else []


def _matching_summaries(paths: list[Path], row: dict[str, Any]) -> list[Path]:
    matches = []
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            seed_matches = int(payload.get("experimental_seed")) == int(row.get("seed", "0"))
        except (TypeError, ValueError):
            seed_matches = False
        if (
            payload.get("agent") == row["agent"]
            and payload.get("benchmark") == row["task"]
            and payload.get("model") == row["model"]
            and payload.get("provider") == row["provider"]
            and seed_matches
        ):
            matches.append(path)
    return matches


def _summary_validity(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on governed summaries that lack executable evidence."""
    if row.get("phase") not in GOVERNED_CAMPAIGNS:
        return {"paper_result_eligible": True, "reasons": []}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"paper_result_eligible": False, "reasons": [f"INVALID_JSON:{exc}"]}
    reasons: list[str] = []
    ledger = summary.get("budget_ledger") or {}
    counts = summary.get("execution_counts")
    steps = summary.get("val_steps")
    activations = summary.get("candidate_activation")
    if summary.get("schema_version") != "fml-summary-v2":
        reasons.append("SUMMARY_SCHEMA_NOT_V2")
    if summary.get("resource_accounting_status") != "COMPLETE_V2":
        reasons.append("RESOURCE_ACCOUNTING_NOT_COMPLETE_V2")
    if ledger.get("profile_name") != "matched-v1" or ledger.get("passed") is not True:
        reasons.append("MATCHED_BUDGET_LEDGER_MISSING_OR_FAILED")
    if not isinstance(counts, dict):
        reasons.append("EXECUTION_COUNTS_MISSING")
        counts = {}
    if not isinstance(steps, list):
        reasons.append("VALIDATION_TRAJECTORY_MISSING")
        steps = []
    candidate_count = int(counts.get("candidate_validation_count") or 0)
    if candidate_count and not steps:
        reasons.append("COUNTED_VALIDATION_HAS_NO_TRAJECTORY_STEP")
    if candidate_count and not isinstance(activations, list):
        reasons.append("COUNTED_VALIDATION_HAS_NO_ACTIVATION_LIST")
    elif candidate_count and len(activations) < candidate_count:
        reasons.append("COUNTED_VALIDATION_HAS_INCOMPLETE_ACTIVATION_EVIDENCE")
    infrastructure_signatures = (
        "subprocess.TimeoutExpired", "Command '['ssh'", "SSH connection",
        "remote workspace integrity", "No route to host", "Connection timed out",
    )
    for step in steps:
        error = str((step.get("val_result") or {}).get("error") or "") if isinstance(step, dict) else ""
        if any(signature in error for signature in infrastructure_signatures):
            reasons.append("VALIDATION_INFRASTRUCTURE_FAILURE")
            break
    usage = ledger.get("usage") or {}
    for key in ("candidate_validation_count", "pre_test_validation_count", "protected_test_count"):
        if int(counts.get(key) or 0) != int(usage.get(key) or 0):
            reasons.append(f"LEDGER_COUNT_MISMATCH:{key}")
    return {"paper_result_eligible": not reasons, "reasons": reasons}


def _persist_summary_validity(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    validity = _summary_validity(path, row)
    payload = {
        "schema_version": "fml-campaign-summary-validity-v1",
        "run_id": row["run_id"],
        "summary_path": str(path.resolve()),
        "summary_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **validity,
        "classification": "VALID_PAPER_RESULT" if validity["paper_result_eligible"] else "INVALID_RESULT_CONTRACT",
    }
    sidecar = path.parent / "campaign_validity.json"
    sidecar.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    payload["artifact_path"] = str(sidecar.resolve())
    return payload


def _command_argv(command: str) -> list[str]:
    """Run frozen Python commands with the campaign controller interpreter.

    Protocol rows intentionally remain portable (``python ...``), but the
    controller may have been launched from a dedicated virtual environment.
    Reusing ``sys.executable`` prevents a subprocess from silently falling back
    to the macOS system Python and changing the installed dependency set.
    """
    argv = shlex.split(command)
    if argv and argv[0] in {"python", "python3"}:
        argv[0] = sys.executable
    return argv


def _append_event(path: Path, event: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def _terminate_process_group(process: subprocess.Popen[Any], grace_seconds: float = 10.0) -> None:
    """Terminate exactly the campaign-owned child session and its descendants."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    process.wait()


def _run_logged_command(argv: list[str], *, cwd: Path, log_handle: Any) -> int:
    """Run one arm in an isolated session and never leak it on interruption."""
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        return process.wait()
    except BaseException:
        _terminate_process_group(process)
        raise


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise CampaignError(f"Governed campaign is missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CampaignError(f"Governed campaign has invalid {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise CampaignError(f"Governed campaign {label} must be a JSON object: {path}")
    return payload


def _freeze_governed_campaign(
    *, matrix_path: Path, rows: list[dict[str, Any]], log_dir: Path
) -> dict[str, Any] | None:
    phases = {str(row["phase"]) for row in rows}
    governed = phases & set(GOVERNED_CAMPAIGNS)
    if not governed:
        return None
    if len(governed) != 1 or phases != governed:
        raise CampaignError("A governed campaign manifest may contain exactly one governed phase")
    phase = next(iter(governed))
    requirements = GOVERNED_CAMPAIGNS[phase]
    root = matrix_path.parent
    authorization_path = root / "execution_authorization.json"
    protocol_path = root / requirements["protocol"]
    authorization = _read_object(authorization_path, "execution authorization")
    protocol = _read_object(protocol_path, "frozen protocol")
    scope = authorization.get("authorized_scope") or {}
    if not scope.get("local_ubuntu_gpu_experiments") or not scope.get(requirements["authorization_scope"]):
        raise CampaignError(f"Execution authorization does not permit phase {phase}")
    for row in rows:
        command = str(row["command"])
        if row.get("budget_profile") != "matched-v1" or "--budget-profile matched-v1" not in command:
            raise CampaignError(f"Governed row is not resource matched: {row['run_id']}")
        if "--eval-backend ssh" not in command or "--ssh-host ubuntu-heshi" not in command:
            raise CampaignError(f"Governed row does not use the authorized SSH backend: {row['run_id']}")

    evidence_paths = [matrix_path, authorization_path, protocol_path]
    evidence_status: dict[str, Any] = {}
    if phase == "confirmatory_heldout_transfer":
        exposure_path = root / "heldout_exposure_audit.json"
        preflight_path = root / "preflight" / "preflight.json"
        exposure = _read_object(exposure_path, "held-out exposure audit")
        preflight = _read_object(preflight_path, "held-out preflight")
        if exposure.get("status") != "PASSED_UNEXPOSED":
            raise CampaignError("Held-out exposure audit did not pass before execution")
        if not preflight.get("ready") or not preflight.get("heldout_data_ready"):
            raise CampaignError("Held-out preflight or external-disk dataset gate did not pass")
        evidence_paths.extend([exposure_path, preflight_path])
        evidence_status.update({
            "heldout_exposure_status": exposure.get("status"),
            "heldout_data_ready": preflight.get("heldout_data_ready"),
        })

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in evidence_paths
    }
    manifest = {
        "schema_version": "fml-campaign-execution-manifest-v1",
        "phase": phase,
        "matrix_path": str(matrix_path.resolve()),
        "run_count": len(rows),
        "model": sorted({str(row["model"]) for row in rows}),
        "provider": sorted({str(row["provider"]) for row in rows}),
        "budget_profile": "matched-v1",
        "execution_backend": "ssh",
        "ssh_host": "ubuntu-heshi",
        "artifact_sha256": hashes,
        "authorization": {
            "local_ubuntu_gpu_experiments": True,
            "cloud_purchase": bool(scope.get("cloud_purchase", False)),
            "restart_i4h_automatically": bool(scope.get("restart_i4h_automatically", False)),
        },
        **evidence_status,
    }
    manifest_path = log_dir / "campaign_execution_manifest.json"
    if manifest_path.is_file():
        existing = _read_object(manifest_path, "campaign execution manifest")
        if existing != manifest:
            raise CampaignError("Frozen campaign execution manifest does not match current artifacts")
    else:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return manifest


def _write_campaign_state(
    *,
    path: Path,
    matrix_path: Path,
    selected_run_count: int,
    outcomes: list[dict[str, Any]],
    dry_run: bool,
    events_path: Path,
    current_run_id: str | None,
) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for event in outcomes:
        counts[event["status"]] = counts.get(event["status"], 0) + 1
    processed = len(outcomes)
    state = {
        "schema_version": "fml-scientist-campaign-state-v1",
        "matrix_path": str(matrix_path.resolve()),
        "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
        "selected_run_count": selected_run_count,
        "processed_run_count": processed,
        "remaining_run_count": selected_run_count - processed,
        "current_run_id": current_run_id,
        "status_counts": counts,
        "dry_run": dry_run,
        "event_ledger": str(events_path.resolve()),
        "complete": (
            bool(selected_run_count)
            and processed == selected_run_count
            and all(event["status"] in {"COMPLETE", "SKIPPED_ALREADY_COMPLETE"} for event in outcomes)
        ),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return state


def run_campaign(
    *,
    matrix_path: Path,
    repo: Path,
    log_dir: Path,
    phases: set[str] | None = None,
    max_runs: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    rows = load_run_matrix(matrix_path)
    if phases:
        known = {row["phase"] for row in rows}
        unknown = phases - known
        if unknown:
            raise CampaignError(f"Unknown phases: {sorted(unknown)}")
        rows = [row for row in rows if row["phase"] in phases]
    if max_runs is not None:
        if max_runs < 1:
            raise CampaignError("max_runs must be positive")
        rows = rows[:max_runs]
    log_dir.mkdir(parents=True, exist_ok=True)
    _freeze_governed_campaign(matrix_path=matrix_path, rows=rows, log_dir=log_dir)
    events_path = log_dir / "campaign_events.jsonl"
    state_path = log_dir / "campaign_state.json"
    outcomes: list[dict[str, Any]] = []
    state = _write_campaign_state(
        path=state_path,
        matrix_path=matrix_path,
        selected_run_count=len(rows),
        outcomes=outcomes,
        dry_run=dry_run,
        events_path=events_path,
        current_run_id=None,
    )
    for row in rows:
        before = _summary_paths(repo, row)
        matching_before = _matching_summaries(before, row)
        before_validity = {path: _persist_summary_validity(path, row) for path in matching_before}
        eligible_before = [path for path in matching_before if before_validity[path]["paper_result_eligible"]]
        invalid_before = [path for path in matching_before if not before_validity[path]["paper_result_eligible"]]
        event = {
            "schema_version": "fml-scientist-campaign-event-v1",
            "run_id": row["run_id"],
            "phase": row["phase"],
            "agent": row["agent"],
            "task": row["task"],
            "trial": int(row["trial"]),
            "command_sha256": hashlib.sha256(row["command"].encode()).hexdigest(),
            "started_at": _now(),
        }
        # A dry run validates the frozen command/matrix without interpreting or
        # mutating result state. This keeps preflight deterministic after real
        # campaign artifacts have accumulated in the repository.
        if dry_run:
            event.update({"status": "DRY_RUN_READY", "summary_paths": []})
        elif eligible_before:
            event.update({"status": "SKIPPED_ALREADY_COMPLETE", "summary_paths": [str(path) for path in eligible_before]})
        elif before and set(before) - set(invalid_before):
            event.update(
                {
                    "status": "RESULT_ROOT_COLLISION",
                    "summary_paths": [str(path) for path in before],
                    "reason": "Existing summaries do not match the frozen agent/task/model/provider/seed row.",
                }
            )
        else:
            _append_event(events_path, {**event, "status": "STARTED", "summary_paths": []})
            _write_campaign_state(
                path=state_path,
                matrix_path=matrix_path,
                selected_run_count=len(rows),
                outcomes=outcomes,
                dry_run=dry_run,
                events_path=events_path,
                current_run_id=row["run_id"],
            )
            log_path = log_dir / f"{row['run_id']}.log"
            try:
                with log_path.open("w", encoding="utf-8") as log_handle:
                    returncode = _run_logged_command(
                        _command_argv(row["command"]), cwd=repo, log_handle=log_handle
                    )
            except KeyboardInterrupt:
                event.update({
                    "status": "ABORTED_OPERATOR_INTERRUPT",
                    "log_path": str(log_path.resolve()),
                    "summary_paths": [],
                    "paper_result_eligible": False,
                    "finished_at": _now(),
                })
                _append_event(events_path, event)
                outcomes.append(event)
                _write_campaign_state(
                    path=state_path,
                    matrix_path=matrix_path,
                    selected_run_count=len(rows),
                    outcomes=outcomes,
                    dry_run=dry_run,
                    events_path=events_path,
                    current_run_id=None,
                )
                raise
            after = _summary_paths(repo, row)
            new_paths = [path for path in after if path not in before]
            matching_after = _matching_summaries(new_paths, row)
            validity = [_persist_summary_validity(path, row) for path in matching_after]
            eligible_after = [path for path, check in zip(matching_after, validity) if check["paper_result_eligible"]]
            if returncode == 0 and eligible_after:
                status = "COMPLETE"
            elif returncode == 0 and matching_after:
                status = "INVALID_RESULT_CONTRACT"
            else:
                status = "FAILED_EXECUTION"
            event.update(
                {
                    "status": status,
                    "exit_code": returncode,
                    "log_path": str(log_path.resolve()),
                    "summary_paths": [str(path) for path in matching_after],
                    "eligible_summary_paths": [str(path) for path in eligible_after],
                    "superseded_invalid_summary_paths": [str(path) for path in invalid_before],
                    "summary_validity": validity,
                }
            )
        event["finished_at"] = _now()
        _append_event(events_path, event)
        outcomes.append(event)
        state = _write_campaign_state(
            path=state_path,
            matrix_path=matrix_path,
            selected_run_count=len(rows),
            outcomes=outcomes,
            dry_run=dry_run,
            events_path=events_path,
            current_run_id=None,
        )
    return state
