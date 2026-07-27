"""Resumable execution of a frozen FML controlled-run matrix."""

from __future__ import annotations

import csv
import hashlib
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CampaignError(ValueError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_run_matrix(path: Path) -> list[dict[str, str]]:
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


def _summary_paths(repo: Path, row: dict[str, str]) -> list[Path]:
    base = repo / row["result_root"] / row["agent"] / row["task"]
    return sorted(base.rglob("summary.json")) if base.exists() else []


def _matching_summaries(paths: list[Path], row: dict[str, str]) -> list[Path]:
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
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
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
        if matching_before:
            event.update({"status": "SKIPPED_ALREADY_COMPLETE", "summary_paths": [str(path) for path in matching_before]})
        elif before:
            event.update(
                {
                    "status": "RESULT_ROOT_COLLISION",
                    "summary_paths": [str(path) for path in before],
                    "reason": "Existing summaries do not match the frozen agent/task/model/provider/seed row.",
                }
            )
        elif dry_run:
            event.update({"status": "DRY_RUN_READY", "summary_paths": []})
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
            with log_path.open("w", encoding="utf-8") as log_handle:
                completed = subprocess.run(
                    _command_argv(row["command"]),
                    cwd=repo,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            after = _summary_paths(repo, row)
            matching_after = _matching_summaries(after, row)
            status = "COMPLETE" if completed.returncode == 0 and matching_after else "FAILED_EXECUTION"
            event.update(
                {
                    "status": status,
                    "exit_code": completed.returncode,
                    "log_path": str(log_path.resolve()),
                    "summary_paths": [str(path) for path in matching_after],
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
