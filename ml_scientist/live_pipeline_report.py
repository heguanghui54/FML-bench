"""Generate an honest, self-contained live campaign and pipeline dashboard."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from .reporting import collect_experiment_records


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _graph_counts(graph: dict[str, Any]) -> dict[str, int]:
    nodes = graph.get("nodes") or []
    counts = Counter(str(node.get("node_type", node.get("type", "Unknown"))) for node in nodes)
    return {
        "nodes": len(nodes),
        "edges": len(graph.get("edges") or []),
        "policies": counts.get("Policy", 0),
        "skills": counts.get("Skill", 0),
        "episodes": counts.get("AdaptiveEpisode", counts.get("Episode", 0)),
        "adaptive_decisions": counts.get("AdaptiveDecision", counts.get("Decision", 0)),
    }


def _jsonl_count(path: Path | None) -> int:
    if not path or not path.exists():
        return 0
    try:
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    except OSError:
        return 0


def _campaign_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return events


def _live_log_activity(campaign_dir: Path, run_id: str | None) -> dict[str, Any]:
    if not run_id:
        return {"status": "IDLE"}
    path = campaign_dir / f"{run_id}.log"
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {"status": "LOG_NOT_YET_AVAILABLE", "log_path": str(path)}
    ideas = re.findall(r"Generating idea\s+(\d+)/(\d+)", text)
    iterations = re.findall(r"Iteration\s+(\d+)/(\d+)", text)
    token_calls = [
        (int(a.replace(",", "")), int(b.replace(",", "")))
        for a, b in re.findall(r"Tokens:\s+([\d,]+)\s+sent,\s+([\d,]+)\s+received", text)
    ]
    current_idea, total_ideas = ideas[-1] if ideas else (None, None)
    current_iteration, total_iterations = iterations[-1] if iterations else (None, None)
    return {
        "status": "ACTIVE",
        "log_path": str(path),
        "log_bytes": path.stat().st_size,
        "current_idea": int(current_idea) if current_idea else None,
        "total_ideas": int(total_ideas) if total_ideas else None,
        "current_iteration": int(current_iteration) if current_iteration else None,
        "total_iterations": int(total_iterations) if total_iterations else None,
        "observed_llm_call_count": len(token_calls),
        "observed_prompt_tokens": sum(row[0] for row in token_calls),
        "observed_completion_tokens": sum(row[1] for row in token_calls),
        "token_note": "live log sum; final summary.json remains authoritative",
    }


def _current_validation_preview(
    results_root: Path,
    run_id: str | None,
    catalog: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the newest current-run validation artifact, never a test claim."""
    if not run_id:
        return None
    agent_ids = [str(row.get("agent_id")) for row in catalog.get("agents", [])]
    task_ids = [str(row.get("task_id")) for row in catalog.get("tasks", [])]
    agent_id = next((item for item in agent_ids if f"-{item}-" in run_id), None)
    task_id = next((item for item in task_ids if f"-{item}-" in run_id), None)
    if not agent_id or not task_id:
        return None
    candidates = list(results_root.glob(f"**/{agent_id}/{task_id}/**/val_info.json"))
    if not candidates:
        return None
    path = max(candidates, key=lambda item: item.stat().st_mtime)
    payload = _read_json(path, {})
    dataset_id = next(iter(payload), None)
    means = payload.get(dataset_id, {}).get("means", {}) if dataset_id else {}
    if not isinstance(means, dict) or not means:
        return None
    return {
        "status": "PROVISIONAL_VALIDATION_ONLY",
        "agent": agent_id,
        "task": task_id,
        "dataset": dataset_id,
        "metrics": means,
        "artifact_path": str(path),
        "note": "Visible validation preview; not a protected-test result and not final comparison evidence.",
    }


def _posthoc_validation_diagnostic(summary_path: str | None) -> dict[str, Any] | None:
    if not summary_path:
        return None
    path = Path(summary_path).parent / "posthoc_policy_assessment.json"
    payload = _read_json(path, {})
    assessment = payload.get("assessment") or {}
    if assessment.get("review_status") != "VALIDATED_POSTHOC_ARTIFACT_REASSESSMENT":
        return None
    evidence = assessment.get("evidence_artifacts") or []
    artifact = next(
        (row for row in evidence if row.get("artifact_type") in {"validation_result", "validation_failure"}),
        None,
    )
    if not artifact:
        return None
    return {
        "outcome": assessment.get("outcome"),
        "artifact_type": artifact.get("artifact_type"),
        "metrics": (artifact.get("content_summary") or {}).get("observed")
        or artifact.get("content_summary"),
        "artifact_path": artifact.get("path"),
    }


def _execution_contract_rollup(records: list[dict[str, Any]]) -> dict[str, Any]:
    summaries = []
    for record in records:
        path = Path(str(record.get("summary_path") or ""))
        if path.is_file():
            summaries.append(_read_json(path, {}))
    activations = [row for summary in summaries for row in summary.get("candidate_activation", [])]
    repeats = [
        (summary.get("reproducibility") or {}).get("protected_test_comparison")
        for summary in summaries
    ]
    repeats = [row for row in repeats if isinstance(row, dict)]
    telemetry = [row for summary in summaries for row in summary.get("gpu_telemetry", [])]
    return {
        "summary_count": len(summaries),
        "complete_v2_count": sum(summary.get("resource_accounting_status") == "COMPLETE_V2" for summary in summaries),
        "resource_accounting_incomplete_count": sum(summary.get("resource_accounting_status") != "COMPLETE_V2" for summary in summaries),
        "budget_ledgers": [summary.get("budget_ledger") for summary in summaries if summary.get("budget_ledger")],
        "activation_pass_count": sum(row.get("passed") is True for row in activations),
        "activation_invalid_count": sum(row.get("passed") is False for row in activations),
        "repeat_consistent_count": sum(row.get("status") == "CONSISTENT" for row in repeats),
        "repeat_uncertain_count": sum(row.get("status") == "STOCHASTIC_UNCERTAIN" for row in repeats),
        "peak_temperature_c": max((float(row["peak_temperature_c"]) for row in telemetry if row.get("peak_temperature_c") is not None), default=None),
        "peak_memory_mb": max((float(row["peak_memory_mb"]) for row in telemetry if row.get("peak_memory_mb") is not None), default=None),
        "gpu_active_seconds": sum(float(row.get("gpu_active_seconds") or 0) for row in telemetry),
        "safety_terminations": [row.get("safety_termination_reason") for row in telemetry if row.get("safety_termination_reason")],
    }


def _full_dag_rollup(root: Path | None) -> dict[str, Any]:
    runs = list(root.rglob("full_dag_run.json")) if root is not None and root.exists() else []
    if not runs:
        return {"status": "NO_EXECUTABLE_DAG_RUN", "complete": False, "node_counts": {}}
    path = max(runs, key=lambda item: item.stat().st_mtime_ns)
    payload = _read_json(path, {})
    return {
        "status": "COMPLETE" if payload.get("complete") else "INCOMPLETE",
        "complete": bool(payload.get("complete")),
        "executed_node_count": payload.get("executed_node_count", 0),
        "node_counts": dict(Counter(payload.get("node_status", {}).values())),
        "artifact_path": str(path),
        "budget_ledger": payload.get("budget_ledger", {}),
    }


def _novelty_rollup(root: Path | None) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    paths: list[str] = []
    if root is not None and root.exists():
        for path in root.rglob("skill_distillation_report.json"):
            payload = _read_json(path, {})
            for row in payload.get("write_actions", payload.get("actions", [])):
                counts[str(row.get("intent", "NONE"))] += 1
            paths.append(str(path))
    return {
        "decisions": dict(counts),
        "artifact_paths": paths,
        "status": "OBSERVED" if paths else "NO_SKILL_EVOLUTION_RUN",
    }


def _validation_milestone_rollup(root: Path | None) -> dict[str, Any]:
    """Expose valid partial experiments without promoting them to summaries."""
    paths = list(root.rglob("validation_milestone.json")) if root and root.exists() else []
    latest_path = max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None
    latest = _read_json(latest_path, {}) if latest_path else {}
    rows = [_read_json(path, {}) for path in paths]
    return {
        "count": len(rows),
        "paper_claim_eligible_count": sum(row.get("paper_claim_eligible") is True for row in rows),
        "latest_status": latest.get("status"),
        "latest_workspace_label": latest.get("workspace_label"),
        "latest_visible_validation": latest.get("visible_validation"),
        "latest_resources": latest.get("resources"),
        "latest_claim_boundary": latest.get("claim_boundary"),
        "latest_artifact_path": str(latest_path) if latest_path else None,
    }


def _format_diagnostic(diagnostic: dict[str, Any] | None) -> str:
    if not diagnostic:
        return "—"
    metrics = diagnostic.get("metrics") or {}
    preferred = (
        "primary_metric", "auc_gap_mean", "tpr_at_0_1_fpr_gate_value",
        "test_acc_mean", "success",
    )
    parts = [f"{key}={_fmt(metrics[key])}" for key in preferred if key in metrics]
    return f"{diagnostic.get('outcome')}: " + (" · ".join(parts) if parts else "artifact available")


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    head = "".join(f"<th>{html.escape(str(item))}</th>" for item in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(item))}</td>" for item in row) + "</tr>"
        for row in rows
    )
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def write_live_pipeline_report(
    *,
    results_root: Path,
    campaign_dir: Path,
    matrix_path: Path,
    graph_path: Path,
    issue_registry_path: Path,
    out_dir: Path,
    catalog: dict[str, Any],
    new_pipeline_results_root: Path | None = None,
    episode_buffer_path: Path | None = None,
) -> dict[str, Any]:
    """Refresh a live report without claiming an unexecuted adaptive arm."""
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    state_path = campaign_dir / "campaign_state.json"
    events_path = campaign_dir / "campaign_events.jsonl"
    state = _read_json(state_path, {})
    events = _campaign_events(events_path)
    issues_payload = _read_json(issue_registry_path, {"issues": []})
    issues = issues_payload.get("issues") or []
    graph = _read_json(graph_path, {})
    graph_counts = _graph_counts(graph)
    graph_hash = _sha256(graph_path) if graph_path.exists() else None
    buffered_episode_count = _jsonl_count(episode_buffer_path)
    adaptive_episode_path = (
        episode_buffer_path.parent / "adaptive_episodes.jsonl"
        if episode_buffer_path is not None else None
    )
    adaptive_episode_count = _jsonl_count(adaptive_episode_path)

    baseline_records = collect_experiment_records(results_root, catalog)
    adaptive_records = (
        collect_experiment_records(new_pipeline_results_root, catalog)
        if new_pipeline_results_root and new_pipeline_results_root.exists()
        else []
    )
    matrix_rows: list[dict[str, Any]] = []
    try:
        with matrix_path.open(encoding="utf-8", newline="") as handle:
            matrix_rows = list(csv.DictReader(handle))
    except OSError:
        pass
    pilot_rows = [row for row in matrix_rows if row.get("phase") == "pilot"]

    baseline_table = []
    for record in sorted(baseline_records, key=lambda row: (str(row.get("task")), str(row.get("agent")))):
        baseline_table.append(
            {
                "arm": "FML baseline",
                "agent": record.get("agent"),
                "task": record.get("task"),
                "test_metric": record.get("test_metric"),
                "fml_credit_metric": record.get("fml_credit_metric"),
                "credit_status": record.get("fml_credit_status"),
                "normalized_improvement": record.get("normalized_improvement"),
                "tokens": record.get("total_tokens"),
                "duration_seconds": record.get("total_duration_seconds"),
                "resource_accounting_status": record.get("resource_accounting_status"),
                "resource_matched_comparable": record.get("resource_matched_comparable"),
                "candidate_validation_count": record.get("candidate_validation_count"),
                "pre_test_validation_count": record.get("pre_test_validation_count"),
                "protected_test_count": record.get("protected_test_count"),
                "gpu_active_seconds": record.get("gpu_active_seconds"),
                "summary_path": record.get("summary_path"),
            }
        )
    for record in adaptive_records:
        diagnostic = _posthoc_validation_diagnostic(record.get("summary_path"))
        baseline_table.append(
            {
                "arm": "new adaptive pipeline",
                "agent": record.get("agent"),
                "task": record.get("task"),
                "test_metric": record.get("test_metric"),
                "fml_credit_metric": record.get("fml_credit_metric"),
                "credit_status": record.get("fml_credit_status"),
                "normalized_improvement": record.get("normalized_improvement"),
                "tokens": record.get("total_tokens"),
                "duration_seconds": record.get("total_duration_seconds"),
                "resource_accounting_status": record.get("resource_accounting_status"),
                "resource_matched_comparable": record.get("resource_matched_comparable"),
                "candidate_validation_count": record.get("candidate_validation_count"),
                "pre_test_validation_count": record.get("pre_test_validation_count"),
                "protected_test_count": record.get("protected_test_count"),
                "gpu_active_seconds": record.get("gpu_active_seconds"),
                "summary_path": record.get("summary_path"),
                "diagnostic_validation": diagnostic,
            }
        )

    completed_run_ids = {
        event.get("run_id")
        for event in events
        if event.get("status") in {"COMPLETE", "SKIPPED_ALREADY_COMPLETE"}
    }
    progress_done = int(state.get("processed_run_count") or len(completed_run_ids))
    progress_total = int(state.get("selected_run_count") or len(pilot_rows))
    measured_n = sum(row["credit_status"] == "measured" for row in baseline_table if row["arm"] == "FML baseline")
    fallback_n = sum(row["credit_status"] == "baseline_fallback" for row in baseline_table if row["arm"] == "FML baseline")
    executor_path = Path(__file__).resolve().parents[1] / "agents" / "adaptive_pipeline" / "agent.py"
    registry_path = Path(__file__).resolve().parents[1] / "agents" / "registry.py"
    registered_post_gate = (
        registry_path.exists()
        and "AgentType.ADAPTIVE_PIPELINE" in registry_path.read_text(encoding="utf-8")
    )
    adaptive_status = (
        "HAS_REAL_RESULTS" if adaptive_records else
        "RUNNING_OR_READY_POST_BASELINE_GATE" if registered_post_gate and bool(state.get("complete")) else
        "NOT_RUN_EXECUTOR_READY_AWAITING_FROZEN_BASELINE_GATE" if executor_path.exists() else
        "NOT_RUN_EXECUTOR_BRIDGE_PENDING"
    )
    live_activity = _live_log_activity(campaign_dir, state.get("current_run_id"))
    validation_preview = _current_validation_preview(
        results_root, state.get("current_run_id"), catalog,
    )
    execution_contracts = _execution_contract_rollup([*baseline_records, *adaptive_records])
    dag_progress = _full_dag_rollup(new_pipeline_results_root)
    novelty = _novelty_rollup(new_pipeline_results_root)
    validation_milestones = _validation_milestone_rollup(new_pipeline_results_root)

    report = {
        "schema_version": "fml-adaptive-live-dashboard-v2",
        "generated_at": generated_at,
        "campaign": {
            "processed": progress_done,
            "selected": progress_total,
            "remaining": state.get("remaining_run_count", max(progress_total - progress_done, 0)),
            "current_run_id": state.get("current_run_id"),
            "complete": bool(state.get("complete")),
            "matrix_sha256": state.get("matrix_sha256"),
            "live_activity": live_activity,
            "validation_preview": validation_preview,
        },
        "baseline": {
            "record_count": len(baseline_records),
            "measured_count": measured_n,
            "fallback_count": fallback_n,
        },
        "new_pipeline": {
            "status": adaptive_status,
            "record_count": len(adaptive_records),
            "executor_adapter_present": executor_path.exists(),
            "executor_adapter_path": str(executor_path),
            "registered_in_frozen_harness": False,
            "registered_post_baseline_gate": registered_post_gate,
            "registration_timing": "POST_BASELINE_GATE" if registered_post_gate else "NOT_REGISTERED",
            "claim_boundary": (
                "Real matched summaries permit descriptive single-trial comparison only; no superiority or statistical claim is allowed."
                if adaptive_records else
                "No baseline-comparison claim is allowed until a separately labeled, matched FML executor arm produces real summaries."
            ),
        },
        "memory": {
            **graph_counts,
            "buffered_observational_episodes": buffered_episode_count,
            "imported_adaptive_episodes": adaptive_episode_count,
            "graph_sha256": graph_hash,
        },
        "execution_contracts": execution_contracts,
        "dag_progress": dag_progress,
        "skill_novelty": novelty,
        "validation_milestones": validation_milestones,
        "issues": issues,
        "records": baseline_table,
    }

    comparison_fields = [
        "arm", "agent", "task", "test_metric", "fml_credit_metric", "credit_status",
        "normalized_improvement", "tokens", "duration_seconds", "summary_path", "diagnostic_validation",
        "resource_accounting_status", "resource_matched_comparable", "candidate_validation_count",
        "pre_test_validation_count", "protected_test_count", "gpu_active_seconds",
    ]
    _write_csv(out_dir / "tables" / "baseline_comparison.csv", baseline_table, comparison_fields)
    issue_fields = ["id", "scope", "status", "severity", "problem", "evidence", "recommended_change"]
    _write_csv(out_dir / "tables" / "pipeline_issue_registry.csv", issues, issue_fields)
    _write_json(out_dir / "live-dashboard.json", report)

    adaptive_limit = (
        "- The adaptive arm has real summaries, but they are single-trial diagnostics and do not establish superiority."
        if adaptive_records else
        "- The new adaptive pipeline has no performance result until its separate executor arm writes real FML summaries."
    )
    limitations = f"""# Live report limitations

- Pilot results are single-trial diagnostics, not confirmatory evidence.
- FML baseline fallback values are displayed as fallback, never as measured test results.
- Agent calls are step-matched by the frozen matrix but not token- or wall-clock-matched.
{adaptive_limit}
- Memory/skill counts describe control-plane readiness, not model-quality improvement.
"""
    (out_dir / "limitations.md").write_text(limitations, encoding="utf-8")

    issue_rows = [
        [item.get("id"), item.get("scope"), item.get("severity"), item.get("status"), item.get("problem"), item.get("recommended_change")]
        for item in issues
    ]
    result_rows = [
        [
            row["arm"], row["agent"], row["task"], _fmt(row["test_metric"]),
            _fmt(row["fml_credit_metric"]), row["credit_status"],
            _fmt(row["normalized_improvement"]), _fmt(row["tokens"], 0), _fmt(row["duration_seconds"], 1),
            _format_diagnostic(row.get("diagnostic_validation")),
        ]
        for row in baseline_table
    ]
    progress_pct = (100 * progress_done / progress_total) if progress_total else 0
    live_progress = "—"
    if live_activity.get("current_idea"):
        live_progress = f"idea {live_activity['current_idea']}/{live_activity['total_ideas']} · iteration {live_activity.get('current_iteration') or '—'}/{live_activity.get('total_iterations') or '—'}"
    preview_metrics = (validation_preview or {}).get("metrics") or {}
    preview_text = (
        " · ".join(f"{key}={_fmt(value)}" for key, value in preview_metrics.items())
        if preview_metrics else "尚无已完成 validation artifact"
    )
    adaptive_text = (
        f"已有 {len(adaptive_records)} 条真实结果；仍需检查任务、步数、模型和资源是否与基线匹配。"
        if adaptive_records
        else (
            "冻结基线 gate 已完成，adaptive_pipeline 已在门后注册并处于运行或待运行状态；"
            "当前仍无可计入排名的完整 summary。"
            if registered_post_gate and bool(state.get("complete"))
            else
            "尚未运行。独立 adaptive_pipeline 执行器已经实现，但为避免改变正在运行的冻结基线 harness，"
            "将在 14/14 基线 gate 完成后才注册并启动。"
            if executor_path.exists()
            else "尚未运行。当前完成的是控制平面与基线观测；真正的 Stage 4 FML 执行器桥接仍待接通。"
        )
    )
    milestone_text = (
        f"{validation_milestones['latest_status']} · "
        f"{validation_milestones['latest_claim_boundary']}"
        if validation_milestones["count"] else "尚无部分验证里程碑"
    )
    dashboard_html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>自进化 ML Scientist 实时实验面板</title>
<style>
:root{{--bg:#07111f;--panel:#101d30;--line:#263b57;--text:#ecf4ff;--muted:#9eb1c9;--cyan:#54d2e8;--green:#65d49a;--amber:#f4bd62;--red:#ff7f87}}
*{{box-sizing:border-box}} body{{margin:0;background:linear-gradient(145deg,#07111f,#0b1830);color:var(--text);font:15px/1.55 system-ui,-apple-system,sans-serif}}
main{{max-width:1440px;margin:auto;padding:32px}} h1{{font-size:clamp(28px,4vw,48px);margin:.1em 0}} h2{{margin:34px 0 12px}} .muted{{color:var(--muted)}}
.hero,.card{{background:rgba(16,29,48,.94);border:1px solid var(--line);border-radius:16px;padding:22px;box-shadow:0 16px 45px #02081466}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:18px 0}} .kpi{{font-size:28px;font-weight:750;color:var(--cyan)}}
.bar{{height:12px;background:#06101d;border-radius:10px;overflow:hidden}} .bar>i{{display:block;height:100%;width:{progress_pct:.2f}%;background:linear-gradient(90deg,var(--cyan),var(--green))}}
.status{{display:inline-block;padding:4px 10px;border-radius:999px;background:#162a44;border:1px solid #355275}} .warn{{color:var(--amber)}} .bad{{color:var(--red)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:13px}} table{{width:100%;border-collapse:collapse;min-width:850px;background:#0d192a}} th,td{{padding:10px 12px;border-bottom:1px solid #20344d;text-align:left;vertical-align:top}} th{{position:sticky;top:0;background:#13243a;color:#bdeaf1}}
code{{color:#bdeaf1}} footer{{margin:40px 0;color:var(--muted)}}
</style></head><body><main>
<section class="hero"><div class="status">LIVE / PROVISIONAL</div><h1>自进化 ML Scientist 实时实验面板</h1>
<p class="muted">最后刷新：{html.escape(generated_at)} · 当前活动：<code>{html.escape(str(state.get('current_run_id') or '无'))}</code></p>
<div class="bar"><i></i></div><p>{progress_done} / {progress_total} 个冻结 pilot 单元已处理（{progress_pct:.1f}%）</p></section>
<div class="grid">
<section class="card"><div class="muted">基线 summary</div><div class="kpi">{len(baseline_records)}</div><div>{measured_n} measured / {fallback_n} fallback</div></section>
<section class="card"><div class="muted">新管线真实结果</div><div class="kpi">{len(adaptive_records)}</div><div class="warn">{html.escape(adaptive_status)}</div></section>
<section class="card"><div class="muted">部分验证里程碑</div><div class="kpi">{validation_milestones['count']}</div><div class="warn">{html.escape(str(validation_milestones['latest_status'] or 'NONE'))}</div></section>
<section class="card"><div class="muted">当前代理内部进度</div><div class="kpi" style="font-size:20px">{html.escape(live_progress)}</div><div>{_fmt(live_activity.get('observed_llm_call_count'), 0)} LLM calls · {_fmt(live_activity.get('observed_prompt_tokens'), 0)} observed prompt tokens</div></section>
<section class="card"><div class="muted">临时 validation 预览</div><div style="font-size:15px;color:var(--cyan)">{html.escape(preview_text)}</div><div class="warn">不是 protected test，不计入最终对比</div></section>
<section class="card"><div class="muted">知识图谱</div><div class="kpi">{graph_counts['nodes']} / {graph_counts['edges']}</div><div>nodes / edges</div></section>
<section class="card"><div class="muted">可选策略单元</div><div class="kpi">{graph_counts['policies']}</div><div>{graph_counts['skills']} promoted skills · {buffered_episode_count} baseline observations · {adaptive_episode_count} adaptive episodes</div></section>
<section class="card"><div class="muted">资源账本 v2</div><div class="kpi">{execution_contracts['complete_v2_count']}</div><div>{execution_contracts['resource_accounting_incomplete_count']} legacy/incomplete · GPU active {_fmt(execution_contracts['gpu_active_seconds'], 1)}s</div></section>
<section class="card"><div class="muted">运行时归因 / 重复一致性</div><div class="kpi">{execution_contracts['activation_pass_count']} / {execution_contracts['repeat_consistent_count']}</div><div>{execution_contracts['activation_invalid_count']} invalid activation · {execution_contracts['repeat_uncertain_count']} stochastic uncertain</div></section>
<section class="card"><div class="muted">GPU 安全遥测</div><div class="kpi">{_fmt(execution_contracts['peak_temperature_c'], 1)}°C</div><div>peak memory {_fmt(execution_contracts['peak_memory_mb'], 0)} MB · {len(execution_contracts['safety_terminations'])} safety terminations</div></section>
<section class="card"><div class="muted">29-node DAG</div><div class="kpi">{dag_progress.get('executed_node_count', 0)}</div><div>{html.escape(dag_progress['status'])} · {html.escape(str(dag_progress.get('node_counts', {})))}</div></section>
<section class="card"><div class="muted">Skill novelty</div><div class="kpi">{sum(novelty['decisions'].values())}</div><div>{html.escape(str(novelty['decisions'] or {'status': novelty['status']}))}</div></section>
</div>
<h2>当前结论边界</h2><section class="card"><p>{html.escape(adaptive_text)}</p><p class="bad">不能把 AdaptiveSearch 或其他 FML 基线当作新管线结果；也不能用单次 pilot 宣称优越性。</p></section>
<h2>最近阶段验证</h2><section class="card"><p>{html.escape(milestone_text)}</p><p class="muted"><code>{html.escape(str(validation_milestones['latest_artifact_path'] or 'none'))}</code></p></section>
<h2>基线与新管线对比数据</h2>{_html_table(['实验臂','Agent','Task','原始 test','FML credit','credit 类型','归一化提升','tokens','秒','诊断 validation'], result_rows)}
<h2>运行中发现的管线问题与改进</h2>{_html_table(['ID','范围','严重度','状态','问题','改进'], issue_rows)}
<h2>内存与技能进化状态</h2><section class="card"><p>图谱 SHA-256：<code>{html.escape(str(graph_hash or 'missing'))}</code></p>
<p>当前图谱中有 {graph_counts['policies']} 个原子 Policy、{graph_counts['skills']} 个已晋升 Skill、{graph_counts['episodes']} 个 Episode；轨迹缓冲区包含 {buffered_episode_count} 个 bundle-only 基线观察和 {adaptive_episode_count} 个显式原子策略 adaptive episode。基线观察不会伪造原子 Policy credit。Skill 为 0 表示尚无满足 held-out 证据门槛的 CREATE/PATCH 候选。</p></section>
<footer>数据文件：live-dashboard.json · tables/baseline_comparison.csv · tables/pipeline_issue_registry.csv · limitations.md</footer>
</main></body></html>"""
    (out_dir / "live-dashboard.html").write_text(dashboard_html, encoding="utf-8")

    materialized = [
        out_dir / "live-dashboard.html", out_dir / "live-dashboard.json",
        out_dir / "tables" / "baseline_comparison.csv",
        out_dir / "tables" / "pipeline_issue_registry.csv", out_dir / "limitations.md",
    ]
    manifest = {
        "schema_version": "fml-adaptive-live-dashboard-manifest-v1",
        "generated_at": generated_at,
        "files": [{"path": str(path.relative_to(out_dir)), "sha256": _sha256(path)} for path in materialized],
        "inputs": {
            "results_root": str(results_root), "campaign_dir": str(campaign_dir),
            "matrix_path": str(matrix_path), "graph_path": str(graph_path),
            "issue_registry_path": str(issue_registry_path),
            "new_pipeline_results_root": str(new_pipeline_results_root) if new_pipeline_results_root else None,
            "episode_buffer_path": str(episode_buffer_path) if episode_buffer_path else None,
        },
    }
    _write_json(out_dir / "manifest.json", manifest)
    return report
