"""Paper-ready tables and dependency-free vector figures.

The inventory figures are descriptive repository facts.  Experimental figures
are emitted only when real summary.json records exist; empty result sets never
produce synthetic bars or placeholder statistics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from .published_prior import task_card_rows
from .execution_contracts import normalize_summary_contract


OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _escape(value: Any) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _save_svg(path: Path, body: str, width: int, height: int, title: str, desc: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{_escape(title)}</title><desc>{_escape(desc)}</desc>\n"
        '<rect width="100%" height="100%" fill="white"/>\n'
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#222}.axis{stroke:#222;stroke-width:1}.grid{stroke:#ddd;stroke-width:1}.label{font-size:12px}.small{font-size:10px}.title{font-size:17px;font-weight:bold}</style>\n'
        f"{body}\n</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")


def _domain_bar_chart(catalog: dict[str, Any], path: Path) -> None:
    counts = Counter(task["domain"] for task in catalog["tasks"])
    rows = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    width, height = 900, 100 + 38 * len(rows)
    left, top, plot_width = 230, 58, 600
    maximum = max(counts.values())
    body = ['<text x="24" y="30" class="title">FML-bench task coverage by research domain</text>']
    for index, (label, count) in enumerate(rows):
        y = top + index * 38
        bar = plot_width * count / maximum
        body.append(f'<text x="{left-10}" y="{y+16}" text-anchor="end" class="label">{_escape(label)}</text>')
        body.append(f'<rect x="{left}" y="{y}" width="{bar:.1f}" height="22" fill="{OKABE_ITO[index % len(OKABE_ITO)]}"/>')
        body.append(f'<text x="{left+bar+8:.1f}" y="{y+16}" class="label">{count}</text>')
    body.append(f'<text x="{left}" y="{height-18}" class="small">n = {len(catalog["tasks"])} tasks; bars show task counts, not performance.</text>')
    _save_svg(path, "\n".join(body), width, height, "FML-bench domain coverage", "Count of configured tasks per ML research domain.")


def _agent_strategy_matrix(catalog: dict[str, Any], path: Path) -> None:
    families = [
        "greedy_hill_climbing",
        "parallel_multi_idea",
        "greedy_solution_tree",
        "best_first_tree_search",
        "monte_carlo_tree_search",
        "map_elites_evolution",
        "regime_adaptive_search",
    ]
    labels = ["Greedy", "Multi-idea", "Solution tree", "BFTS", "MCTS", "Evolution", "Adaptive"]
    agents = catalog["agents"]
    width, height = 1040, 110 + 48 * len(agents)
    left, top, cell_w, cell_h = 250, 65, 105, 38
    body = ['<text x="24" y="30" class="title">Baseline-agent search topology</text>']
    for j, label in enumerate(labels):
        x = left + j * cell_w + cell_w / 2
        body.append(f'<text x="{x}" y="{top-12}" text-anchor="middle" class="small">{_escape(label)}</text>')
    for i, agent in enumerate(agents):
        y = top + i * 48
        body.append(f'<text x="{left-10}" y="{y+25}" text-anchor="end" class="label">{_escape(agent["display_name"])}</text>')
        for j, family in enumerate(families):
            active = agent["strategy_family"] == family
            fill = OKABE_ITO[j % len(OKABE_ITO)] if active else "#F1F1F1"
            body.append(f'<rect x="{left+j*cell_w}" y="{y}" width="{cell_w-6}" height="{cell_h}" rx="3" fill="{fill}" stroke="#ccc"/>')
            if active:
                body.append(f'<text x="{left+j*cell_w+(cell_w-6)/2}" y="{y+25}" text-anchor="middle" fill="white" class="label">●</text>')
    body.append(f'<text x="{left}" y="{height-18}" class="small">n = {len(agents)} repository implementations; one primary topology per agent.</text>')
    _save_svg(path, "\n".join(body), width, height, "Baseline-agent topology matrix", "Audited primary search strategy for each registered FML-bench agent.")


def _pipeline_architecture(path: Path) -> None:
    width, height = 1560, 760
    body = [
        '<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8 Z" fill="#555"/></marker></defs>',
        '<text x="24" y="30" class="title">Evidence-gated ML research-to-paper pipeline</text>',
        '<text x="24" y="52" class="small">Every node is planned at Stage 3; execution and review occur through typed inner loops.</text>',
    ]
    stage_names = ["0 Memory", "1 Contract", "2 Skills", "3 Plan DAG", "4 Execute", "5 Review", "6 Repair", "7 Freeze/evolve"]
    stage_x = []
    for index, name in enumerate(stage_names):
        x = 24 + index * 190
        stage_x.append(x)
        fill = OKABE_ITO[index % len(OKABE_ITO)]
        body.append(f'<rect x="{x}" y="74" width="162" height="48" rx="7" fill="{fill}" opacity="0.90"/>')
        body.append(f'<text x="{x+81}" y="103" text-anchor="middle" fill="white" class="label">{_escape(name)}</text>')
        if index:
            body.append(f'<line x1="{x-28}" y1="98" x2="{x-4}" y2="98" stroke="#555" marker-end="url(#arrow)"/>')
    body.append('<path d="M 1105 128 C 1105 165, 915 165, 915 128" fill="none" stroke="#D55E00" stroke-width="2" marker-end="url(#arrow)"/>')
    body.append('<text x="1010" y="158" text-anchor="middle" class="small">node-local revise/debug/backtrack loop</text>')

    def box(x: int, y: int, w: int, text_value: str, fill: str, locked: bool = False) -> None:
        body.append(f'<rect x="{x}" y="{y}" width="{w}" height="54" rx="6" fill="{fill}" opacity="0.93" stroke="#444"/>')
        body.append(f'<text x="{x+w/2}" y="{y+24}" text-anchor="middle" fill="white" class="label">{_escape(text_value)}</text>')
        if locked:
            body.append(f'<text x="{x+w/2}" y="{y+42}" text-anchor="middle" fill="white" class="small">LOCKED until dependencies pass</text>')

    def arrow(x1: int, y1: int, x2: int, y2: int, dashed: bool = False) -> None:
        dash = ' stroke-dasharray="6,5"' if dashed else ""
        body.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#555" stroke-width="1.5"{dash} marker-end="url(#arrow)"/>')

    body.append('<text x="24" y="205" class="label">Research and evidence lane</text>')
    research = [
        (24, "Baseline + hypotheses"),
        (242, "Code candidates"),
        (460, "Visible validation"),
        (678, "Replication + ablation"),
        (896, "Process metrics"),
        (1114, "Evidence freeze"),
        (1332, "Protected test + stats"),
    ]
    for index, (x, label) in enumerate(research):
        box(x, 222, 188, label, OKABE_ITO[index % len(OKABE_ITO)])
        if index:
            arrow(x - 28, 249, x - 5, 249)
    body.append('<text x="1426" y="293" text-anchor="middle" class="small">matched analysis; never search feedback</text>')

    body.append('<text x="24" y="350" class="label">Paper and review lane</text>')
    writing = [
        (24, "Outline + methods", False),
        (242, "Final writing", True),
        (460, "Pre-review integrity", True),
        (678, "Full peer review", True),
        (896, "Revision + re-review", True),
        (1114, "Final integrity", True),
        (1332, "Finalize + process", True),
    ]
    for index, (x, label, locked) in enumerate(writing):
        box(x, 367, 188, label, OKABE_ITO[(index + 1) % len(OKABE_ITO)], locked)
        if index:
            arrow(x - 28, 394, x - 5, 394)
    arrow(1426, 281, 336, 360, dashed=True)
    body.append('<text x="880" y="322" text-anchor="middle" class="small">frozen evidence unlocks result prose</text>')

    body.append('<rect x="340" y="500" width="860" height="154" rx="10" fill="#F7F7F7" stroke="#777" stroke-dasharray="7,5"/>')
    body.append('<text x="770" y="526" text-anchor="middle" class="label">When writing or review finds an empirical error</text>')
    box(385, 548, 205, "Trace claim to evidence", "#0072B2")
    box(665, 548, 205, "Versioned amendment", "#E69F00")
    box(945, 548, 205, "Code / experiment rerun", "#009E73")
    arrow(590, 575, 660, 575)
    arrow(870, 575, 940, 575)
    body.append('<text x="770" y="630" text-anchor="middle" class="small">If the protected test was exposed, empirical modification requires a new hidden evaluation.</text>')
    body.append('<text x="24" y="706" class="small">Four separately governed skill registries: research execution, research review, paper writing, and paper review. Skill updates are evaluated only after the run snapshot closes.</text>')
    _save_svg(path, "\n".join(body), width, height, "Evidence-gated ML research-to-paper pipeline", "Outer stages, research evidence lane, publication review lane, and versioned upstream amendment path.")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_transform(value: Any, task_id: str) -> float | None:
    if value is None:
        return None
    try:
        transformed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(transformed):
        return None
    if task_id == "Unlearning_open_unlearning":
        if transformed <= 0:
            return None
        return -math.log10(transformed)
    return transformed


def _normalized_improvement(test_metric: Any, task: dict[str, Any]) -> float | None:
    agent = _display_transform(test_metric, task["task_id"])
    baseline = _display_transform(task["baseline_test"], task["task_id"])
    best = float(task["normalization_best"])
    if agent is None or baseline is None:
        return None
    worst_spec = task["normalization_worst"]
    worst = baseline if worst_spec == "baseline" else float(worst_spec)
    denominator = abs(best - worst)
    if denominator == 0:
        return None
    if task["report_direction"] == "higher":
        signed = (agent - baseline) / denominator
    else:
        signed = (baseline - agent) / denominator
    return max(signed, 0.0)


def _path_metadata(summary_path: Path, results_root: Path) -> tuple[str | None, int | None]:
    try:
        parts = summary_path.relative_to(results_root).parts
    except ValueError:
        parts = summary_path.parts
    phases = {"pilot", "confirmatory_lite", "confirmatory_full"}
    phase = next((part for part in parts if part in phases), None)
    trial = None
    for part in parts:
        if part.startswith("trial_"):
            try:
                trial = int(part.removeprefix("trial_"))
            except ValueError:
                pass
    return phase, trial


def collect_experiment_records(
    results_root: Path, catalog: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    records = []
    task_index = {task["task_id"]: task for task in (catalog or {}).get("tasks", [])}
    if not results_root.exists():
        return records
    for summary_path in sorted(results_root.rglob("summary.json")):
        validity_path = summary_path.parent / "campaign_validity.json"
        if validity_path.is_file():
            try:
                validity = json.loads(validity_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if validity.get("paper_result_eligible") is not True:
                continue
        try:
            summary = normalize_summary_contract(json.loads(summary_path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
        test_result = summary.get("test_result") or {}
        task_id = summary.get("benchmark")
        task = task_index.get(task_id)
        test_success = test_result.get("success") is True
        all_val_failed = summary.get("best_val_metric") is None
        # Match FML scoring credit: failed protected tests and runs without a
        # valid validation candidate receive the static task baseline.
        credited_metric = test_result.get("primary_metric")
        credit_status = "measured"
        if task and (not test_success or all_val_failed or credited_metric is None):
            credited_metric = task["baseline_test"]
            credit_status = "baseline_fallback"
        phase, trial = _path_metadata(summary_path, results_root)
        params = summary.get("agent_params") or {}
        experimental_seed = summary.get("experimental_seed")
        if experimental_seed is None:
            experimental_seed = params.get("seed", params.get("random_seed"))
        records.append(
            {
                "agent": summary.get("agent"),
                "task": task_id,
                "phase": phase,
                "trial": trial,
                "experimental_seed": experimental_seed,
                "workspace_label": summary.get("workspace_label"),
                "harness_git_commit": summary.get("harness_git_commit"),
                "model": summary.get("model"),
                "provider": summary.get("provider"),
                "baseline_primary_metric": summary.get("baseline_primary_metric"),
                "best_val_metric": summary.get("best_val_metric"),
                "test_metric": test_result.get("primary_metric"),
                "test_success": test_success,
                "fml_credit_metric": credited_metric,
                "fml_credit_status": credit_status,
                "normalized_improvement": _normalized_improvement(credited_metric, task) if task else None,
                "total_steps": summary.get("total_steps"),
                "total_ideas": summary.get("total_ideas"),
                "total_duration_seconds": summary.get("total_duration_seconds"),
                "total_tokens": (summary.get("token_usage") or {}).get("total_tokens"),
                "resource_accounting_status": summary.get("resource_accounting_status"),
                "resource_matched_comparable": summary.get("resource_matched_comparable", False),
                "candidate_validation_count": (summary.get("execution_counts") or {}).get("candidate_validation_count"),
                "pre_test_validation_count": (summary.get("execution_counts") or {}).get("pre_test_validation_count"),
                "protected_test_count": (summary.get("execution_counts") or {}).get("protected_test_count"),
                "gpu_active_seconds": (summary.get("budget_ledger") or {}).get("usage", {}).get("gpu_active_seconds"),
                "summary_path": str(summary_path),
                "summary_sha256": _hash(summary_path),
            }
        )
    return records


_T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _t95(df: int) -> float:
    return _T95.get(df, 1.96 if df > 30 else 12.706)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("phase") or "unassigned"),
            str(record.get("agent")),
            str(record.get("task")),
            str(record.get("model")),
            str(record.get("provider")),
        )
        groups[key].append(record)
    rows = []
    for (phase, agent, task, model, provider), group in sorted(groups.items()):
        values = [float(record["normalized_improvement"]) for record in group if record.get("normalized_improvement") is not None]
        raw_values = [float(record["fml_credit_metric"]) for record in group if record.get("fml_credit_metric") is not None]
        if not values:
            continue
        n = len(values)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if n >= 2 else None
        ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
        rows.append(
            {
                "phase": phase,
                "agent": agent,
                "task": task,
                "model": model,
                "provider": provider,
                "attempted_n": len(group),
                "successful_test_n": sum(record.get("test_success") is True for record in group),
                "n": n,
                "mean_raw_fml_credit": statistics.fmean(raw_values) if raw_values else None,
                "mean_normalized_improvement": mean,
                "sd_normalized_improvement": sd,
                "ci95_halfwidth_student_t": ci,
            }
        )
    return rows


_EXPECTED_TASKS_BY_PHASE = {"pilot": 2, "confirmatory_lite": 8, "confirmatory_full": 18}
_EXPECTED_TRIALS_BY_PHASE = {"pilot": 1, "confirmatory_lite": 3, "confirmatory_full": 3}


def summarize_agent_performance(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    trial_groups: dict[tuple[str, str, str, str, int | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record.get("phase") or "unassigned"),
            str(record.get("agent")),
            str(record.get("model")),
            str(record.get("provider")),
            record.get("trial"),
        )
        trial_groups[key].append(record)
    trial_rows = []
    for (phase, agent, model, provider, trial), group in sorted(trial_groups.items(), key=lambda item: str(item[0])):
        values = [float(record["normalized_improvement"]) for record in group if record.get("normalized_improvement") is not None]
        tasks = {str(record["task"]) for record in group if record.get("normalized_improvement") is not None}
        expected = _EXPECTED_TASKS_BY_PHASE.get(phase, len(tasks))
        complete = len(tasks) == expected and len(values) == expected
        trial_rows.append(
            {
                "phase": phase,
                "agent": agent,
                "model": model,
                "provider": provider,
                "trial": trial,
                "expected_task_n": expected,
                "observed_task_n": len(tasks),
                "complete_task_block": complete,
                "mean_normalized_improvement_across_tasks": statistics.fmean(values) if values else None,
            }
        )
    overall_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in trial_rows:
        value = row["mean_normalized_improvement_across_tasks"]
        if row["complete_task_block"] and value is not None:
            overall_groups[(row["phase"], row["agent"], row["model"], row["provider"])].append(float(value))
    overall_rows = []
    for (phase, agent, model, provider), values in sorted(overall_groups.items()):
        n = len(values)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if n >= 2 else None
        ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
        overall_rows.append(
            {
                "phase": phase,
                "agent": agent,
                "model": model,
                "provider": provider,
                "complete_trial_n": n,
                "mean_normalized_improvement": mean,
                "sd_across_trial_means": sd,
                "ci95_halfwidth_student_t": ci,
            }
        )
    return trial_rows, overall_rows


def _paired_summary(values: list[float]) -> dict[str, Any]:
    n = len(values)
    if not values:
        return {
            "n": 0, "mean": None, "median": None, "sd": None,
            "ci_halfwidth": None, "ci_low": None, "ci_high": None,
            "cohen_dz": None,
        }
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n >= 2 else None
    ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
    return {
        "n": n,
        "mean": mean,
        "median": statistics.median(values),
        "sd": sd,
        "ci_halfwidth": ci,
        "ci_low": mean - ci if ci is not None else None,
        "ci_high": mean + ci if ci is not None else None,
        "cohen_dz": mean / sd if sd is not None and sd > 0 else None,
    }


def _sign_counts(values: list[float], tolerance: float = 1e-12) -> tuple[int, int, int]:
    wins = sum(value > tolerance for value in values)
    losses = sum(value < -tolerance for value in values)
    ties = len(values) - wins - losses
    return wins, ties, losses


def _exact_two_sided_sign_p(wins: int, losses: int) -> float | None:
    """Exact two-sided binomial sign test after excluding ties."""
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    probability = 2.0 * sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, probability)


def _holm_adjust(
    rows: list[dict[str, Any]],
    *,
    p_field: str,
    adjusted_field: str,
    family_size_field: str,
) -> None:
    families: dict[tuple[str, str, str], list[tuple[int, float]]] = defaultdict(list)
    planned_family_sizes = Counter((row["phase"], row["model"], row["provider"]) for row in rows)
    for index, row in enumerate(rows):
        p_value = row.get(p_field)
        if p_value is not None:
            families[(row["phase"], row["model"], row["provider"])].append((index, float(p_value)))
    for family, members in families.items():
        ordered = sorted(members, key=lambda item: item[1])
        # Keep missing/unevaluable preregistered pairwise hypotheses in the
        # multiplicity family; treating them as absent would shrink the family
        # after observing incomplete data.
        family_size = planned_family_sizes[family]
        running_max = 0.0
        for rank, (index, p_value) in enumerate(ordered):
            adjusted = min(1.0, (family_size - rank) * p_value)
            running_max = max(running_max, adjusted)
            rows[index][adjusted_field] = running_max
            rows[index][family_size_field] = family_size
    for row in rows:
        row.setdefault(adjusted_field, None)
        row.setdefault(
            family_size_field,
            planned_family_sizes[(row["phase"], row["model"], row["provider"])],
        )


def _paired_outcomes(records: list[dict[str, Any]]) -> dict[tuple[str, str, str, int | None, str, str], float]:
    grouped: dict[tuple[str, str, str, int | None, str, str], list[float]] = defaultdict(list)
    for record in records:
        if record.get("normalized_improvement") is None:
            continue
        key = (
            str(record.get("phase") or "unassigned"),
            str(record.get("model")),
            str(record.get("provider")),
            record.get("trial"),
            str(record.get("task")),
            str(record.get("agent")),
        )
        grouped[key].append(float(record["normalized_improvement"]))
    # A duplicate run cell is ambiguous, so it must make the paired block
    # incomplete instead of being silently overwritten or averaged.
    return {key: values[0] for key, values in grouped.items() if len(values) == 1}


def audit_experiment_cells(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = ("phase", "trial", "agent", "task", "model", "provider")
    issues: list[dict[str, Any]] = []
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        missing = [field for field in required if record.get(field) is None]
        if missing:
            issues.append(
                {
                    "issue": "missing_cell_metadata",
                    "phase": record.get("phase"),
                    "trial": record.get("trial"),
                    "agent": record.get("agent"),
                    "task": record.get("task"),
                    "model": record.get("model"),
                    "provider": record.get("provider"),
                    "record_count": 1,
                    "detail": "missing fields: " + ",".join(missing),
                    "source_paths": record.get("summary_path"),
                    "record_index": index,
                }
            )
            continue
        grouped[tuple(record[field] for field in required)].append(record)
    for key, members in sorted(grouped.items(), key=lambda item: str(item[0])):
        if len(members) <= 1:
            continue
        issues.append(
            {
                "issue": "duplicate_agent_task_seed_cell",
                "phase": key[0],
                "trial": key[1],
                "agent": key[2],
                "task": key[3],
                "model": key[4],
                "provider": key[5],
                "record_count": len(members),
                "detail": "duplicate cells are excluded from paired analysis, never averaged or overwritten",
                "source_paths": ";".join(str(member.get("summary_path") or "") for member in members),
                "record_index": None,
            }
        )
    return issues


def _pair_blocks(
    values: dict[tuple[str, str, str, int | None, str, str], float],
    context: tuple[str, str, str],
    left: str,
    right: str,
) -> tuple[dict[int | None, dict[str, float]], dict[int | None, dict[str, float]]]:
    phase, model, provider = context
    candidate_trials = {
        key[3]
        for key in values
        if key[:3] == context and key[5] in {left, right}
    }
    matched: dict[int | None, dict[str, float]] = defaultdict(dict)
    for trial in candidate_trials:
        matched[trial] = {}
    for key, left_value in values.items():
        if key[:3] != context or key[5] != left:
            continue
        _, _, _, trial, task, _ = key
        right_key = (phase, model, provider, trial, task, right)
        if right_key in values:
            matched[trial][task] = left_value - values[right_key]
    expected = _EXPECTED_TASKS_BY_PHASE.get(phase)
    complete = {
        trial: differences
        for trial, differences in matched.items()
        if differences and (expected is None or len(differences) == expected)
    }
    return dict(matched), complete


def paired_agent_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = _paired_outcomes(records)
    contexts = sorted({key[:3] for key in values})
    rows = []
    for phase, model, provider in contexts:
        agents = sorted({key[5] for key in values if key[:3] == (phase, model, provider)})
        for left_index, left in enumerate(agents):
            for right in agents[left_index + 1:]:
                matched, complete = _pair_blocks(values, (phase, model, provider), left, right)
                complete_trial_means = [statistics.fmean(differences.values()) for differences in complete.values()]
                primary = _paired_summary(complete_trial_means)
                seed_wins, seed_ties, seed_losses = _sign_counts(complete_trial_means)
                cell_differences = [difference for block in complete.values() for difference in block.values()]
                cell_wins, cell_ties, cell_losses = _sign_counts(cell_differences)
                tasks = sorted({task for block in complete.values() for task in block})
                task_means = [
                    statistics.fmean(block[task] for block in complete.values() if task in block)
                    for task in tasks
                ]
                task_wins, task_ties, task_losses = _sign_counts(task_means)
                rows.append(
                    {
                        "phase": phase,
                        "model": model,
                        "provider": provider,
                        "agent_left": left,
                        "agent_right": right,
                        "analysis_class": "confirmatory" if phase.startswith("confirmatory") else "diagnostic",
                        "expected_trial_n": _EXPECTED_TRIALS_BY_PHASE.get(phase),
                        "expected_task_n_per_trial": _EXPECTED_TASKS_BY_PHASE.get(phase),
                        "observed_matched_trial_n": len(matched),
                        "complete_paired_trial_n": primary["n"],
                        "excluded_incomplete_paired_trial_n": len(matched) - len(complete),
                        "complete_block_gate_passed": (
                            primary["n"] == _EXPECTED_TRIALS_BY_PHASE[phase]
                            if phase in _EXPECTED_TRIALS_BY_PHASE else primary["n"] > 0
                        ),
                        "mean_difference_left_minus_right": primary["mean"],
                        "median_difference_left_minus_right": primary["median"],
                        "sd_across_paired_trial_means": primary["sd"],
                        "ci95_halfwidth_student_t": primary["ci_halfwidth"],
                        "ci95_low": primary["ci_low"],
                        "ci95_high": primary["ci_high"],
                        "cohen_dz": primary["cohen_dz"],
                        "seed_block_wins_left": seed_wins,
                        "seed_block_ties": seed_ties,
                        "seed_block_losses_left": seed_losses,
                        "seed_block_win_rate_excluding_ties": seed_wins / (seed_wins + seed_losses) if seed_wins + seed_losses else None,
                        "exact_sign_p_seed_blocks": _exact_two_sided_sign_p(seed_wins, seed_losses),
                        "task_n": len(tasks),
                        "task_wins_left": task_wins,
                        "task_ties": task_ties,
                        "task_losses_left": task_losses,
                        "task_win_rate_excluding_ties": task_wins / (task_wins + task_losses) if task_wins + task_losses else None,
                        "exact_sign_p_task_means_descriptive": _exact_two_sided_sign_p(task_wins, task_losses),
                        "matched_task_trial_cell_n": len(cell_differences),
                        "cell_wins_left": cell_wins,
                        "cell_ties": cell_ties,
                        "cell_losses_left": cell_losses,
                        "cell_win_rate_excluding_ties": cell_wins / (cell_wins + cell_losses) if cell_wins + cell_losses else None,
                        "interpretation": "positive favors agent_left; seed-block statistics are primary, while task and task-seed win rates are fixed-suite descriptive diagnostics",
                    }
                )
    _holm_adjust(
        rows,
        p_field="exact_sign_p_seed_blocks",
        adjusted_field="holm_p_seed_blocks",
        family_size_field="holm_seed_family_size",
    )
    _holm_adjust(
        rows,
        p_field="exact_sign_p_task_means_descriptive",
        adjusted_field="holm_p_task_means_descriptive",
        family_size_field="holm_task_family_size",
    )
    for row in rows:
        row["holm_seed_significant_0_05"] = (
            row["analysis_class"] == "confirmatory"
            and row["complete_block_gate_passed"]
            and row["holm_p_seed_blocks"] is not None
            and row["holm_p_seed_blocks"] < 0.05
        )
    return rows


def paired_agent_task_effects(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = _paired_outcomes(records)
    contexts = sorted({key[:3] for key in values})
    rows: list[dict[str, Any]] = []
    for phase, model, provider in contexts:
        agents = sorted({key[5] for key in values if key[:3] == (phase, model, provider)})
        for left_index, left in enumerate(agents):
            for right in agents[left_index + 1:]:
                _, complete = _pair_blocks(values, (phase, model, provider), left, right)
                tasks = sorted({task for block in complete.values() for task in block})
                for task in tasks:
                    differences = [block[task] for block in complete.values() if task in block]
                    summary = _paired_summary(differences)
                    wins, ties, losses = _sign_counts(differences)
                    rows.append(
                        {
                            "phase": phase,
                            "model": model,
                            "provider": provider,
                            "agent_left": left,
                            "agent_right": right,
                            "task": task,
                            "matched_seed_n": summary["n"],
                            "mean_difference_left_minus_right": summary["mean"],
                            "median_difference_left_minus_right": summary["median"],
                            "sd_paired_difference": summary["sd"],
                            "ci95_halfwidth_student_t": summary["ci_halfwidth"],
                            "ci95_low": summary["ci_low"],
                            "ci95_high": summary["ci_high"],
                            "cohen_dz": summary["cohen_dz"],
                            "seed_wins_left": wins,
                            "seed_ties": ties,
                            "seed_losses_left": losses,
                            "seed_win_rate_excluding_ties": wins / (wins + losses) if wins + losses else None,
                            "claim_boundary": "secondary fixed-task heterogeneity estimate; not an independent primary comparison",
                        }
                    )
    return rows


def adaptive_opportunity_interactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preregistered AdaptiveSearch contrast across published opportunity strata.

    The dense/sparse labels are frozen from the published post-hoc partition.
    New campaign outcomes test the interaction; published outcomes never enter
    this estimator.
    """
    partition = {row["task_id"]: row["partition"] for row in task_card_rows()}
    values = _paired_outcomes(records)
    contexts = sorted({key[:3] for key in values})
    rows: list[dict[str, Any]] = []
    adaptive = "adaptivesearch"
    for phase, model, provider in contexts:
        agents = sorted({key[5] for key in values if key[:3] == (phase, model, provider)})
        if adaptive not in agents:
            continue
        for baseline in (agent for agent in agents if agent != adaptive):
            matched, complete = _pair_blocks(values, (phase, model, provider), adaptive, baseline)
            interactions: list[float] = []
            dense_task_ids: set[str] = set()
            sparse_task_ids: set[str] = set()
            for block in complete.values():
                dense = [difference for task, difference in block.items() if partition.get(task) == "DENSE-OPP"]
                sparse = [difference for task, difference in block.items() if partition.get(task) == "SPARSE-OPP"]
                dense_task_ids.update(task for task in block if partition.get(task) == "DENSE-OPP")
                sparse_task_ids.update(task for task in block if partition.get(task) == "SPARSE-OPP")
                if dense and sparse:
                    interactions.append(statistics.fmean(dense) - statistics.fmean(sparse))
            summary = _paired_summary(interactions)
            wins, ties, losses = _sign_counts(interactions)
            expected_trials = _EXPECTED_TRIALS_BY_PHASE.get(phase)
            rows.append(
                {
                    "phase": phase,
                    "model": model,
                    "provider": provider,
                    "adaptive_agent": adaptive,
                    "baseline_agent": baseline,
                    "analysis_class": "confirmatory" if phase.startswith("confirmatory") else "diagnostic",
                    "partition_source": "arXiv:2605.17373v2 Table 5 post-hoc opportunity-density median split",
                    "partition_use": "frozen hypothesis stratum; new outcomes only",
                    "dense_task_n": len(dense_task_ids),
                    "sparse_task_n": len(sparse_task_ids),
                    "observed_matched_trial_n": len(matched),
                    "complete_interaction_trial_n": summary["n"],
                    "complete_block_gate_passed": (
                        summary["n"] == expected_trials if expected_trials is not None else summary["n"] > 0
                    ),
                    "mean_dense_minus_sparse_adaptive_advantage": summary["mean"],
                    "median_dense_minus_sparse_adaptive_advantage": summary["median"],
                    "sd_across_trial_interactions": summary["sd"],
                    "ci95_halfwidth_student_t": summary["ci_halfwidth"],
                    "ci95_low": summary["ci_low"],
                    "ci95_high": summary["ci_high"],
                    "cohen_dz": summary["cohen_dz"],
                    "trial_wins_positive_interaction": wins,
                    "trial_ties": ties,
                    "trial_losses_negative_interaction": losses,
                    "exact_sign_p": _exact_two_sided_sign_p(wins, losses),
                    "interpretation": "positive means AdaptiveSearch has a larger average advantage over this baseline on dense than sparse tasks",
                }
            )
    _holm_adjust(
        rows,
        p_field="exact_sign_p",
        adjusted_field="holm_p_across_adaptive_baselines",
        family_size_field="holm_family_size",
    )
    for row in rows:
        row["holm_significant_0_05"] = (
            row["analysis_class"] == "confirmatory"
            and row["complete_block_gate_passed"]
            and row["holm_p_across_adaptive_baselines"] is not None
            and row["holm_p_across_adaptive_baselines"] < 0.05
        )
    return rows


def collect_scorer_semantics_sensitivity(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Audit scorer-semantic differences without replacing official metrics."""
    rows: list[dict[str, Any]] = []
    for record in records:
        summary_path = Path(str(record.get("summary_path") or ""))
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        val_steps = summary.get("val_steps") or []
        best_value = summary.get("best_val_metric")
        successful_ids: set[int] = set()
        matches: list[int] = []
        for fallback_id, step in enumerate(val_steps, start=1):
            step_id = step.get("step_id")
            try:
                step_id = int(step_id)
            except (TypeError, ValueError):
                step_id = fallback_id
            successful = (step.get("val_result") or {}).get("success") is True and step.get("primary_metric") is not None
            if successful:
                successful_ids.add(step_id)
                if best_value is not None and step.get("primary_metric") == best_value:
                    matches.append(step_id)
        snapshot_ids: set[int] = set()
        for snapshot in (summary_path.parent / "step_snapshots").glob("step_*_code.json"):
            match = re.search(r"step_(\d+)_code\.json$", snapshot.name)
            if match:
                snapshot_ids.add(int(match.group(1)))
        invalid_snapshot_ids = snapshot_ids - successful_ids
        missing_success_snapshot_ids = successful_ids - snapshot_ids
        rows.append(
            {
                "phase": record.get("phase"),
                "trial": record.get("trial"),
                "agent": record.get("agent"),
                "task": record.get("task"),
                "summary_path": str(summary_path),
                "summary_sha256": record.get("summary_sha256"),
                "best_val_metric": best_value,
                "official_last_matching_best_step": matches[-1] if matches else None,
                "sensitivity_first_matching_best_step": matches[0] if matches else None,
                "best_step_shift_last_minus_first": matches[-1] - matches[0] if matches else None,
                "best_step_semantics_differ": len(matches) > 1 and matches[-1] != matches[0],
                "validation_step_n": len(val_steps),
                "successful_validation_step_n": len(successful_ids),
                "official_persisted_snapshot_n": len(snapshot_ids),
                "valid_only_persisted_snapshot_n": len(snapshot_ids & successful_ids),
                "invalid_persisted_snapshot_n": len(invalid_snapshot_ids),
                "successful_step_missing_snapshot_n": len(missing_success_snapshot_ids),
                "exploration_membership_differs": bool(invalid_snapshot_ids or missing_success_snapshot_ids),
                "valid_only_exploration_status": (
                    "RECOMPUTE_GRAPHCODEBERT_SENSITIVITY"
                    if invalid_snapshot_ids or missing_success_snapshot_ids
                    else "MEMBERSHIP_EQUIVALENT" if snapshot_ids else "NO_SNAPSHOTS_AVAILABLE"
                ),
                "official_metrics_replaced": False,
            }
        )
    return rows


def _agent_overall_chart(rows: list[dict[str, Any]], path: Path) -> bool:
    rows = [row for row in rows if row["complete_trial_n"] >= 2]
    if not rows:
        return False
    width, height = 980, 110 + 42 * len(rows)
    left, right, top = 315, 70, 64
    plot_width = width - left - right
    upper = max(1.0, max(row["mean_normalized_improvement"] + (row["ci95_halfwidth_student_t"] or 0) for row in rows))
    body = ['<text x="24" y="30" class="title">Overall FML normalized improvement across complete task blocks</text>']
    for tick in range(6):
        value = upper * tick / 5
        x = left + plot_width * tick / 5
        body.append(f'<line x1="{x:.1f}" y1="{top-10}" x2="{x:.1f}" y2="{height-40}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height-20}" text-anchor="middle" class="small">{value:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * 42
        mean, ci = row["mean_normalized_improvement"], row["ci95_halfwidth_student_t"] or 0.0
        x = left + plot_width * mean / upper
        x_low = left + plot_width * max(0.0, mean - ci) / upper
        x_high = left + plot_width * (mean + ci) / upper
        label = f'{row["phase"]} / {row["agent"]}'
        body.append(f'<text x="{left-10}" y="{y+5}" text-anchor="end" class="label">{_escape(label)}</text>')
        body.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="#222" stroke-width="2"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="5" fill="{OKABE_ITO[index % len(OKABE_ITO)]}"/>')
        body.append(f'<text x="{x_high+8:.1f}" y="{y+4}" class="small">trials={row["complete_trial_n"]}</text>')
    body.append(f'<text x="{left}" y="{height-5}" class="small">Each observation is one complete trial mean across all tasks; incomplete task blocks are excluded, not silently averaged.</text>')
    _save_svg(path, "\n".join(body), width, height, "Overall FML normalized improvement", "Agent means and Student-t intervals across complete independent trial blocks.")
    return True


def _replicated_performance_chart(rows: list[dict[str, Any]], path: Path) -> bool:
    rows = [row for row in rows if row["n"] >= 2]
    if not rows:
        return False
    width = 1250
    height = 105 + 31 * len(rows)
    left, right, top = 500, 70, 60
    plot_width = width - left - right
    upper = max(
        1.0,
        max(row["mean_normalized_improvement"] + (row["ci95_halfwidth_student_t"] or 0) for row in rows),
    )
    body = ['<text x="24" y="30" class="title">Replicated FML normalized improvement by agent and task</text>']
    for tick in range(6):
        value = upper * tick / 5
        x = left + plot_width * tick / 5
        body.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-42}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" class="small">{value:.2f}</text>')
    for index, row in enumerate(rows):
        y = top + index * 31
        mean = row["mean_normalized_improvement"]
        ci = row["ci95_halfwidth_student_t"] or 0.0
        low = max(0.0, mean - ci)
        high = mean + ci
        x = left + plot_width * mean / upper
        x_low = left + plot_width * low / upper
        x_high = left + plot_width * high / upper
        label = f'{row["phase"]} / {row["agent"]} / {row["task"]}'
        body.append(f'<text x="{left-10}" y="{y+5}" text-anchor="end" class="small">{_escape(label)}</text>')
        body.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="#222" stroke-width="1.5"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="{OKABE_ITO[index % len(OKABE_ITO)]}"/>')
        body.append(f'<text x="{x_high+7:.1f}" y="{y+4}" class="small">n={row["n"]}</text>')
    body.append(f'<text x="{left}" y="{height-6}" class="small">Points are means; intervals are two-sided 95% Student-t intervals across independent trials.</text>')
    _save_svg(path, "\n".join(body), width, height, "Replicated FML normalized improvement", "Task-level normalized improvement means and Student-t intervals from real result summaries.")
    return True


def _float_or_none(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(parsed) or math.isinf(parsed) else parsed


def collect_process_metric_records(
    metric_reports_root: Path | None,
    experiment_records: list[dict[str, Any]],
    catalog: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if metric_reports_root is None or not metric_reports_root.exists():
        return []
    process_names = {
        metric["name"]
        for metric in (catalog or {}).get("metrics", [])
        if metric.get("category") != "task_performance"
    }
    experiment_index = {
        (
            str(record.get("phase") or "unassigned"),
            record.get("trial"),
            str(record.get("agent")),
            str(record.get("task")),
        ): record
        for record in experiment_records
    }
    records: list[dict[str, Any]] = []
    for path in sorted(metric_reports_root.rglob("table_process_metrics.csv")):
        try:
            relative = path.relative_to(metric_reports_root)
            parts = relative.parts
            phase = next((part for part in parts if part in {"pilot", "confirmatory_lite", "confirmatory_full"}), "unassigned")
            trial = next((int(part.removeprefix("trial_")) for part in parts if part.startswith("trial_")), None)
            agent = path.parent.name
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except (OSError, ValueError, csv.Error):
            continue
        source_hash = _hash(path)
        for row in rows:
            task = row.get("task") or row.get("") or row.get("Task")
            if not task or task == "MEAN":
                continue
            experiment = experiment_index.get((phase, trial, agent, task), {})
            for metric_name in sorted(process_names & set(row)):
                value = _float_or_none(row.get(metric_name))
                if value is None:
                    continue
                records.append(
                    {
                        "phase": phase,
                        "trial": trial,
                        "agent": agent,
                        "task": task,
                        "model": experiment.get("model"),
                        "provider": experiment.get("provider"),
                        "metric": metric_name,
                        "value": value,
                        "source_path": str(path),
                        "source_sha256": source_hash,
                    }
                )
    return records


def summarize_process_metrics(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        groups[(str(record["phase"]), str(record["agent"]), str(record["task"]), str(record["metric"]))].append(float(record["value"]))
    summaries = []
    for (phase, agent, task, metric), values in sorted(groups.items()):
        n = len(values)
        mean = statistics.fmean(values)
        sd = statistics.stdev(values) if n >= 2 else None
        ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
        summaries.append(
            {
                "phase": phase,
                "agent": agent,
                "task": task,
                "metric": metric,
                "n": n,
                "mean": mean,
                "sd": sd,
                "ci95_halfwidth_student_t": ci,
            }
        )
    return summaries


def _safe_slug(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def _process_metric_chart(metric: str, rows: list[dict[str, Any]], path: Path) -> bool:
    rows = [row for row in rows if row["metric"] == metric and row["n"] >= 2]
    if not rows:
        return False
    lows = [row["mean"] - (row["ci95_halfwidth_student_t"] or 0) for row in rows]
    highs = [row["mean"] + (row["ci95_halfwidth_student_t"] or 0) for row in rows]
    minimum, maximum = min(lows), max(highs)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.1, 1.0)
    else:
        padding = (maximum - minimum) * 0.08
    minimum -= padding
    maximum += padding
    width, height = 1250, 105 + 31 * len(rows)
    left, right, top = 500, 75, 60
    plot_width = width - left - right
    body = [f'<text x="24" y="30" class="title">{_escape(metric)} by agent and task</text>']
    for tick in range(6):
        value = minimum + (maximum - minimum) * tick / 5
        x = left + plot_width * tick / 5
        body.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-42}" class="grid"/>')
        body.append(f'<text x="{x:.1f}" y="{height-22}" text-anchor="middle" class="small">{value:.3g}</text>')
    for index, row in enumerate(rows):
        y = top + index * 31
        ci = row["ci95_halfwidth_student_t"] or 0.0
        x = left + plot_width * (row["mean"] - minimum) / (maximum - minimum)
        x_low = left + plot_width * (row["mean"] - ci - minimum) / (maximum - minimum)
        x_high = left + plot_width * (row["mean"] + ci - minimum) / (maximum - minimum)
        label = f'{row["phase"]} / {row["agent"]} / {row["task"]}'
        body.append(f'<text x="{left-10}" y="{y+5}" text-anchor="end" class="small">{_escape(label)}</text>')
        body.append(f'<line x1="{x_low:.1f}" y1="{y}" x2="{x_high:.1f}" y2="{y}" stroke="#222" stroke-width="1.5"/>')
        body.append(f'<circle cx="{x:.1f}" cy="{y}" r="4.5" fill="{OKABE_ITO[index % len(OKABE_ITO)]}"/>')
        body.append(f'<text x="{x_high+7:.1f}" y="{y+4}" class="small">n={row["n"]}</text>')
    body.append(f'<text x="{left}" y="{height-6}" class="small">Means and two-sided 95% Student-t intervals. Interpret direction using the metric definition; this plot does not assert that larger is always better.</text>')
    _save_svg(path, "\n".join(body), width, height, metric, f"Replicated {metric} values from FML process-metric reports.")
    return True


def write_catalog_artifacts(catalog: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = out_dir / "fml_catalog.json"
    _write_json(catalog_path, catalog)
    _write_csv(
        out_dir / "agents.csv",
        catalog["agents"],
        ["agent_id", "display_name", "strategy_family", "state_representation", "proposal_operator", "selection_policy", "debug_policy", "memory_policy", "diversity_mechanism", "paper_boundary", "source", "source_sha256", "config", "config_sha256"],
    )
    _write_csv(
        out_dir / "tasks.csv",
        catalog["tasks"],
        ["task_id", "domain", "lite", "canonical_dataset", "canonical_metric", "canonical_direction", "baseline_validation", "baseline_test", "repository", "pinned_commit", "conda_env", "config_sha256", "prompt_sha256"],
    )
    _write_csv(
        out_dir / "metrics.csv",
        catalog["metrics"],
        ["metric_id", "name", "category", "task_id", "dataset", "direction", "definition", "baseline_validation", "baseline_test", "source"],
    )
    figures = out_dir / "figures"
    _domain_bar_chart(catalog, figures / "fml_task_domain_coverage.svg")
    _agent_strategy_matrix(catalog, figures / "baseline_agent_topology.svg")
    _pipeline_architecture(figures / "unified_research_paper_pipeline.svg")
    manifest = {
        "schema_version": "fml-scientist-figure-manifest-v1",
        "figures": [
            {
                "path": "figures/fml_task_domain_coverage.svg",
                "source_table": "tasks.csv",
                "sample_size": len(catalog["tasks"]),
                "uncertainty": "not applicable; complete configured-task inventory",
                "claim_boundary": "describes suite composition, not task quality or agent performance",
            },
            {
                "path": "figures/baseline_agent_topology.svg",
                "source_table": "agents.csv",
                "sample_size": len(catalog["agents"]),
                "uncertainty": "not applicable; audited implementation taxonomy",
                "claim_boundary": "describes primary topology, not empirical superiority",
            },
            {
                "path": "figures/unified_research_paper_pipeline.svg",
                "source_table": "ml_scientist/planner.py",
                "sample_size": "not applicable; architecture contract",
                "uncertainty": "not applicable; system design",
                "claim_boundary": "describes dependency and amendment semantics, not empirical performance",
            },
        ],
        "style": {"format": "SVG vector", "palette": "Okabe-Ito", "background": "white"},
    }
    _write_json(out_dir / "figure_manifest.json", manifest)
    return manifest


def write_experiment_artifacts(
    results_root: Path,
    out_dir: Path,
    catalog: dict[str, Any] | None = None,
    metric_reports_root: Path | None = None,
) -> dict[str, Any]:
    records = collect_experiment_records(results_root, catalog)
    _write_csv(
        out_dir / "experiment_records.csv",
        records,
        ["agent", "task", "phase", "trial", "experimental_seed", "workspace_label", "harness_git_commit", "model", "provider", "baseline_primary_metric", "best_val_metric", "test_metric", "test_success", "fml_credit_metric", "fml_credit_status", "normalized_improvement", "total_steps", "total_ideas", "total_duration_seconds", "total_tokens", "resource_accounting_status", "resource_matched_comparable", "candidate_validation_count", "pre_test_validation_count", "protected_test_count", "gpu_active_seconds", "summary_path", "summary_sha256"],
    )
    integrity_issues = audit_experiment_cells(records)
    _write_csv(
        out_dir / "experiment_cell_integrity_issues.csv",
        integrity_issues,
        ["issue", "phase", "trial", "agent", "task", "model", "provider", "record_count", "detail", "source_paths", "record_index"],
    )
    _write_json(
        out_dir / "experiment_cell_integrity_status.json",
        {
            "status": "PASS" if not integrity_issues else "FAIL_EXCLUDE_AMBIGUOUS_CELLS",
            "record_n": len(records),
            "issue_n": len(integrity_issues),
            "duplicate_cell_n": sum(issue["issue"] == "duplicate_agent_task_seed_cell" for issue in integrity_issues),
            "missing_metadata_record_n": sum(issue["issue"] == "missing_cell_metadata" for issue in integrity_issues),
            "rule": "duplicate or incompletely identified cells cannot enter paired comparisons",
        },
    )
    summaries = summarize_records(records)
    _write_csv(out_dir / "experiment_group_statistics.csv", summaries, ["phase", "agent", "task", "model", "provider", "attempted_n", "successful_test_n", "n", "mean_raw_fml_credit", "mean_normalized_improvement", "sd_normalized_improvement", "ci95_halfwidth_student_t"])
    figure_path = out_dir / "figures" / "replicated_normalized_improvement.svg"
    figure_emitted = _replicated_performance_chart(summaries, figure_path)
    trial_aggregates, overall_statistics = summarize_agent_performance(records)
    _write_csv(
        out_dir / "agent_trial_aggregates.csv",
        trial_aggregates,
        ["phase", "agent", "model", "provider", "trial", "expected_task_n", "observed_task_n", "complete_task_block", "mean_normalized_improvement_across_tasks"],
    )
    _write_csv(
        out_dir / "agent_overall_statistics.csv",
        overall_statistics,
        ["phase", "agent", "model", "provider", "complete_trial_n", "mean_normalized_improvement", "sd_across_trial_means", "ci95_halfwidth_student_t"],
    )
    paired_comparisons = paired_agent_comparisons(records)
    _write_csv(
        out_dir / "paired_agent_comparisons.csv",
        paired_comparisons,
        [
            "phase", "model", "provider", "agent_left", "agent_right", "analysis_class",
            "expected_trial_n", "expected_task_n_per_trial", "observed_matched_trial_n",
            "complete_paired_trial_n", "excluded_incomplete_paired_trial_n",
            "complete_block_gate_passed", "mean_difference_left_minus_right",
            "median_difference_left_minus_right", "sd_across_paired_trial_means",
            "ci95_halfwidth_student_t", "ci95_low", "ci95_high", "cohen_dz",
            "seed_block_wins_left", "seed_block_ties", "seed_block_losses_left",
            "seed_block_win_rate_excluding_ties", "exact_sign_p_seed_blocks",
            "holm_p_seed_blocks", "holm_seed_family_size", "holm_seed_significant_0_05",
            "task_n", "task_wins_left", "task_ties", "task_losses_left",
            "task_win_rate_excluding_ties", "exact_sign_p_task_means_descriptive",
            "holm_p_task_means_descriptive", "holm_task_family_size",
            "matched_task_trial_cell_n", "cell_wins_left", "cell_ties", "cell_losses_left",
            "cell_win_rate_excluding_ties", "interpretation",
        ],
    )
    task_effects = paired_agent_task_effects(records)
    _write_csv(
        out_dir / "paired_agent_task_effects.csv",
        task_effects,
        [
            "phase", "model", "provider", "agent_left", "agent_right", "task",
            "matched_seed_n", "mean_difference_left_minus_right",
            "median_difference_left_minus_right", "sd_paired_difference",
            "ci95_halfwidth_student_t", "ci95_low", "ci95_high", "cohen_dz",
            "seed_wins_left", "seed_ties", "seed_losses_left",
            "seed_win_rate_excluding_ties", "claim_boundary",
        ],
    )
    adaptive_interactions = adaptive_opportunity_interactions(records)
    _write_csv(
        out_dir / "adaptive_opportunity_interactions.csv",
        adaptive_interactions,
        [
            "phase", "model", "provider", "adaptive_agent", "baseline_agent",
            "analysis_class", "partition_source", "partition_use", "dense_task_n",
            "sparse_task_n", "observed_matched_trial_n", "complete_interaction_trial_n",
            "complete_block_gate_passed", "mean_dense_minus_sparse_adaptive_advantage",
            "median_dense_minus_sparse_adaptive_advantage", "sd_across_trial_interactions",
            "ci95_halfwidth_student_t", "ci95_low", "ci95_high", "cohen_dz",
            "trial_wins_positive_interaction", "trial_ties",
            "trial_losses_negative_interaction", "exact_sign_p",
            "holm_p_across_adaptive_baselines", "holm_family_size",
            "holm_significant_0_05", "interpretation",
        ],
    )
    sensitivity_rows = collect_scorer_semantics_sensitivity(records)
    _write_csv(
        out_dir / "scorer_semantics_sensitivity.csv",
        sensitivity_rows,
        [
            "phase", "trial", "agent", "task", "summary_path", "summary_sha256",
            "best_val_metric", "official_last_matching_best_step",
            "sensitivity_first_matching_best_step", "best_step_shift_last_minus_first",
            "best_step_semantics_differ", "validation_step_n", "successful_validation_step_n",
            "official_persisted_snapshot_n", "valid_only_persisted_snapshot_n",
            "invalid_persisted_snapshot_n", "successful_step_missing_snapshot_n",
            "exploration_membership_differs", "valid_only_exploration_status",
            "official_metrics_replaced",
        ],
    )
    sensitivity_status = {
        "schema_version": "fml-scientist-scorer-sensitivity-v1",
        "status": "REAL_RECORD_AUDIT" if sensitivity_rows else "NO_REAL_RECORDS",
        "run_n": len(sensitivity_rows),
        "first_vs_last_best_step_difference_n": sum(row["best_step_semantics_differ"] for row in sensitivity_rows),
        "exploration_membership_difference_n": sum(row["exploration_membership_differs"] for row in sensitivity_rows),
        "official_scorer_frozen": True,
        "official_metrics_replaced": False,
        "best_step_sensitivity": "first successful exact match is reported beside, never instead of, the official last successful exact match",
        "exploration_sensitivity": "membership differences are audited now; GraphCodeBERT metrics require separately named recomputation when real snapshots exist",
    }
    _write_json(out_dir / "scorer_semantics_sensitivity_status.json", sensitivity_status)
    overall_figure_path = out_dir / "figures" / "agent_overall_normalized_improvement.svg"
    overall_figure_emitted = _agent_overall_chart(overall_statistics, overall_figure_path)
    process_records = collect_process_metric_records(metric_reports_root, records, catalog)
    _write_csv(
        out_dir / "process_metric_records.csv",
        process_records,
        ["phase", "trial", "agent", "task", "model", "provider", "metric", "value", "source_path", "source_sha256"],
    )
    process_summaries = summarize_process_metrics(process_records)
    _write_csv(
        out_dir / "process_metric_group_statistics.csv",
        process_summaries,
        ["phase", "agent", "task", "metric", "n", "mean", "sd", "ci95_halfwidth_student_t"],
    )
    process_figures = []
    for metric in sorted({row["metric"] for row in process_summaries}):
        path = out_dir / "figures" / "process" / f"{_safe_slug(metric)}.svg"
        if _process_metric_chart(metric, process_summaries, path):
            process_figures.append(str(path))
    status = {
        "schema_version": "fml-scientist-experiment-dataset-v2",
        "results_root": str(results_root),
        "record_count": len(records),
        "experiment_cell_integrity_issue_count": len(integrity_issues),
        "group_count": len(summaries),
        "replicated_group_count": sum(row["n"] >= 2 for row in summaries),
        "complete_agent_trial_block_count": sum(row["complete_task_block"] for row in trial_aggregates),
        "overall_agent_group_count": len(overall_statistics),
        "paired_agent_comparison_count": len(paired_comparisons),
        "paired_agent_task_effect_count": len(task_effects),
        "adaptive_opportunity_interaction_count": len(adaptive_interactions),
        "complete_confirmatory_adaptive_interaction_count": sum(
            row["analysis_class"] == "confirmatory" and row["complete_block_gate_passed"]
            for row in adaptive_interactions
        ),
        "scorer_sensitivity_run_count": len(sensitivity_rows),
        "best_step_semantics_difference_count": sensitivity_status["first_vs_last_best_step_difference_n"],
        "exploration_membership_difference_count": sensitivity_status["exploration_membership_difference_n"],
        "process_metric_record_count": len(process_records),
        "process_metric_run_coverage_count": len(
            {(row["phase"], row["trial"], row["agent"], row["task"]) for row in process_records}
        ),
        "process_metric_group_count": len(process_summaries),
        "process_metric_figure_count": len(process_figures),
        "experimental_figures_emitted": figure_emitted or overall_figure_emitted or bool(process_figures),
        "figures": ([str(figure_path)] if figure_emitted else []) + ([str(overall_figure_path)] if overall_figure_emitted else []) + process_figures,
        "reason": "No synthetic performance figures are emitted; figures require real replicated records and catalog-grounded FML normalization." if not (figure_emitted or overall_figure_emitted or process_figures) else "Real replicated records were normalized with the checked-in FML contracts; primary pairwise estimates use matched complete seed blocks with Student-t intervals, effect sizes, exact sign tests, and Holm adjustment; task heterogeneity and scorer sensitivities remain separately labeled.",
    }
    _write_json(out_dir / "experiment_dataset_status.json", status)
    return status
