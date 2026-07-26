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
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


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
        (1332, "Protected test"),
    ]
    for index, (x, label) in enumerate(research):
        box(x, 222, 188, label, OKABE_ITO[index % len(OKABE_ITO)])
        if index:
            arrow(x - 28, 249, x - 5, 249)
    body.append('<text x="1426" y="293" text-anchor="middle" class="small">one-way; never search feedback</text>')

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
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
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


def paired_agent_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    values = {
        (
            str(record.get("phase") or "unassigned"),
            str(record.get("model")),
            str(record.get("provider")),
            record.get("trial"),
            str(record.get("task")),
            str(record.get("agent")),
        ): float(record["normalized_improvement"])
        for record in records
        if record.get("normalized_improvement") is not None
    }
    contexts = sorted({key[:3] for key in values})
    rows = []
    for phase, model, provider in contexts:
        agents = sorted({key[5] for key in values if key[:3] == (phase, model, provider)})
        for left_index, left in enumerate(agents):
            for right in agents[left_index + 1:]:
                trial_differences: dict[int | None, list[float]] = defaultdict(list)
                keys = [key for key in values if key[:3] == (phase, model, provider) and key[5] == left]
                for key in keys:
                    _, _, _, trial, task, _ = key
                    right_key = (phase, model, provider, trial, task, right)
                    if right_key in values:
                        trial_differences[trial].append(values[key] - values[right_key])
                expected = _EXPECTED_TASKS_BY_PHASE.get(phase)
                complete_trial_means = [
                    statistics.fmean(differences)
                    for differences in trial_differences.values()
                    if differences and (expected is None or len(differences) == expected)
                ]
                if not complete_trial_means:
                    continue
                n = len(complete_trial_means)
                mean = statistics.fmean(complete_trial_means)
                sd = statistics.stdev(complete_trial_means) if n >= 2 else None
                ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
                rows.append(
                    {
                        "phase": phase,
                        "model": model,
                        "provider": provider,
                        "agent_left": left,
                        "agent_right": right,
                        "complete_paired_trial_n": n,
                        "mean_difference_left_minus_right": mean,
                        "sd_across_paired_trial_means": sd,
                        "ci95_halfwidth_student_t": ci,
                        "interpretation": "positive favors agent_left; interval is descriptive and unadjusted for multiplicity",
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
        ["agent", "task", "phase", "trial", "experimental_seed", "workspace_label", "harness_git_commit", "model", "provider", "baseline_primary_metric", "best_val_metric", "test_metric", "test_success", "fml_credit_metric", "fml_credit_status", "normalized_improvement", "total_steps", "total_ideas", "total_duration_seconds", "total_tokens", "summary_path", "summary_sha256"],
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
        ["phase", "model", "provider", "agent_left", "agent_right", "complete_paired_trial_n", "mean_difference_left_minus_right", "sd_across_paired_trial_means", "ci95_halfwidth_student_t", "interpretation"],
    )
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
        "schema_version": "fml-scientist-experiment-dataset-v1",
        "results_root": str(results_root),
        "record_count": len(records),
        "group_count": len(summaries),
        "replicated_group_count": sum(row["n"] >= 2 for row in summaries),
        "complete_agent_trial_block_count": sum(row["complete_task_block"] for row in trial_aggregates),
        "overall_agent_group_count": len(overall_statistics),
        "paired_agent_comparison_count": len(paired_comparisons),
        "process_metric_record_count": len(process_records),
        "process_metric_group_count": len(process_summaries),
        "process_metric_figure_count": len(process_figures),
        "experimental_figures_emitted": figure_emitted or overall_figure_emitted or bool(process_figures),
        "figures": ([str(figure_path)] if figure_emitted else []) + ([str(overall_figure_path)] if overall_figure_emitted else []) + process_figures,
        "reason": "No synthetic performance figures are emitted; figures require real replicated records and catalog-grounded FML normalization." if not (figure_emitted or overall_figure_emitted or process_figures) else "Real replicated records were normalized with the checked-in FML contracts; overall estimates use complete task blocks, intervals use Student-t critical values, and process metrics retain separate axes.",
    }
    _write_json(out_dir / "experiment_dataset_status.json", status)
    return status
