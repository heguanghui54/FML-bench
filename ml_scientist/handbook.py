"""Human-readable baseline, metric, task, and paper-planning artifacts."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


TRANSFER = {
    "theaiscientist": "Use independent multi-idea roots when the hypothesis space is still broad.",
    "ai_scientist_v2": "Allocate explicit basic, tuning, creative, and ablation sub-budgets inside experiment nodes.",
    "aide": "Represent draft, improve, and bounded debug actions as inspectable solution-tree children.",
    "aira_mcts": "Use uncertainty-aware tree selection when evidence is sparse instead of always choosing the incumbent.",
    "autoresearch": "Use the strict keep/discard/revert loop as the minimum code-modification inner loop.",
    "openevolve": "Maintain islands and behavioral diversity when local search repeatedly converges to one code family.",
    "adaptivesearch": "Detect stagnation and switch from greedy exploitation to a controlled multi-branch frontier.",
}


def _cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def write_handbook(catalog: dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    agents = catalog["agents"]
    tasks = catalog["tasks"]
    process_metrics = [metric for metric in catalog["metrics"] if metric["category"] != "task_performance"]
    sections = [
        "# FML-Bench Baseline and Evaluation Handbook",
        "",
        f"This handbook is generated from repository commit `{catalog['repository_commit']}`. "
        "It covers all configured implementations and hashes their source/config files in `fml_catalog.json`.",
        "",
        "## What the benchmark tests",
        "",
        "FML-Bench tests whether an ML research agent can modify constrained task code, use visible validation feedback to search, and deliver a frozen candidate that generalizes to a protected test. It evaluates both the resulting ML metric and the dynamics, reliability, efficiency, and cost of the research process.",
        "",
        "## Seven baseline search strategies",
        "",
        _table(
            ["Agent", "Primary topology", "Selection", "Memory", "Paper boundary"],
            [[a["display_name"], a["strategy_family"], a["selection_policy"], a["memory_policy"], a["paper_boundary"]] for a in agents],
        ),
        "",
        "## What the unified pipeline learns from each baseline",
        "",
        _table(
            ["Agent", "Reusable operator"],
            [[a["display_name"], TRANSFER[a["agent_id"]]] for a in agents],
        ),
        "",
        "These operators are not collapsed into one opaque controller. Stage 3 places them in the same dependency graph; each code, experiment, analysis, review, and writing node receives its own bounded proposal-review-repair loop.",
        "",
        "## Eighteen tasks and native metrics",
        "",
        _table(
            ["Task", "Domain", "Lite", "Dataset", "Metric", "Better", "Val baseline", "Test baseline"],
            [[t["task_id"], t["domain"], "yes" if t["lite"] else "no", t["canonical_dataset"], t["canonical_metric"], t["canonical_direction"], t["baseline_validation"], t["baseline_test"]] for t in tasks],
        ),
        "",
        "Raw native metrics retain their domain meaning and are never averaged across tasks. Cross-task comparisons use FML normalized improvement with the checked-in `RANGE_META` contract.",
        "",
        "## Twelve process metrics used by Stage 5",
        "",
        _table(
            ["Family", "Metric", "Definition"],
            [[m["category"], m["name"], m["definition"]] for m in process_metrics],
        ),
        "",
        "Stage 5 evaluates each experiment twice: first with its task-native metric and hard constraints, then with FML's process metrics. Paper nodes use a third reviewer family—claim/evidence traceability, statistics, citations, reproducibility, and clarity—but may not change empirical outcomes.",
        "",
        "## Evidence and paper boundary",
        "",
        "The provisional outline and methods nodes may execute early. Results, abstract conclusions, and final paper writing remain locked until baseline reproduction, candidate validation, replication, ablation, process scoring, evidence freeze, and the one-way protected test have passed. Writing is followed by blocking pre-review integrity, full multi-role review, issue-by-issue amendment triage, revision, focused re-review, optional re-revision, and an independent final integrity pass. If writing or review reveals an upstream defect, it creates a versioned amendment that returns to code or experiment nodes; the exposed protected test is never reused as search feedback and empirical changes require a new hidden evaluation.",
    ]
    path = out_dir / "fml_baseline_evaluation_handbook.md"
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return path


def write_provisional_paper(catalog: dict[str, Any], out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outline = out_dir / "provisional_paper_outline.md"
    outline.write_text(
        "\n".join(
            [
                "# A Self-Evolving, Evidence-Gated ML Research-to-Paper Pipeline",
                "",
                "Status: PROVISIONAL. Empirical claims and result prose are locked.",
                "",
                "## Abstract",
                "",
                "[LOCKED: populate only from the frozen evidence bundle.]",
                "",
                "## 1. Introduction",
                "",
                "Motivate the gap between code-search agents and a traceable research-to-paper system.",
                "",
                "## 2. Related work",
                "",
                "Compare greedy, multi-idea, tree, MCTS, evolutionary, and adaptive research-agent search.",
                "",
                "## 3. Method",
                "",
                "Describe the Stage 0-7 controller, unified Stage-3 graph, node-local inner loops, four skill registries, amendment semantics, and protected-test boundary.",
                "",
                "## 4. Experimental protocol",
                "",
                "Report the frozen FML task/metric contracts, matched budgets, independent trials, replication, ablation, and analysis policy.",
                "",
                "## 5. Results",
                "",
                "[LOCKED: tables and plots must be generated from hashed result records.]",
                "",
                "## 6. Error analysis and ablations",
                "",
                "[LOCKED: distinguish invalid execution, regression, uncertainty, and promotable improvements.]",
                "",
                "## 7. Limitations and threats to validity",
                "",
                "Include model/provider dependence, task coverage, stochasticity, benchmark contamination, and protected-test limitations.",
                "",
                "## 8. Conclusion",
                "",
                "[LOCKED: no superiority claim before confirmatory evidence.]",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    methods = out_dir / "provisional_methods.md"
    methods.write_text(
        "\n".join(
            [
                "# Provisional Methods",
                "",
                f"Repository commit: `{catalog['repository_commit']}`.",
                f"Benchmark inventory: {catalog['counts']['agents']} agents, {catalog['counts']['tasks']} tasks, {catalog['counts']['lite_tasks']} Lite tasks, {catalog['counts']['process_metrics']} process metrics.",
                "",
                "The controller freezes memory and evaluation contracts, selects baseline-derived search operators, and plans the full experiment-to-paper DAG at Stage 3. Every node executes a typed inner proposal-review-repair loop. Validation may guide search; the protected test is executed only after the candidate and evidence bundle are frozen and cannot feed back into method development.",
                "",
                "This file may evolve before evidence freeze. Every numerical result remains prohibited here until it resolves to a hashed row in the claim-evidence registry.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry = out_dir / "claim_evidence_registry.csv"
    with registry.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["claim_id", "paper_section", "claim", "required_evidence", "artifact_hash", "status"])
        writer.writerow(["C1", "Results", "relative agent performance", "confirmatory normalized-improvement table", "", "LOCKED_NO_DATA"])
        writer.writerow(["C2", "Results", "search-process differences", "FML process-metric table with uncertainty", "", "LOCKED_NO_DATA"])
        writer.writerow(["C3", "Ablations", "benefit of each pipeline mechanism", "matched-budget ablation and replication results", "", "LOCKED_NO_DATA"])
    return {"outline": outline, "methods": methods, "registry": registry}
