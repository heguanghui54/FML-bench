"""Build a provenance-rich catalog of FML-bench agents, tasks, and metrics.

The catalog is deliberately generated from the checked-out repository.  The
hand-written agent profiles describe search semantics that are not represented
in YAML, while hashes and source paths make every profile auditable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


AGENT_PROFILES: dict[str, dict[str, Any]] = {
    "theaiscientist": {
        "display_name": "The AI Scientist v1",
        "source": "agents/theaiscientist/theaiscientist.py",
        "strategy_family": "parallel_multi_idea",
        "state_representation": "independent idea workspaces with iterative runs",
        "proposal_operator": "generate multiple reflected research ideas",
        "selection_policy": "best successful validation result across ideas and retries",
        "debug_policy": "bounded per-run retries with experiment notes",
        "memory_policy": "idea-local notes and accumulated experiment results",
        "diversity_mechanism": "independent idea generation",
        "paper_boundary": "FML adapter stops after experiments; no manuscript writeup",
    },
    "ai_scientist_v2": {
        "display_name": "The AI Scientist v2",
        "source": "agents/ai_scientist_v2/agent.py",
        "strategy_family": "best_first_tree_search",
        "state_representation": "stage-local journals of draft/improve/debug tree nodes",
        "proposal_operator": "BFTS across basic, tuning, creative, and ablation stages",
        "selection_policy": "best valid node with inter-stage carryover",
        "debug_policy": "probabilistic bounded debug descendants",
        "memory_policy": "LLM journal summaries with intentional forgetting between stages",
        "diversity_mechanism": "multiple drafts and simulated parallel batches",
        "paper_boundary": "FML adapter explicitly omits writeup and multi-seed evaluation",
    },
    "aide": {
        "display_name": "AIDE",
        "source": "agents/aide/agent.py",
        "strategy_family": "greedy_solution_tree",
        "state_representation": "draft/improve/debug tree with code snapshots",
        "proposal_operator": "draft new roots or improve the best valid node",
        "selection_policy": "greedy best-node improvement with stochastic debug selection",
        "debug_policy": "bounded debug depth for buggy leaves",
        "memory_policy": "journal of plans, analyses, metrics, and errors",
        "diversity_mechanism": "multiple initial drafts",
        "paper_boundary": "experiment search only in the FML adapter",
    },
    "aira_mcts": {
        "display_name": "AIRA-MCTS",
        "source": "agents/aira_mcts/agent.py",
        "strategy_family": "monte_carlo_tree_search",
        "state_representation": "MCTS tree with visit counts and propagated values",
        "proposal_operator": "UCT-selected draft/improve/debug expansion",
        "selection_policy": "upper-confidence tree traversal and value backpropagation",
        "debug_policy": "bounded debug descendants",
        "memory_policy": "metric-ranked journal supplied to later expansions",
        "diversity_mechanism": "UCT exploration bonus",
        "paper_boundary": "experiment search only in the FML adapter",
    },
    "autoresearch": {
        "display_name": "AutoResearch",
        "source": "agents/autoresearch/agent.py",
        "strategy_family": "greedy_hill_climbing",
        "state_representation": "single incumbent plus sequential experiment log",
        "proposal_operator": "one LLM-generated modification per step",
        "selection_policy": "strict improvement keeps candidate; otherwise restore incumbent",
        "debug_policy": "debug simple crashes, skip fundamentally broken ideas",
        "memory_policy": "linear keep/discard/crash history",
        "diversity_mechanism": "prompt-conditioned proposal history only",
        "paper_boundary": "no paper generation",
    },
    "openevolve": {
        "display_name": "OpenEvolve",
        "source": "agents/openevolve/agent.py",
        "strategy_family": "map_elites_evolution",
        "state_representation": "multi-island program population and archives",
        "proposal_operator": "evolve sampled parents using elites and inspirations",
        "selection_policy": "exploration/exploitation/weighted parent sampling with migration",
        "debug_policy": "bounded probabilistic debug descendants",
        "memory_policy": "population history, island elites, and post-execution analyses",
        "diversity_mechanism": "MAP-Elites bins, islands, inspirations, and edit distance",
        "paper_boundary": "experiment search only in the FML adapter",
    },
    "adaptivesearch": {
        "display_name": "AdaptiveSearch",
        "source": "agents/adaptivesearch/agent.py",
        "strategy_family": "regime_adaptive_search",
        "state_representation": "greedy incumbent followed by round-robin branch frontier",
        "proposal_operator": "AutoResearch proposals then branch-specific adaptive prompts",
        "selection_policy": "switch from greedy to multi-branch search after stagnation",
        "debug_policy": "AutoResearch-compatible bounded crash recovery",
        "memory_policy": "phase-one log plus branch-local histories and embeddings",
        "diversity_mechanism": "GraphCodeBERT reach/effective-dimension feedback and branching",
        "paper_boundary": "no paper generation",
    },
}


PROCESS_METRICS = [
    ("Exploration Spread", "exploration", "Mean L2 distance of every persisted step-snapshot embedding from their float32 centroid."),
    ("Exploration Uniqueness", "exploration", "Agglomerative cosine-cluster count divided by all persisted snapshots; average linkage, threshold 0.015, with zero-norm snapshots assigned one extra cluster."),
    ("Exploration Reach", "exploration", "Maximum float64 L2 distance from a persisted step snapshot to the baseline-code embedding."),
    ("Effective dim", "exploration", "Participation ratio of singular-value-derived covariance eigenvalues over all persisted step snapshots."),
    ("Val-test |gap|", "generalization", "Absolute difference between FML-normalized best-validation and protected-test improvements."),
    ("Valid step ratio", "reliability", "Validation results marked success divided by summary total_steps."),
    ("AUC-over-steps", "efficiency", "Arithmetic mean of the best-so-far normalized-improvement curve over every recorded validation step; failed steps carry the incumbent forward."),
    ("First-improvement step", "efficiency", "First 1-based curve position with normalized improvement greater than 1e-15."),
    ("Late-gain fraction", "efficiency", "Share of final best-so-far improvement gained after floor(K/2) recorded validation steps; undefined with negligible total gain."),
    ("Best-improvement step", "efficiency", "Last successful recorded step whose raw primary metric exactly equals summary best_val_metric."),
    ("Token cost (M)", "cost", "Summary total_tokens divided by one million."),
    ("Wall-clock time (h)", "cost", "Summary total_duration_seconds divided by 3600."),
]


FML_LITE_TASKS = {
    "Continual_Learning_pycil",
    "Data_Efficiency_usb",
    "Generalization_domainbed",
    "Generalization_domainbed_officehome",
    "Robustness_openood",
    "Privacy_opacus",
    "Privacy_privacymeter",
    "Robustness_and_Reliability_art",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head(root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    lowered = value.lower()
    if lowered in {"null", "none", "~"}:
        return None
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value.strip('"\'')


def load_simple_yaml(path: Path) -> dict[str, Any]:
    """Read the small mapping-only YAML subset used by configs/agents."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for original in path.read_text(encoding="utf-8").splitlines():
        content = original.split("#", 1)[0].rstrip()
        if not content.strip():
            continue
        indent = len(content) - len(content.lstrip(" "))
        key, sep, raw = content.strip().partition(":")
        if not sep:
            continue
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if raw.strip():
            parent[key] = _parse_scalar(raw)
        else:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
    return root


def _load_ast_constant(path: Path, name: str) -> Any:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _extract_baseline(path: Path, dataset: str, metric: str) -> float | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        value = payload[dataset]
        if isinstance(value, dict) and "means" in value:
            value = value["means"]
        return float(value[metric])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _domain(task: str) -> str:
    prefixes = [
        "Representation_Learning",
        "Continual_Learning",
        "Data_Efficiency",
        "Fairness_and_Bias",
        "Federated_Learning",
        "Robustness_and_Reliability",
    ]
    for prefix in prefixes:
        if task.startswith(prefix):
            return prefix.replace("_and_", " & ").replace("_", " ")
    return task.split("_", 1)[0].replace("_", " ")


def build_agent_cards(root: Path) -> list[dict[str, Any]]:
    cards = []
    for config_path in sorted((root / "configs" / "agents").glob("*.yaml")):
        config = load_simple_yaml(config_path)
        agent = config["agent"]
        agent_id = agent["type"]
        profile = dict(AGENT_PROFILES[agent_id])
        source_path = root / profile["source"]
        profile.update(
            {
                "agent_id": agent_id,
                "parameters": agent.get(agent_id, {}),
                "config": str(config_path.relative_to(root)),
                "source_sha256": _sha256(source_path),
                "config_sha256": _sha256(config_path),
            }
        )
        cards.append(profile)
    if {card["agent_id"] for card in cards} != set(AGENT_PROFILES):
        raise ValueError("Agent configs and audited profiles do not match")
    return cards


def build_task_cards(root: Path) -> list[dict[str, Any]]:
    scoring = root / "compute_agent_metrics.py"
    tasks = _load_ast_constant(scoring, "TASKS")
    task_meta = _load_ast_constant(scoring, "TASK_META")
    range_meta = _load_ast_constant(scoring, "RANGE_META")
    task_configs: dict[str, Path] = {}
    for candidate in sorted((root / "configs" / "tasks").glob("*.yaml")):
        parsed = load_simple_yaml(candidate)
        task_name = (parsed.get("benchmark") or {}).get("name")
        if task_name:
            task_configs[str(task_name)] = candidate
    cards = []
    for task in tasks:
        task_root = root / "ml_tasks" / task
        config_path = task_root / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        prompt_path = task_root / "prompt.json"
        prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
        task_config_path = task_configs.get(task)
        if task_config_path is None:
            raise ValueError(f"No configs/tasks YAML maps to {task}")
        dataset, canonical_metric, canonical_direction = task_meta[task]
        report_direction, p_best, p_worst = range_meta[task]
        val_path = task_root / "baseline_results" / "val_info.json"
        test_path = task_root / "baseline_results" / "test_info.json"
        cards.append(
            {
                "task_id": task,
                "task_config": str(task_config_path.relative_to(root)),
                "domain": _domain(task),
                "lite": task in FML_LITE_TASKS,
                "repository": config["repo_dir"],
                "pinned_commit": config["pinned_commit"],
                "conda_env": config["conda_env"],
                "target_files": config["target_files"],
                "validation_command": config["val_command"],
                "protected_test_command": config["test_command"],
                "task_metric": config["metric"],
                "task_metric_direction": config["metric_direction"],
                "canonical_dataset": dataset,
                "canonical_metric": canonical_metric,
                "canonical_direction": canonical_direction,
                "report_direction": report_direction,
                "normalization_best": p_best,
                "normalization_worst": p_worst,
                "baseline_validation": _extract_baseline(val_path, dataset, canonical_metric),
                "baseline_test": _extract_baseline(test_path, dataset, canonical_metric),
                "task_description": prompt.get("task_description", ""),
                "system_contract": prompt.get("system", ""),
                "config_sha256": _sha256(config_path),
                "task_config_sha256": _sha256(task_config_path),
                "prompt_sha256": _sha256(prompt_path),
                "baseline_validation_sha256": _sha256(val_path),
                "baseline_test_sha256": _sha256(test_path),
            }
        )
    return cards


def build_metric_cards(task_cards: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_metrics = [
        {
            "metric_id": f"task::{card['task_id']}::{card['canonical_metric']}",
            "name": card["canonical_metric"],
            "category": "task_performance",
            "task_id": card["task_id"],
            "dataset": card["canonical_dataset"],
            "direction": card["canonical_direction"],
            "definition": (
                f"Canonical protected-test metric for {card['task_id']} on "
                f"{card['canonical_dataset']}; {card['canonical_direction']} is better."
            ),
            "baseline_validation": card["baseline_validation"],
            "baseline_test": card["baseline_test"],
            "source": "compute_agent_metrics.py::TASK_META",
        }
        for card in task_cards
    ]
    process_metrics = [
        {
            "metric_id": f"process::{name}",
            "name": name,
            "category": category,
            "definition": definition,
            "source": "compute_agent_metrics.py::score_task",
        }
        for name, category, definition in PROCESS_METRICS
    ]
    return task_metrics + process_metrics


def build_catalog(root: Path) -> dict[str, Any]:
    root = root.resolve()
    agents = build_agent_cards(root)
    tasks = build_task_cards(root)
    metrics = build_metric_cards(tasks)
    return {
        "schema_version": "fml-scientist-catalog-v1",
        "repository_commit": _git_head(root),
        "counts": {
            "agents": len(agents),
            "tasks": len(tasks),
            "lite_tasks": sum(card["lite"] for card in tasks),
            "task_metrics": len(tasks),
            "process_metrics": len(PROCESS_METRICS),
        },
        "agents": agents,
        "tasks": tasks,
        "metrics": metrics,
        "evaluation_boundary": {
            "search_feedback": "validation only",
            "protected_test": "run once after search freeze; result is not returned to the agent",
            "paper_claims": "must resolve to frozen evidence artifacts",
        },
    }
