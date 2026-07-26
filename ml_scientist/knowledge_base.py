"""Source-grounded agent and task dossiers for experiment and paper planning."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .catalog import load_simple_yaml
from .published_prior import agent_summary_rows, agent_task_rows, process_rows, task_card_rows


CONTROL_METHODS = {
    "theaiscientist": ["run", "_generate_ideas", "_experiment_loop", "_is_better_metric", "_run_final_test"],
    "ai_scientist_v2": ["run", "_bfts_loop", "_select_batch", "_transition_substage", "_draft", "_improve", "_debug", "_run_final_test"],
    "aide": ["run", "_search_policy", "_draft", "_improve", "_debug", "_run_final_test"],
    "aira_mcts": ["run", "_expand_leaf_and_backprop", "_backprop", "_draft", "_improve", "_debug_cycle", "_collect_results"],
    "autoresearch": ["run", "_main_loop", "_is_strict_improvement", "_keep", "_discard", "_handle_crash", "_run_final_test"],
    "openevolve": ["run", "_seed_population", "_evolution_loop", "_evolve", "_debug", "_run_final_test"],
    "adaptivesearch": ["run", "_phase1_step", "_check_phase1_trigger", "_setup_phase2", "_phase2_step", "_eval_subrules", "_run_final_test"],
}

FAILURE_MODES = {
    "theaiscientist": ["independent ideas cannot learn from one another", "large up-front idea batch can spend budget before evidence arrives", "idea-local retries may repeatedly explore similar directions"],
    "ai_scientist_v2": ["deterministic best-first selection can overexploit noisy validation leaders", "stage transitions can strand useful unfinished branches", "journal summaries can omit minority or negative evidence"],
    "aide": ["greedy best-node improvement can collapse diversity after drafts", "random buggy-leaf selection may spend budget on low-value repairs", "bounded debug depth can abandon recoverable branches"],
    "aira_mcts": ["UCT values are noisy with few visits", "backpropagated mean fitness can undervalue rare high-upside descendants", "multi-child expansion spends budget before reliable ranking"],
    "autoresearch": ["strict greedy acceptance cannot cross temporary regressions", "one incumbent creates path dependence", "validation noise can cause false keeps or discards"],
    "openevolve": ["archive diversity may remain close to the baseline", "small budgets may not permit meaningful island turnover", "feature bins can reward behavioral diversity without performance"],
    "adaptivesearch": ["the switch is irreversible", "stagnation thresholds and opportunity partition are study-specific", "embedding feedback can mischaracterize semantically important small edits"],
}

RECOMMENDED_USE = {
    "theaiscientist": "Use for broad, independently testable hypotheses when cross-idea learning is intentionally excluded or ablated.",
    "ai_scientist_v2": "Use when work naturally separates into implementation, tuning, creative, and ablation stages and evidence should carry through a journal.",
    "aide": "Use when fast solution drafting plus bounded repair is more valuable than global uncertainty-aware exploration.",
    "aira_mcts": "Use as an uncertainty-aware tree-search comparison when validation evaluations are sufficiently reliable for UCT values.",
    "autoresearch": "Use as the minimum strong greedy control and for opportunity-dense tasks where frequent local gains are expected.",
    "openevolve": "Use when multiple code families, population diversity, and escape from a single incumbent are central hypotheses.",
    "adaptivesearch": "Use when task opportunity density is unknown and online stagnation can justify switching from exploitation to broader search.",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _source_symbols(path: Path, root: Path) -> list[dict[str, Any]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    symbols = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append(
                {
                    "kind": "function",
                    "name": node.name,
                    "source": str(path.relative_to(root)),
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                }
            )
        elif isinstance(node, ast.ClassDef):
            symbols.append(
                {
                    "kind": "class",
                    "name": node.name,
                    "source": str(path.relative_to(root)),
                    "line_start": node.lineno,
                    "line_end": node.end_lineno,
                }
            )
            for method in node.body:
                if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append(
                        {
                            "kind": "method",
                            "class": node.name,
                            "name": method.name,
                            "source": str(path.relative_to(root)),
                            "line_start": method.lineno,
                            "line_end": method.end_lineno,
                        }
                    )
    return symbols


def _literal(node: ast.AST) -> Any:
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _detected_config_defaults(paths: list[Path]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            value = None
            targets: list[ast.AST] = []
            if isinstance(node, ast.Assign):
                value, targets = node.value, list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                value, targets = node.value, [node.target]
            if not isinstance(value, ast.Call) or not isinstance(value.func, ast.Attribute) or value.func.attr != "get":
                continue
            if not value.args or not isinstance(value.args[0], ast.Constant) or not isinstance(value.args[0].value, str):
                continue
            key = value.args[0].value
            default = _literal(value.args[1]) if len(value.args) >= 2 else None
            if any(isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self" for target in targets):
                defaults.setdefault(key, default)
    return defaults


def build_agent_dossiers(root: Path, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    published_summary = {row["agent_id"]: row for row in agent_summary_rows()}
    process = process_rows()
    labels = {
        "theaiscientist": "TAS v1", "ai_scientist_v2": "TAS v2", "aide": "AIDE",
        "aira_mcts": "AIRA", "autoresearch": "AutoR", "openevolve": "OEvolve",
    }
    dossiers = []
    for card in catalog["agents"]:
        agent_id = card["agent_id"]
        agent_dir = root / "agents" / agent_id
        modules = sorted(path for path in agent_dir.glob("*.py") if path.name != "__init__.py")
        symbols = [symbol for path in modules for symbol in _source_symbols(path, root)]
        by_name: dict[str, list[dict[str, Any]]] = {}
        for symbol in symbols:
            by_name.setdefault(symbol["name"], []).append(symbol)
        control_flow = []
        for name in CONTROL_METHODS[agent_id]:
            matches = by_name.get(name, [])
            if not matches:
                raise ValueError(f"Missing audited control symbol {agent_id}.{name}")
            control_flow.append(matches[0])
        config_parameters = dict(card["parameters"])
        detected_defaults = _detected_config_defaults(modules)
        effective_parameters = dict(detected_defaults)
        effective_parameters.update(config_parameters)
        published_process = None
        if agent_id in labels:
            published_process = {
                row["metric"]: row[labels[agent_id]]
                for row in process
            }
        dossiers.append(
            {
                "agent_id": agent_id,
                "display_name": card["display_name"],
                "knowledge_status": "SOURCE_AUDITED_AND_PUBLISHED_PRIOR_LINKED",
                "learning_object": "The controllable research search policy; the benchmark does not train the agent's language-model weights.",
                "strategy": {
                    key: card[key]
                    for key in (
                        "strategy_family", "state_representation", "proposal_operator", "selection_policy",
                        "debug_policy", "memory_policy", "diversity_mechanism", "paper_boundary",
                    )
                },
                "configuration": {
                    "config_path": card["config"],
                    "explicit_parameters": config_parameters,
                    "source_detected_defaults": detected_defaults,
                    "effective_parameters": effective_parameters,
                    "config_sha256": card["config_sha256"],
                },
                "source_modules": [
                    {
                        "source": str(path.relative_to(root)),
                        "sha256": _sha256(path),
                        "line_count": len(path.read_text(encoding="utf-8").splitlines()),
                    }
                    for path in modules
                ],
                "control_flow_symbols": control_flow,
                "all_symbols": symbols,
                "published_prior": published_summary.get(agent_id),
                "published_process_means": published_process,
                "known_failure_modes_to_test": FAILURE_MODES[agent_id],
                "recommended_pipeline_use": RECOMMENDED_USE[agent_id],
                "paper_interpretation_rule": "Attribute FML scores only to the ported search strategy under shared infrastructure, not to omitted native paper-writing, retrieval, or review subsystems.",
            }
        )
    return dossiers


def _canonical_baseline_payload(task_root: Path, dataset: str) -> dict[str, Any]:
    payload = {}
    for split in ("val", "test"):
        source = task_root / "baseline_results" / f"{split}_info.json"
        full = json.loads(source.read_text(encoding="utf-8"))
        payload[split] = {
            "source": str(source.relative_to(task_root.parents[1])),
            "sha256": _sha256(source),
            "canonical_dataset_payload": full.get(dataset),
        }
    return payload


def build_task_dossiers(root: Path, catalog: dict[str, Any]) -> list[dict[str, Any]]:
    prior_cards = {row["task_id"]: row for row in task_card_rows()}
    prior_results: dict[str, list[dict[str, Any]]] = {}
    for row in agent_task_rows():
        prior_results.setdefault(row["task_id"], []).append(row)
    dossiers = []
    for card in catalog["tasks"]:
        task_id = card["task_id"]
        task_root = root / "ml_tasks" / task_id
        metric_yaml = load_simple_yaml(root / card["task_config"])["metrics"]
        source_files = sorted(
            path for path in task_root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
        )
        code_files = [
            path for path in source_files
            if path.suffix == ".py" and "original_file_backup" not in path.parts
        ]
        baseline_payload = _canonical_baseline_payload(task_root, card["canonical_dataset"])
        transformed_metric = task_id == "Unlearning_open_unlearning"
        baseline_validation_display = -math.log10(card["baseline_validation"]) if transformed_metric else card["baseline_validation"]
        baseline_test_display = -math.log10(card["baseline_test"]) if transformed_metric else card["baseline_test"]
        normalization_worst = baseline_test_display if card["normalization_worst"] == "baseline" else card["normalization_worst"]
        dossiers.append(
            {
                "task_id": task_id,
                "knowledge_status": "SOURCE_AUDITED_AND_PUBLISHED_PRIOR_LINKED",
                "domain": card["domain"],
                "fml_lite": card["lite"],
                "published_task_card": prior_cards[task_id],
                "objective": card["task_description"],
                "system_contract": card["system_contract"],
                "execution": {
                    "repository": card["repository"],
                    "pinned_commit": card["pinned_commit"],
                    "conda_env": card["conda_env"],
                    "editable_target_files": card["target_files"],
                    "validation_command": card["validation_command"],
                    "protected_test_command": card["protected_test_command"],
                },
                "metric_contract": {
                    "canonical_dataset": card["canonical_dataset"],
                    "canonical_metric": card["canonical_metric"],
                    "raw_native_direction": card["canonical_direction"],
                    "task_runner_metric": card["task_metric"],
                    "task_runner_direction": card["task_metric_direction"],
                    "fml_display_transform": "-log10(raw forget_quality)" if transformed_metric else "identity",
                    "fml_display_direction": card["report_direction"],
                    "included_metrics": metric_yaml.get("include_metrics", []),
                    "all_declared_metric_directions": metric_yaml.get("per_metric_direction", {}),
                    "baseline_validation_raw": card["baseline_validation"],
                    "baseline_test_raw": card["baseline_test"],
                    "baseline_validation_display": baseline_validation_display,
                    "baseline_test_display": baseline_test_display,
                    "normalization": {
                        "space": "FML display-transformed metric",
                        "report_direction": card["report_direction"],
                        "best": card["normalization_best"],
                        "worst": normalization_worst,
                        "unbounded_worst_uses_baseline": card["normalization_worst"] == "baseline",
                        "negative_improvement_clamped_to_zero": True,
                    },
                },
                "baseline_evidence": baseline_payload,
                "published_agent_results": prior_results[task_id],
                "source_files": [
                    {"source": str(path.relative_to(root)), "sha256": _sha256(path), "bytes": path.stat().st_size}
                    for path in source_files
                ],
                "code_symbols": [symbol for path in code_files for symbol in _source_symbols(path, root)],
                "experiment_checklist": [
                    "reproduce the untouched baseline before candidate search",
                    "modify only the declared target files",
                    "use validation feedback only until the candidate is frozen",
                    "report the native metric and normalized improvement separately",
                    "retain constraint and auxiliary metrics even when they are not the optimization target",
                    "execute the protected test once and never feed it back to search",
                ],
            }
        )
    return dossiers


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def build_metric_implementation_audit(root: Path, catalog: dict[str, Any]) -> dict[str, Any]:
    scorer = root / "compute_agent_metrics.py"
    symbols = _source_symbols(scorer, root)
    by_name = {row["name"]: row for row in symbols if row["kind"] == "function"}
    required = ["display_transform", "normalized_improvement", "compute_auc_for_run", "compute_step_embeddings", "compute_exploration_metrics", "score_task"]
    missing = [name for name in required if name not in by_name]
    if missing:
        raise ValueError(f"Metric scorer symbols missing: {missing}")
    process_definitions = {
        row["name"]: row["definition"]
        for row in catalog["metrics"]
        if row["category"] != "task_performance"
    }
    return {
        "schema_version": "fml-scientist-metric-implementation-audit-v1",
        "scorer_source": "compute_agent_metrics.py",
        "scorer_sha256": _sha256(scorer),
        "audited_symbols": {name: by_name[name] for name in required},
        "implemented_process_metric_definitions": process_definitions,
        "semantic_findings": [
            {
                "finding_id": "UNLEARNING_RAW_VS_DISPLAY",
                "status": "INTENTIONAL_TRANSFORM_REQUIRES_DUAL_REPORTING",
                "paper_or_task_contract": "raw forget_quality is a KS-test p-value and higher is better",
                "checked_in_scorer": "display_transform converts positive values to -log10(p); FML normalization is lower-is-better in transformed space",
                "source_locator": by_name["display_transform"],
                "required_action": "report raw p-value semantics and transformed FML scoring direction separately",
            },
            {
                "finding_id": "EXPLORATION_VALID_STEP_SCOPE",
                "status": "PAPER_SCORER_SEMANTIC_DIFFERENCE",
                "paper_or_task_contract": "paper describes exploration embeddings over valid steps",
                "checked_in_scorer": "compute_step_embeddings loads every persisted step_*_code.json snapshot without consulting validation success",
                "source_locator": by_name["compute_step_embeddings"],
                "required_action": "freeze the scorer version, describe persisted-snapshot scope, and optionally report a valid-only sensitivity analysis without replacing the official metric",
            },
            {
                "finding_id": "BEST_IMPROVEMENT_FIRST_VS_LAST",
                "status": "PAPER_SCORER_SEMANTIC_DIFFERENCE",
                "paper_or_task_contract": "paper defines when peak validation performance was first achieved",
                "checked_in_scorer": "score_task returns the last successful step whose raw metric exactly equals best_val_metric",
                "source_locator": by_name["score_task"],
                "required_action": "use and disclose the checked-in last-match implementation for reproducibility; optionally report first-achieved sensitivity separately",
            },
        ],
        "study_rule": "Do not change metric semantics after campaign start. Any corrected metric is a separately named sensitivity analysis, never an in-place replacement.",
    }


def write_knowledge_base(root: Path, catalog: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    agents = build_agent_dossiers(root, catalog)
    tasks = build_task_dossiers(root, catalog)
    for dossier in agents:
        _json(out_dir / "agents" / f"{dossier['agent_id']}.json", dossier)
    for dossier in tasks:
        _json(out_dir / "tasks" / f"{dossier['task_id']}.json", dossier)
    _json(out_dir / "agents" / "index.json", agents)
    _json(out_dir / "tasks" / "index.json", tasks)
    metric_audit = build_metric_implementation_audit(root, catalog)
    _json(out_dir / "metric_implementation_audit.json", metric_audit)

    agent_md = [
        "# Source-Grounded FML Agent Dossiers", "",
        "These cards describe the checked-out FML ports. Published scores belong to the controlled shared-infrastructure comparison and do not measure omitted paper-writing or review modules.", "",
        _markdown_table(
            ["Agent", "Topology", "Selection", "Memory", "Published mean", "Published win %"],
            [
                [
                    dossier["display_name"], dossier["strategy"]["strategy_family"], dossier["strategy"]["selection_policy"],
                    dossier["strategy"]["memory_policy"],
                    (dossier["published_prior"] or {}).get("mean_normalized_test_improvement", "not reported"),
                    (dossier["published_prior"] or {}).get("pairwise_win_rate_percent", "not reported"),
                ]
                for dossier in agents
            ],
        ), "",
    ]
    for dossier in agents:
        agent_md.extend(
            [
                f"## {dossier['display_name']}", "",
                f"Recommended use: {dossier['recommended_pipeline_use']}", "",
                "Audited control flow: " + ", ".join(
                    f"`{symbol['source']}:{symbol['line_start']}` `{symbol['name']}`"
                    for symbol in dossier["control_flow_symbols"]
                ) + ".", "",
                "Failure modes to test:", "",
                *[f"- {risk}" for risk in dossier["known_failure_modes_to_test"]], "",
            ]
        )
    (out_dir / "agents" / "baseline_agent_dossiers.md").write_text("\n".join(agent_md) + "\n", encoding="utf-8")

    task_rows = []
    for dossier in tasks:
        metric = dossier["metric_contract"]
        published = dossier["published_task_card"]
        task_rows.append(
            {
                "task_id": dossier["task_id"], "domain": dossier["domain"], "lite": dossier["fml_lite"],
                "dataset": metric["canonical_dataset"], "baseline_method": published["baseline_method"],
                "metric": metric["canonical_metric"],
                "raw_direction": metric["raw_native_direction"], "display_transform": metric["fml_display_transform"],
                "fml_report_direction": metric["fml_display_direction"],
                "baseline_validation_raw": metric["baseline_validation_raw"], "baseline_test_raw": metric["baseline_test_raw"],
                "baseline_validation_display": metric["baseline_validation_display"], "baseline_test_display": metric["baseline_test_display"],
                "normalization_best": metric["normalization"]["best"], "normalization_worst": metric["normalization"]["worst"],
                "opportunity_density_published_post_hoc": published["opportunity_density"], "partition_published_post_hoc": published["partition"],
                "auxiliary_metric_n": len(metric["all_declared_metric_directions"]),
            }
        )
    _write_csv(out_dir / "tasks" / "task_metric_matrix.csv", task_rows, list(task_rows[0]))
    task_md = [
        "# FML Task and Metric Dossiers", "",
        "Native metrics retain their domain meaning. Cross-task normalized improvement and process metrics are reported alongside, never in place of, native results.", "",
        _markdown_table(
            ["Task", "Domain", "Baseline", "Raw metric", "Raw better", "FML transform", "FML better", "Raw test", "Displayed test", "Opp. density"],
            [[row["task_id"], row["domain"], row["baseline_method"], row["metric"], row["raw_direction"], row["display_transform"], row["fml_report_direction"], row["baseline_test_raw"], row["baseline_test_display"], row["opportunity_density_published_post_hoc"]] for row in task_rows],
        ), "",
        "Every detailed JSON card includes the full task prompt/contract, editable files, validation and protected-test commands, all declared auxiliary metrics, raw baseline payload hashes, code symbols, and published agent aggregates.",
    ]
    (out_dir / "tasks" / "fml_task_metric_dossiers.md").write_text("\n".join(task_md) + "\n", encoding="utf-8")
    metric_audit_lines = [
        "# FML Metric Implementation Audit", "",
        "The checked-in scorer is the authoritative implementation for a reproduced campaign. Paper wording and implementation differ in two places, so the scorer is frozen and the differences are disclosed rather than silently changed.", "",
        "| Finding | Status | Paper/task description | Checked-in scorer | Required action |",
        "| --- | --- | --- | --- | --- |",
    ]
    for finding in metric_audit["semantic_findings"]:
        metric_audit_lines.append(
            f"| {finding['finding_id']} | {finding['status']} | {finding['paper_or_task_contract']} | {finding['checked_in_scorer']} | {finding['required_action']} |"
        )
    metric_audit_lines.extend(["", f"Study rule: {metric_audit['study_rule']}"])
    (out_dir / "metric_implementation_audit.md").write_text("\n".join(metric_audit_lines) + "\n", encoding="utf-8")

    status = {
        "status": "SOURCE_AUDIT_COMPLETE_PUBLISHED_PRIOR_LINKED_NEW_EXPERIMENTS_PENDING",
        "repository_commit": catalog["repository_commit"],
        "agent_dossier_n": len(agents),
        "task_dossier_n": len(tasks),
        "task_metric_n": len(tasks),
        "process_metric_n": catalog["counts"]["process_metrics"],
        "all_agent_control_symbols_resolved": all(dossier["control_flow_symbols"] for dossier in agents),
        "all_task_baselines_present": all(
            dossier["metric_contract"]["baseline_validation_raw"] is not None and dossier["metric_contract"]["baseline_test_raw"] is not None
            for dossier in tasks
        ),
        "real_new_experiment_record_n": 0,
        "remaining_completion_evidence": [
            "successful pilot executions on the frozen model/provider",
            "balanced replicated confirmatory FML-Lite campaign on Linux NVIDIA infrastructure",
            "protected tests and process metrics from those runs",
            "completed paper-writing ablation and blinded manuscript reviews",
        ],
    }
    _json(out_dir / "knowledge_status.json", status)
    completion_audit = {
        "objective": "Learn all FML baseline agents, tasks, and metrics; support sound experiments and papers; record data and statistical charts.",
        "overall_status": "PARTIAL_REAL_EXPERIMENTS_AND_MANUSCRIPT_EVALUATION_PENDING",
        "requirements": [
            {
                "requirement": "learn every baseline agent",
                "status": "PROVEN",
                "evidence": ["agents/index.json", "agents/baseline_agent_dossiers.md"],
                "proof": f"{len(agents)} of {catalog['counts']['agents']} configured agents have source hashes, resolved control symbols, parameters, published priors, and failure hypotheses.",
            },
            {
                "requirement": "learn every task and native metric",
                "status": "PROVEN",
                "evidence": ["tasks/index.json", "tasks/task_metric_matrix.csv", "tasks/fml_task_metric_dossiers.md"],
                "proof": f"{len(tasks)} of {catalog['counts']['tasks']} tasks have prompts, commands, boundaries, metrics, directions, baselines, normalization, and source hashes.",
            },
            {
                "requirement": "learn FML process evaluation",
                "status": "PROVEN",
                "evidence": ["../published_prior/published_evaluation_contract.json", "../published_prior/published_process_metrics.csv"],
                "proof": f"All {catalog['counts']['process_metrics']} process metrics and both final evaluation metrics are represented; metric_implementation_audit.json records scorer semantics and paper-code differences.",
            },
            {
                "requirement": "record data and statistical charts",
                "status": "PROVEN_FOR_PUBLISHED_PRIOR_ONLY",
                "evidence": ["../published_prior/provenance_manifest.json", "../published_prior/figures/", "../published_prior/paper_ready_prior_evidence_brief.md"],
                "proof": "Published aggregate tables, derived descriptive tables, and six visually checked SVG figures exist with provenance and interpretation warnings.",
            },
            {
                "requirement": "run sound new experiments",
                "status": "NOT_YET_PROVEN",
                "evidence": ["../protocol/experiment_protocol.json", "../protocol/preflight.json", "../experiments/experiment_dataset_status.json"],
                "proof": "The protocol exists, but the reporter has zero real new records and preflight is not ready.",
            },
            {
                "requirement": "produce and evaluate a good new paper",
                "status": "NOT_YET_PROVEN",
                "evidence": ["../paper/evidence_locked_manuscript.md", "../paper/claim_evidence_registry.csv", "../paper/paper_readiness.json", "../paper_evaluation/paper_evaluation_protocol.json", "../paper_evaluation/paper_evaluation_status.json"],
                "proof": "A substantive Methods manuscript, claim map, figure plan, and readiness gates exist, but numerical Results remain locked and no blinded paper reviews have been completed.",
            },
        ],
    }
    _json(out_dir / "objective_completion_audit.json", completion_audit)
    audit_lines = [
        "# Objective Completion Audit", "",
        f"Overall status: `{completion_audit['overall_status']}`", "",
        "| Requirement | Status | Proof |", "| --- | --- | --- |",
    ]
    for row in completion_audit["requirements"]:
        audit_lines.append(f"| {row['requirement']} | {row['status']} | {row['proof']} |")
    audit_lines.extend(["", "This audit prevents source learning and published-prior charts from being mistaken for completed new experiments or a validated new paper."])
    (out_dir / "objective_completion_audit.md").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    return status
