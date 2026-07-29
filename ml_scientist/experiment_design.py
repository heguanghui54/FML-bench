"""Controlled experiment manifests for comparing FML research agents.

The manifest is deliberately separate from execution.  It freezes the factors,
budgets, trials, protected-test boundary, and exact commands before any result is
seen.  This prevents the reporting layer from silently changing the experiment.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml

from .statistical_analysis import write_statistical_analysis_protocol


PROVIDER_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
    "CodexCLI": None,
}

# Diagnostic-only pilot: one published dense-opportunity/lower-is-better task
# and one sparse-opportunity/higher-is-better task. Both are in FML-Lite and
# have relatively high across-agent discrimination in the published aggregates.
PILOT_TASKS = ("Privacy_privacymeter", "Generalization_domainbed")
# Paper-eligible transfer confirmation uses the untouched paired tasks in the
# same two domains.  The selection rule is structural (paired implementation /
# dataset), not based on observed scores from the development tasks.
HELDOUT_TRANSFER_TASKS = ("Privacy_opacus", "Generalization_domainbed_officehome")
DEFAULT_TRIAL_SEEDS = (1103, 2207, 3301)
EXECUTION_ORDER_SEED = 20260727
ADAPTIVESEARCH_CONTROLLER_PACKAGES = {
    "torch": "2.10.0",
    "transformers": "5.3.0",
}
CONTROLLER_PYTHON = ".controller-venv/bin/python"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _command(
    *,
    agent: dict[str, Any],
    task: dict[str, Any],
    model: str,
    provider: str,
    output_dir: str,
    phase: str,
    trial: int,
    seed: int,
    max_steps: int,
    eval_backend: str,
    ssh_host: str,
    remote_project_root: str,
    budget_profile: str = "legacy-unbounded",
) -> str:
    label = f"{phase}-{agent['agent_id']}-{task['task_id']}-trial{trial:02d}"
    trial_output = f"{output_dir}/{phase}/trial_{trial:02d}"
    args = [
        CONTROLLER_PYTHON,
        "run_agent_benchmark.py",
        "--agent-config",
        agent["config"],
        "--task-config",
        task["task_config"],
        "--model",
        model,
        "--provider",
        provider,
        "--seed",
        str(seed),
        "--workspace-label",
        label,
        "--output-dir",
        trial_output,
        "--budget-profile",
        budget_profile,
    ]
    if eval_backend != "local":
        args.extend(["--eval-backend", eval_backend])
    if eval_backend == "ssh":
        args.extend(
            [
                "--ssh-host",
                ssh_host,
                "--remote-project-root",
                remote_project_root,
            ]
        )
    args.append(f"agent.{agent['agent_id']}.max_steps={max_steps}")
    return shlex.join(args)


def _phase_rows(
    *,
    phase: str,
    agents: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    trials: tuple[int, ...],
    seeds: tuple[int, ...],
    max_steps: int,
    model: str,
    provider: str,
    output_dir: str,
    eval_backend: str,
    ssh_host: str,
    remote_project_root: str,
    budget_profile: str = "legacy-unbounded",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for agent in agents:
        for task in tasks:
            for trial, seed in zip(trials, seeds, strict=True):
                run_id = f"{phase}-{agent['agent_id']}-{task['task_id']}-trial{trial:02d}"
                rows.append(
                    {
                        "run_id": run_id,
                        "phase": phase,
                        "agent": agent["agent_id"],
                        "strategy_family": agent["strategy_family"],
                        "task": task["task_id"],
                        "domain": task["domain"],
                        "lite": task["lite"],
                        "trial": trial,
                        "seed": seed,
                        "agent_seed": seed,
                        "seed_role": "controller_and_research_agent_randomness",
                        "task_evaluation_seed_policy": "benchmark_owned_frozen_seed",
                        "model": model,
                        "provider": provider,
                        "max_steps": max_steps,
                        "protected_test_max_runs": 1,
                        "budget_profile": budget_profile,
                        "paper_claim_eligible": phase.startswith("confirmatory_"),
                        "evidence_class": (
                            "confirmatory_new_evidence"
                            if phase.startswith("confirmatory_")
                            else "diagnostic_pilot"
                        ),
                        "status": "PLANNED",
                        "result_root": f"{output_dir}/{phase}/trial_{trial:02d}",
                        "command": _command(
                            agent=agent,
                            task=task,
                            model=model,
                            provider=provider,
                            output_dir=output_dir,
                            phase=phase,
                            trial=trial,
                            seed=seed,
                            max_steps=max_steps,
                            eval_backend=eval_backend,
                            ssh_host=ssh_host,
                            remote_project_root=remote_project_root,
                            budget_profile=budget_profile,
                        ),
                    }
                )
    return rows


def build_resource_matched_sensitivity_protocol(
    catalog: dict[str, Any],
    *,
    model: str = "gpt-5.6-sol",
    provider: str = "CodexCLI",
    output_dir: str = "benchmark_results/resource_matched_sensitivity",
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    eval_backend: str = "ssh",
    ssh_host: str = "ubuntu-heshi",
    remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build a new comparison matrix without modifying the frozen 14-run pilot."""
    task_by_id = {task["task_id"]: task for task in catalog["tasks"]}
    tasks = [task_by_id[task_id] for task_id in PILOT_TASKS]
    agents = list(catalog["agents"]) + [{
        "agent_id": "adaptive_pipeline",
        "display_name": "Graph-bound adaptive pipeline",
        "strategy_family": "graph_memory_adaptive_composition",
        "config": "configs/local_extensions/adaptive_pipeline.yaml",
    }]
    trials = tuple(range(1, len(trial_seeds) + 1))
    rows = _phase_rows(
        phase="resource_matched_sensitivity",
        agents=agents,
        tasks=tasks,
        trials=trials,
        seeds=trial_seeds,
        max_steps=8,
        model=model,
        provider=provider,
        output_dir=output_dir,
        eval_backend=eval_backend,
        ssh_host=ssh_host,
        remote_project_root=remote_project_root,
        budget_profile="matched-v1",
    )
    rows = _freeze_execution_order(rows)
    protocol = {
        "schema_version": "fml-resource-matched-sensitivity-protocol-v1",
        "status": "FROZEN_AWAITING_EXPLICIT_GPU_AUTHORIZATION",
        "frozen_pilot_unchanged": True,
        "result_root": output_dir,
        "condition_label": "resource_matched_sensitivity",
        "model": model,
        "provider": provider,
        "controller_python": CONTROLLER_PYTHON,
        "agents": [row["agent_id"] for row in agents],
        "tasks": list(PILOT_TASKS),
        "trial_seeds": list(trial_seeds),
        "seed_interpretation": {
            "role": "controller and research-agent stochasticity",
            "not_claimed_as": "independent task-training seeds",
            "task_evaluation_seed_policy": (
                "Each benchmark retains its checked-in fixed split/training seed; "
                "the same task seed is shared across agents and agent trials."
            ),
            "replication_unit": "matched research-agent trial block",
        },
        "budget_profile": {
            "name": "matched-v1",
            "tokens": 120000,
            "wall_clock_seconds": 7200,
            "proposals": 8,
            "reviews": 8,
            "candidate_validations": 1,
            "pre_test_validations": 1,
            "protected_tests": 1,
        },
        "gpu_safety_contract": {
            "sample_seconds": 5,
            "warning_temperature_c": 82,
            "abort_temperature_c": 88,
            "thermal_throttle_abort_samples": 3,
            "thermal_pacing_enabled": True,
            "thermal_pacing_start_c": 78,
            "thermal_pacing_resume_c": 72,
        },
        "fairness_gate": "every included arm must have COMPLETE_V2 accounting and comparable ledgers",
        "authorization": {
            "gpu_execution": False,
            "restart_i4h_automatically": False,
            "purchase_cloud_compute_automatically": False,
            "cloud_cost_report_if_projected_local_gpu_hours_gt": 24,
        },
        "run_count": len(rows),
    }
    return protocol, rows


def write_resource_matched_sensitivity_protocol(
    catalog: dict[str, Any],
    out_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    protocol, rows = build_resource_matched_sensitivity_protocol(catalog, **kwargs)
    _write_json(out_dir / "resource_matched_sensitivity_protocol.json", protocol)
    fields = list(rows[0]) if rows else []
    _write_csv(out_dir / "resource_matched_sensitivity_matrix.csv", rows, fields)
    _write_json(out_dir / "resource_matched_sensitivity_matrix.json", rows)
    return {"protocol": protocol, "matrix_path": str(out_dir / "resource_matched_sensitivity_matrix.json")}


def build_heldout_transfer_protocol(
    catalog: dict[str, Any],
    *,
    model: str = "gpt-5.6-sol",
    provider: str = "CodexCLI",
    output_dir: str = "benchmark_results/heldout_transfer_confirmatory",
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    eval_backend: str = "ssh",
    ssh_host: str = "ubuntu-heshi",
    remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Freeze the paper-eligible, previously unexposed transfer comparison."""
    task_by_id = {task["task_id"]: task for task in catalog["tasks"]}
    missing = sorted(set(HELDOUT_TRANSFER_TASKS) - set(task_by_id))
    if missing:
        raise ValueError(f"Held-out transfer tasks missing from catalog: {missing}")
    tasks = [task_by_id[task_id] for task_id in HELDOUT_TRANSFER_TASKS]
    agents = list(catalog["agents"]) + [{
        "agent_id": "adaptive_pipeline",
        "display_name": "Graph-bound adaptive pipeline",
        "strategy_family": "graph_memory_adaptive_composition",
        "config": "configs/local_extensions/adaptive_pipeline.yaml",
    }]
    trials = tuple(range(1, len(trial_seeds) + 1))
    rows = _phase_rows(
        phase="confirmatory_heldout_transfer",
        agents=agents,
        tasks=tasks,
        trials=trials,
        seeds=trial_seeds,
        max_steps=8,
        model=model,
        provider=provider,
        output_dir=output_dir,
        eval_backend=eval_backend,
        ssh_host=ssh_host,
        remote_project_root=remote_project_root,
        budget_profile="matched-v1",
    )
    rows = _freeze_execution_order(rows)
    protocol = {
        "schema_version": "fml-heldout-transfer-confirmatory-protocol-v1",
        "status": "FROZEN_AWAITING_EXPOSURE_AUDIT_AND_EXECUTION",
        "paper_claim_eligible": True,
        "development_tasks": list(PILOT_TASKS),
        "heldout_tasks": list(HELDOUT_TRANSFER_TASKS),
        "selection_rule": (
            "Before any held-out result is observed, choose the paired FML-Lite "
            "task in each development domain: PrivacyMeter to Opacus and "
            "DomainBed VLCS to DomainBed OfficeHome."
        ),
        "anti_cherry_pick_rule": (
            "Both frozen held-out tasks, all eight agents, and all three seeds "
            "remain in analysis regardless of failure or score."
        ),
        "model": model,
        "provider": provider,
        "controller_python": CONTROLLER_PYTHON,
        "agents": [row["agent_id"] for row in agents],
        "trial_seeds": list(trial_seeds),
        "seed_interpretation": {
            "role": "controller and research-agent stochasticity",
            "not_claimed_as": "independent task-training seeds",
            "task_evaluation_seed_policy": (
                "Each held-out benchmark retains its checked-in fixed split/training seed; "
                "the same task seed is shared across agents and agent trials."
            ),
            "replication_unit": "matched research-agent trial block",
        },
        "budget_profile": {
            "name": "matched-v1",
            "tokens": 120000,
            "wall_clock_seconds": 7200,
            "proposals": 8,
            "reviews": 8,
            "candidate_validations": 1,
            "pre_test_validations": 1,
            "protected_tests": 1,
        },
        "gpu_safety_contract": {
            "sample_seconds": 5,
            "warning_temperature_c": 82,
            "abort_temperature_c": 88,
            "thermal_throttle_abort_samples": 3,
            "thermal_pacing_enabled": True,
            "thermal_pacing_start_c": 78,
            "thermal_pacing_resume_c": 72,
        },
        "analysis_policy": {
            "unit": "agent-task-agent_seed run",
            "primary_estimand": "paired mean FML normalized-improvement difference across frozen research-agent seed blocks",
            "uncertainty": "paired seed-block Student-t interval and Cohen dz when defined",
            "multiplicity": "Holm correction across unordered agent pairs",
            "failures": "retain every run and apply the frozen FML fallback rule while reporting failure counts separately",
            "protected_feedback": "never route protected metrics into search, memory, or skill evolution",
        },
        "required_pre_execution_artifacts": [
            "heldout_exposure_audit.json",
            "execution_authorization.json",
            "cloud_runtime_cost_comparison.html",
        ],
        "authorization": {
            "local_gpu_execution": True,
            "cloud_purchase": False,
            "restart_i4h_automatically": False,
        },
        "run_count": len(rows),
    }
    return protocol, rows


def write_heldout_transfer_protocol(
    catalog: dict[str, Any],
    out_dir: Path,
    **kwargs: Any,
) -> dict[str, Any]:
    protocol, rows = build_heldout_transfer_protocol(catalog, **kwargs)
    _write_json(out_dir / "heldout_transfer_protocol.json", protocol)
    fields = list(rows[0]) if rows else []
    _write_csv(out_dir / "heldout_transfer_matrix.csv", rows, fields)
    _write_json(out_dir / "heldout_transfer_matrix.json", rows)
    write_statistical_analysis_protocol(
        out_dir,
        primary_phase="confirmatory_heldout_transfer",
        suite_label="the two frozen held-out transfer tasks",
        seed_role="research-agent seed",
    )
    return {"protocol": protocol, "matrix_path": str(out_dir / "heldout_transfer_matrix.json")}


def _freeze_execution_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Block by phase/task/trial and deterministically randomize agent order."""
    phase_rank = {"pilot": 0, "confirmatory_lite": 1, "confirmatory_full": 2}
    ordered: list[dict[str, Any]] = []
    block_keys = sorted(
        {(row["phase"], row["task"], int(row["trial"])) for row in rows},
        key=lambda key: (phase_rank.get(key[0], 99), key[2], key[1]),
    )
    for phase, task, trial in block_keys:
        block = [
            row for row in rows
            if row["phase"] == phase and row["task"] == task and int(row["trial"]) == trial
        ]
        block.sort(
            key=lambda row: hashlib.sha256(
                f"{EXECUTION_ORDER_SEED}|{phase}|{task}|{trial}|{row['agent']}".encode()
            ).hexdigest()
        )
        block_id = f"{phase}|{task}|trial{trial:02d}"
        for row in block:
            row["randomization_block"] = block_id
            ordered.append(row)
    for index, row in enumerate(ordered, start=1):
        row["execution_order"] = index
    return ordered


def build_experiment_protocol(
    catalog: dict[str, Any],
    *,
    model: str = "SET_MODEL",
    provider: str = "OpenAI",
    output_dir: str = "benchmark_results/controlled",
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    pilot_steps: int = 1,
    confirmatory_steps: int = 3,
    include_full_extension: bool = False,
    eval_backend: str = "local",
    ssh_host: str = "ubuntu-heshi",
    remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Return a preregistration-style protocol and exact run matrix.

    The pilot is diagnostic only and cannot support confirmatory paper claims.
    The confirmatory FML-Lite phase uses every registered baseline agent, every
    Lite task, three independent trials, one model, and a matched step budget.
    """
    if len(trial_seeds) < 2:
        raise ValueError("At least two independent trials are required")
    if len(set(trial_seeds)) != len(trial_seeds):
        raise ValueError("Trial seeds must be unique")
    if pilot_steps < 1 or confirmatory_steps < 1:
        raise ValueError("Pilot and confirmatory step budgets must both be positive")
    agents = list(catalog["agents"])
    tasks = list(catalog["tasks"])
    task_by_id = {task["task_id"]: task for task in tasks}
    missing = sorted(set(PILOT_TASKS) - set(task_by_id))
    if missing:
        raise ValueError(f"Pilot tasks missing from catalog: {missing}")
    pilot_tasks = [task_by_id[name] for name in PILOT_TASKS]
    lite_tasks = [task for task in tasks if task["lite"]]
    trials = tuple(range(1, len(trial_seeds) + 1))
    # A single seed is sufficient for an engineering pilot. Confirmatory rows
    # use all frozen seeds and are the only rows eligible for primary claims.
    rows = _phase_rows(
        phase="pilot",
        agents=agents,
        tasks=pilot_tasks,
        trials=(1,),
        seeds=(trial_seeds[0],),
        max_steps=pilot_steps,
        model=model,
        provider=provider,
        output_dir=output_dir,
        eval_backend=eval_backend,
        ssh_host=ssh_host,
        remote_project_root=remote_project_root,
    )
    rows += _phase_rows(
        phase="confirmatory_lite",
        agents=agents,
        tasks=lite_tasks,
        trials=trials,
        seeds=trial_seeds,
        max_steps=confirmatory_steps,
        model=model,
        provider=provider,
        output_dir=output_dir,
        eval_backend=eval_backend,
        ssh_host=ssh_host,
        remote_project_root=remote_project_root,
    )
    if include_full_extension:
        rows += _phase_rows(
            phase="confirmatory_full",
            agents=agents,
            tasks=tasks,
            trials=trials,
            seeds=trial_seeds,
            max_steps=confirmatory_steps,
            model=model,
            provider=provider,
            output_dir=output_dir,
            eval_backend=eval_backend,
            ssh_host=ssh_host,
            remote_project_root=remote_project_root,
        )
    rows = _freeze_execution_order(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["phase"]] = counts.get(row["phase"], 0) + 1
    adaptive_params = next(
        agent["parameters"] for agent in agents if agent["agent_id"] == "adaptivesearch"
    )
    protocol = {
        "schema_version": "fml-scientist-experiment-protocol-v2",
        "catalog_commit": catalog["repository_commit"],
        "status": "TEMPLATE" if model == "SET_MODEL" else "FROZEN_READY_FOR_PREFLIGHT",
        "research_questions": [
            "How does research-agent search topology affect normalized improvement under matched budgets?",
            "Which exploration dynamics predict reliability, efficiency, and validation-to-test generalization?",
            "When does regime-adaptive search outperform fixed greedy, tree, MCTS, and evolutionary strategies?",
        ],
        "primary_estimand": f"Mean FML normalized improvement under a matched {confirmatory_steps}-step early-search budget on the confirmatory suite.",
        "secondary_estimands": [
            "Per-task canonical protected-test metric",
            "Twelve FML process-level metrics",
            "success rate, token cost, and wall-clock cost",
        ],
        "factors": {
            "agents": [agent["agent_id"] for agent in agents],
            "tasks": [task["task_id"] for task in lite_tasks],
            "model": model,
            "provider": provider,
            "trial_seeds": list(trial_seeds),
            "confirmatory_max_steps": confirmatory_steps,
            "execution_backend": eval_backend,
            "adaptive_search_embedding": {
                "model_id": adaptive_params["graphcodebert_model"],
                "device": adaptive_params["embed_device"],
                "max_tokens": adaptive_params["embed_max_tokens"],
                "cache_dir": adaptive_params.get("graphcodebert_cache_dir"),
                "local_files_only": adaptive_params.get("graphcodebert_local_files_only", False),
            },
        },
        "controls": [
            "same model and provider for every arm",
            "same per-task data, pinned task commit, target-file boundary, and metric parser",
            "same maximum agent steps and per-execution timeout",
            "unique workspace per run",
            "validation is the only search feedback",
            "candidate is frozen before one protected-test execution",
        ],
        "resource_budget_amendment": {
            "condition_label": "budget_limited_codexcli_ssh",
            "status": "frozen before any formal pilot run; based only on non-claim engineering calibration runtime",
            "engineering_evidence": "A non-claim calibration on Privacy_privacymeter showed that one native validation trains four models for 50 epochs each and required 1918.81 seconds (31 minutes 58.81 seconds) on the RTX 3060 Ti.",
            "privacy_meter_compute_forecast": {
                "validation_runtime_seconds": 1918.81,
                "validations_per_run": "max_steps + one protected-test checkpoint-regeneration validation",
                "pilot_validation_gpu_hours": 7.46,
                "confirmatory_validation_gpu_hours": 44.77,
                "scope": "PrivacyMeter validation only; excludes controller, workspace, post-processing, and protected-test overhead",
            },
            "calibration_outcome_quarantine": "Calibration validation/test scores are excluded from pilot, confirmatory, model selection, task selection, and paper claims.",
            "pilot_steps": pilot_steps,
            "confirmatory_steps": confirmatory_steps,
            "unchanged_factors": [
                "all seven agents",
                "all eight FML-Lite tasks",
                "three frozen seeds",
                "native datasets and metric directions",
                "one-way protected-test boundary",
            ],
            "claim_scope": "early-search proposal and selection performance, not reproduction of the published 100-step condition",
            "adaptive_search_limitation": "AdaptiveSearch uses a 50-step Phase-1 window, so its adaptive branching cannot activate in this primary matrix; no adaptive-branching advantage may be claimed from these runs.",
        },
        "execution_platform_contract": {
            "confirmatory_target": "Linux x86_64 with an NVIDIA GPU compatible with the task-pinned CUDA environments",
            "local_apple_silicon_role": "catalog, planning, orchestration, source audit, and report generation only",
            "controller_model_condition": (
                "Authenticated local Codex CLI in audited text-only mode; this is a new "
                "experimental condition, not an exact reproduction of published FML API runs."
                if provider == "CodexCLI"
                else f"Provider API condition: {provider}."
            ),
            "evaluation_backend": eval_backend,
            "ssh_host": ssh_host if eval_backend == "ssh" else None,
            "remote_project_root": remote_project_root if eval_backend == "ssh" else None,
            "hardware_must_be_reported": ["host", "CPU", "GPU", "VRAM", "driver", "CUDA", "OS"],
        },
        "analysis_policy": {
            "experimental_run_unit": "one agent-task-seed run",
            "primary_uncertainty_unit": "one matched seed block after equally averaging every preregistered task",
            "primary_phase": "confirmatory_lite",
            "pilot_excluded_from_primary_claims": True,
            "missing_or_failed_test_credit": "use FML scorer fallback policy; report failure counts separately",
            "uncertainty": "paired seed-block mean difference, sample SD, two-sided 95% Student-t interval, and Cohen dz when defined",
            "multiplicity": "Holm family-wise correction across all unordered agent pairs within each phase/model/provider family",
            "task_heterogeneity": "report per-task paired effects and task/task-seed win-tie-loss rates as secondary descriptive analyses",
            "small_n_warning": "three seeds have weak inferential resolution; lead with estimates and intervals and do not infer equivalence from a non-significant result",
            "selection": "no arm, task, metric, or trial removal after protected-test exposure",
            "execution_order": {
                "blocking": "phase x task x trial",
                "within_block": "agents ordered by SHA-256 of the frozen order seed and block identifiers",
                "order_seed": EXECUTION_ORDER_SEED,
                "purpose": "reduce agent-time confounding on the single-GPU runner while preserving exact reproducibility",
            },
        },
        "pilot_selection_basis": {
            "status": "published-prior-informed diagnostic selection; ineligible for primary claims",
            "tasks": list(PILOT_TASKS),
            "coverage": ["dense and sparse published post-hoc opportunity regimes", "lower- and higher-is-better native metrics", "tasks included in the frozen FML-Lite confirmatory suite"],
            "anti_cherry_pick_rule": "pilot outcomes cannot add, remove, or replace confirmatory agents, tasks, metrics, or trials",
        },
        "test_boundary": catalog["evaluation_boundary"],
        "phase_run_counts": counts,
        "total_planned_runs": len(rows),
        "paper_claim_eligibility": {
            "pilot": False,
            "confirmatory_lite": True,
            "confirmatory_full": include_full_extension,
        },
    }
    return protocol, rows


def _codex_cli_status() -> dict[str, Any]:
    executable = shutil.which("codex")
    status = {"executable": executable, "version": None, "logged_in": False, "error": None}
    if not executable:
        status["error"] = "Codex CLI executable was not found."
        return status
    try:
        version = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=30
        )
        login = subprocess.run(
            [executable, "login", "status"], capture_output=True, text=True, timeout=30
        )
        status["version"] = version.stdout.strip()
        login_text = (login.stdout + login.stderr).strip()
        status["logged_in"] = login.returncode == 0 and "logged in" in login_text.lower()
        if not status["logged_in"]:
            status["error"] = "Codex CLI is not authenticated."
    except (OSError, subprocess.SubprocessError) as exc:
        status["error"] = str(exc)
    return status


def _ssh_runner_status(
    catalog: dict[str, Any], *, ssh_host: str, remote_project_root: str
) -> dict[str, Any]:
    remote_base = str(Path(remote_project_root).parent)
    task_names = [task["task_id"] for task in catalog["tasks"]]
    env_by_task = {task["task_id"]: task["conda_env"] for task in catalog["tasks"]}
    checks = "\n".join(
        f"if test -d {shlex.quote(str(Path(remote_project_root) / 'workspace' / task))}; then echo task={shlex.quote(task)}; fi"
        for task in task_names
    )
    env_checks = "\n".join(
        f"if test -f {shlex.quote(str(Path(remote_base) / 'conda-envs' / env / '.fml_setup_complete'))}; then echo environment={shlex.quote(env)}; fi"
        for env in sorted(set(env_by_task.values()))
    )
    heldout_data_specs = {
        "Generalization_domainbed_officehome": (
            str(Path(remote_project_root) / "workspace" / "Generalization_domainbed_officehome" / "data" / "office_home"),
            10000,
            str(Path(remote_base) / "cache" / "torch" / "hub" / "checkpoints" / "resnet50-0676ba61.pth"),
            "0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a",
        ),
        "Privacy_opacus": (
            str(Path(remote_project_root) / "workspace" / "Privacy_opacus" / "opacus" / "data" / "cifar-10-batches-py"),
            7,
            "",
            "",
        ),
    }
    data_checks = []
    for task, (data_path_value, minimum_files, artifact_path_value, artifact_sha256) in heldout_data_specs.items():
        data_checks.append(f"""
data_path={shlex.quote(data_path_value)}
data_ready=no
data_resolved=''
data_count=0
data_external=no
artifact_path={shlex.quote(artifact_path_value)}
artifact_sha256=''
artifact_ready=yes
if test -n "$artifact_path"; then
  artifact_ready=no
  if test -f "$artifact_path"; then
    artifact_sha256=$(sha256sum "$artifact_path" | awk '{{print $1}}')
    if test "$artifact_sha256" = {shlex.quote(artifact_sha256)}; then artifact_ready=yes; fi
  fi
fi
if test -d "$data_path"; then
  data_resolved=$(readlink -f "$data_path" 2>/dev/null || true)
  data_count=$(find -L "$data_path" -type f 2>/dev/null | wc -l | xargs)
  case "$data_resolved" in ({shlex.quote(remote_base)}/*) data_external=yes;; esac
  if test "$data_count" -ge {minimum_files} && test "$data_external" = yes && test "$artifact_ready" = yes; then data_ready=yes; fi
fi
printf 'data=%s|%s|%s|%s|%s|%s|%s|%s\n' {shlex.quote(task)} "$data_ready" "$data_resolved" "$data_count" "$data_external" "$artifact_path" "$artifact_sha256" "$artifact_ready"
""")
    script = f"""
set -u
printf 'platform='; . /etc/os-release; printf '%s %s|' "$NAME" "$VERSION_ID"; uname -m
printf 'machine='; uname -m
printf 'environment_manager='; if test -x {shlex.quote(str(Path(remote_base) / 'miniforge3/bin/conda'))}; then echo {shlex.quote(str(Path(remote_base) / 'miniforge3/bin/conda'))}; else echo; fi
printf 'nvidia_smi='; command -v nvidia-smi || echo
printf 'gpu='; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true
printf 'remote_project_root_exists='; test -d {shlex.quote(remote_project_root)} && echo yes || echo no
{checks}
{env_checks}
{''.join(data_checks)}
exit 0
"""
    try:
        result = subprocess.run(
            [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
                ssh_host, "bash", "-s",
            ],
            input=script,
            capture_output=True,
            text=True,
            timeout=45,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "reachable": False,
            "error": str(exc),
            "workspace_tasks": {task: False for task in task_names},
            "environment_tasks": {task: False for task in task_names},
        }
    values: dict[str, str] = {}
    present_tasks = set()
    present_environments = set()
    data_tasks: dict[str, dict[str, Any]] = {}
    for line in result.stdout.splitlines():
        if line.startswith("task="):
            present_tasks.add(line.split("=", 1)[1])
        elif line.startswith("environment="):
            present_environments.add(line.split("=", 1)[1])
        elif line.startswith("data="):
            parts = line.split("=", 1)[1].split("|", 7)
            if len(parts) >= 5:
                task, ready, resolved, count, external = parts[:5]
                artifact_path = parts[5] if len(parts) > 5 else ""
                artifact_sha256 = parts[6] if len(parts) > 6 else ""
                artifact_ready = parts[7] if len(parts) > 7 else "yes"
                data_tasks[task] = {
                    "ready": ready == "yes",
                    "resolved_path": resolved or None,
                    "file_count": int(count or 0),
                    "external_disk": external == "yes",
                    "required_model_artifact": artifact_path or None,
                    "model_artifact_sha256": artifact_sha256 or None,
                    "model_artifact_ready": artifact_ready == "yes",
                }
        elif "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return {
        "reachable": result.returncode == 0,
        "error": result.stderr.strip() or None,
        "platform": values.get("platform"),
        "machine": values.get("machine"),
        "environment_manager": values.get("environment_manager") or None,
        "nvidia_smi": values.get("nvidia_smi") or None,
        "gpu": values.get("gpu") or None,
        "remote_project_root_exists": values.get("remote_project_root_exists") == "yes",
        "workspace_tasks": {task: task in present_tasks for task in task_names},
        "environment_tasks": {
            task: env_by_task[task] in present_environments for task in task_names
        },
        "heldout_data_tasks": data_tasks,
    }


def _local_environment_tasks(
    catalog: dict[str, Any], env_tool: str | None
) -> dict[str, bool]:
    task_names = [task["task_id"] for task in catalog["tasks"]]
    if env_tool is None:
        return {task: False for task in task_names}
    try:
        result = subprocess.run(
            [env_tool, "env", "list", "--json"], capture_output=True, text=True, timeout=30
        )
        prefixes = [Path(value) for value in json.loads(result.stdout).get("envs", [])]
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return {task: False for task in task_names}
    completed = {
        prefix.name for prefix in prefixes if (prefix / ".fml_setup_complete").is_file()
    }
    return {
        task["task_id"]: task["conda_env"] in completed for task in catalog["tasks"]
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _adaptivesearch_controller_status(repo: Path) -> dict[str, Any]:
    """Verify the local, offline GraphCodeBERT runtime used by AdaptiveSearch."""
    config_path = repo / "configs" / "agents" / "adaptivesearch.yaml"
    status: dict[str, Any] = {
        "python_executable": sys.executable,
        "expected_packages": dict(ADAPTIVESEARCH_CONTROLLER_PACKAGES),
        "installed_packages": {},
        "config": str(config_path),
        "model_id": None,
        "device": None,
        "max_tokens": None,
        "cache_dir": None,
        "local_files_only": None,
        "snapshot_revision": None,
        "snapshot_dir": None,
        "weight_sha256": None,
        "missing_files": [],
        "errors": [],
        "ready": False,
    }
    for distribution, expected in ADAPTIVESEARCH_CONTROLLER_PACKAGES.items():
        try:
            installed = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            installed = None
        status["installed_packages"][distribution] = installed
        if installed != expected:
            status["errors"].append(
                f"{distribution}=={expected} is required; found {installed or 'not installed'}."
            )
    try:
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        params = config["agent"]["adaptivesearch"]
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        status["errors"].append(f"Cannot read AdaptiveSearch config: {exc}")
        return status
    model_id = str(params.get("graphcodebert_model", "microsoft/graphcodebert-base"))
    device = str(params.get("embed_device", "cpu"))
    cache_value = params.get("graphcodebert_cache_dir")
    cache_dir = Path(str(cache_value)).expanduser() if cache_value else None
    if cache_dir is not None and not cache_dir.is_absolute():
        cache_dir = (repo / cache_dir).resolve()
    local_files_only = bool(params.get("graphcodebert_local_files_only", False))
    status.update(
        {
            "model_id": model_id,
            "device": device,
            "max_tokens": int(params.get("embed_max_tokens", 512)),
            "cache_dir": str(cache_dir) if cache_dir else None,
            "local_files_only": local_files_only,
        }
    )
    if device != "cpu":
        status["errors"].append(
            "The macOS CodexCLI controller must use embed_device=cpu; task evaluation remains on the SSH GPU runner."
        )
    if not local_files_only:
        status["errors"].append("graphcodebert_local_files_only must be true for frozen runs.")
    if cache_dir is None:
        status["errors"].append("graphcodebert_cache_dir is not configured.")
        return status
    model_cache = cache_dir / ("models--" + model_id.replace("/", "--"))
    ref_path = model_cache / "refs" / "main"
    try:
        revision = ref_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        status["errors"].append(f"Cannot resolve cached GraphCodeBERT revision: {exc}")
        return status
    snapshot_dir = model_cache / "snapshots" / revision
    status["snapshot_revision"] = revision
    status["snapshot_dir"] = str(snapshot_dir)
    required = [
        "config.json",
        "merges.txt",
        "pytorch_model.bin",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    missing = [name for name in required if not (snapshot_dir / name).is_file()]
    status["missing_files"] = missing
    if missing:
        status["errors"].append(f"Cached GraphCodeBERT snapshot is missing: {missing}")
    else:
        try:
            status["weight_sha256"] = _sha256_file(snapshot_dir / "pytorch_model.bin")
        except OSError as exc:
            status["errors"].append(f"Cannot hash cached GraphCodeBERT weights: {exc}")
    status["ready"] = not status["errors"]
    return status


def preflight_environment(
    catalog: dict[str, Any], *, provider: str, model: str, repo: Path,
    eval_backend: str = "local", ssh_host: str = "ubuntu-heshi",
    remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
) -> dict[str, Any]:
    """Inspect prerequisites without exposing secret values or modifying state."""
    remote = None
    if eval_backend == "ssh":
        remote = _ssh_runner_status(
            catalog, ssh_host=ssh_host, remote_project_root=remote_project_root
        )
        env_tool = remote.get("environment_manager")
    else:
        env_tool = next((name for name in ("conda", "mamba", "micromamba") if shutil.which(name)), None)
    required_key = PROVIDER_KEYS.get(provider)
    codex_status = _codex_cli_status() if provider == "CodexCLI" else None
    adaptive_controller = _adaptivesearch_controller_status(repo)
    key_present = bool(os.environ.get(required_key)) if required_key else provider == "CodexCLI" and bool(codex_status and codex_status["logged_in"])
    if remote is not None:
        workspace_tasks = remote["workspace_tasks"]
        environment_tasks = remote["environment_tasks"]
        heldout_data_tasks = remote.get("heldout_data_tasks", {})
    else:
        workspace_tasks = {
            task["task_id"]: (repo / Path(task["repository"]).parts[0] / Path(task["repository"]).parts[1]).is_dir()
            for task in catalog["tasks"]
        }
        environment_tasks = _local_environment_tasks(catalog, env_tool)
        heldout_data_tasks = {}
    lite_task_names = [task["task_id"] for task in catalog["tasks"] if task["lite"]]
    lite_workspace_tasks = {task: workspace_tasks[task] for task in lite_task_names}
    lite_environment_tasks = {task: environment_tasks[task] for task in lite_task_names}
    full_workspace_ready = all(workspace_tasks.values())
    lite_workspace_ready = all(lite_workspace_tasks.values())
    full_environment_ready = all(environment_tasks.values())
    lite_environment_ready = all(lite_environment_tasks.values())
    heldout_data_ready = (
        all(heldout_data_tasks.get(task, {}).get("ready") for task in HELDOUT_TRANSFER_TASKS)
        if remote is not None else None
    )
    blockers = []
    if env_tool is None:
        blockers.append("No conda, mamba, or micromamba executable is available.")
    if provider == "CodexCLI" and not key_present:
        blockers.append("Codex CLI is not installed and authenticated on the controller.")
    elif provider not in PROVIDER_KEYS:
        blockers.append(f"Unknown provider {provider!r}; provider key requirement cannot be verified.")
    elif required_key and not key_present:
        blockers.append(f"{required_key} is not configured in this process environment.")
    if model == "SET_MODEL":
        blockers.append("The experiment model is still the SET_MODEL placeholder.")
    if not adaptive_controller["ready"]:
        blockers.append(
            "AdaptiveSearch controller runtime is not ready: "
            + "; ".join(adaptive_controller["errors"])
        )
    system = platform.system()
    machine = remote.get("machine") if remote else platform.machine()
    gpu_tool = remote.get("nvidia_smi") if remote else shutil.which("nvidia-smi")
    platform_ok = (
        bool(remote and remote.get("reachable") and machine in {"x86_64", "amd64"} and gpu_tool)
        if eval_backend == "ssh"
        else system == "Linux" and machine in {"x86_64", "amd64"} and gpu_tool is not None
    )
    if not platform_ok:
        blockers.append(
            "The confirmatory suite requires a Linux x86_64 NVIDIA runner for the checked-in CUDA-pinned task environments."
        )
    if not lite_workspace_ready:
        blockers.append("One or more FML-Lite task workspaces have not been bootstrapped by setup.py.")
    if not lite_environment_ready:
        blockers.append("One or more FML-Lite conda environments are missing a completed setup marker.")
    if remote is not None and not heldout_data_ready:
        blockers.append(
            "One or more held-out transfer datasets or required pretrained model artifacts are missing, incomplete, hash-mismatched, or not resolved below the Ubuntu external-disk project root."
        )
    return {
        "schema_version": "fml-scientist-preflight-v4",
        "ready": not blockers,
        "ready_for_confirmatory_lite": not blockers,
        "ready_for_full_extension": not blockers and full_workspace_ready and full_environment_ready,
        "platform": remote.get("platform") if remote else platform.platform(),
        "machine": machine,
        "nvidia_smi": gpu_tool,
        "environment_manager": env_tool,
        "provider": provider,
        "model": model,
        "required_key_name": required_key,
        "required_key_present": key_present,
        "provider_auth": codex_status if codex_status is not None else {"required_key_name": required_key, "present": key_present},
        "adaptivesearch_controller": adaptive_controller,
        "eval_backend": eval_backend,
        "ssh_host": ssh_host if eval_backend == "ssh" else None,
        "remote_project_root": remote_project_root if eval_backend == "ssh" else None,
        "remote_runner": remote,
        "workspace_task_count": sum(workspace_tasks.values()),
        "workspace_task_total": len(workspace_tasks),
        "workspace_tasks": workspace_tasks,
        "lite_workspace_task_count": sum(lite_workspace_tasks.values()),
        "lite_workspace_task_total": len(lite_workspace_tasks),
        "lite_workspace_tasks": lite_workspace_tasks,
        "full_workspace_ready": full_workspace_ready,
        "environment_task_count": sum(environment_tasks.values()),
        "environment_task_total": len(environment_tasks),
        "environment_tasks": environment_tasks,
        "lite_environment_task_count": sum(lite_environment_tasks.values()),
        "lite_environment_task_total": len(lite_environment_tasks),
        "lite_environment_tasks": lite_environment_tasks,
        "full_environment_ready": full_environment_ready,
        "heldout_data_ready": heldout_data_ready,
        "heldout_data_tasks": heldout_data_tasks,
        "blockers": blockers,
    }


def write_experiment_protocol(
    catalog: dict[str, Any],
    out_dir: Path,
    *,
    model: str = "SET_MODEL",
    provider: str = "OpenAI",
    output_dir: str = "benchmark_results/controlled",
    include_full_extension: bool = False,
    pilot_steps: int = 1,
    confirmatory_steps: int = 3,
    eval_backend: str = "local",
    ssh_host: str = "ubuntu-heshi",
    remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
) -> dict[str, Any]:
    protocol, rows = build_experiment_protocol(
        catalog,
        model=model,
        provider=provider,
        output_dir=output_dir,
        pilot_steps=pilot_steps,
        confirmatory_steps=confirmatory_steps,
        include_full_extension=include_full_extension,
        eval_backend=eval_backend,
        ssh_host=ssh_host,
        remote_project_root=remote_project_root,
    )
    _write_json(out_dir / "experiment_protocol.json", protocol)
    _write_csv(
        out_dir / "run_matrix.csv",
        rows,
        [
            "run_id",
            "phase",
            "agent",
            "strategy_family",
            "task",
            "domain",
            "lite",
            "trial",
            "seed",
            "model",
            "provider",
            "max_steps",
            "protected_test_max_runs",
            "paper_claim_eligible",
            "evidence_class",
            "randomization_block",
            "execution_order",
            "status",
            "result_root",
            "command",
        ],
    )
    commands = ["#!/usr/bin/env bash", "set -euo pipefail", ""]
    current_phase = None
    for row in rows:
        if row["phase"] != current_phase:
            current_phase = row["phase"]
            commands.extend([f"# {current_phase}", ""])
        commands.append(row["command"])
    commands.extend(["", "# Score each independent trial outside the immutable result trees", ""])
    phases = sorted({row["phase"] for row in rows})
    agents = sorted({row["agent"] for row in rows})
    trials_by_phase = {
        phase: sorted({int(row["trial"]) for row in rows if row["phase"] == phase})
        for phase in phases
    }
    for phase in phases:
        phase_tasks = sorted({row["task"] for row in rows if row["phase"] == phase})
        if phase == "confirmatory_lite":
            selector = "--suite lite"
        elif phase == "confirmatory_full":
            selector = "--suite full"
        else:
            selector = "--tasks " + ",".join(phase_tasks)
        for trial in trials_by_phase[phase]:
            for agent in agents:
                result_path = shlex.quote(f"{output_dir}/{phase}/trial_{trial:02d}/{agent}")
                report_path = shlex.quote(f"metric_reports/controlled/{phase}/trial_{trial:02d}/{agent}")
                commands.append(
                    f"python compute_agent_metrics.py {result_path} {selector} --output-dir {report_path}"
                )
    path = out_dir / "run_commands.sh"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(commands) + "\n", encoding="utf-8")
    write_statistical_analysis_protocol(out_dir)
    return protocol
