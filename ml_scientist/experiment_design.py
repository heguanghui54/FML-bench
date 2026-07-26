"""Controlled experiment manifests for comparing FML research agents.

The manifest is deliberately separate from execution.  It freezes the factors,
budgets, trials, protected-test boundary, and exact commands before any result is
seen.  This prevents the reporting layer from silently changing the experiment.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import shlex
import shutil
from pathlib import Path
from typing import Any, Iterable


PROVIDER_KEYS = {
    "OpenAI": "OPENAI_API_KEY",
    "Google": "GOOGLE_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "OpenRouter": "OPENROUTER_API_KEY",
}

PILOT_TASKS = ("Fairness_fairlearn", "Causality_gcastle")
DEFAULT_TRIAL_SEEDS = (1103, 2207, 3301)


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
) -> str:
    label = f"{phase}-{agent['agent_id']}-{task['task_id']}-trial{trial:02d}"
    trial_output = f"{output_dir}/{phase}/trial_{trial:02d}"
    args = [
        "python",
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
        f"agent.{agent['agent_id']}.max_steps={max_steps}",
    ]
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
                        "model": model,
                        "provider": provider,
                        "max_steps": max_steps,
                        "protected_test_max_runs": 1,
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
                        ),
                    }
                )
    return rows


def build_experiment_protocol(
    catalog: dict[str, Any],
    *,
    model: str = "SET_MODEL",
    provider: str = "OpenAI",
    output_dir: str = "benchmark_results/controlled",
    trial_seeds: tuple[int, ...] = DEFAULT_TRIAL_SEEDS,
    pilot_steps: int = 15,
    confirmatory_steps: int = 100,
    include_full_extension: bool = False,
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
        )
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["phase"]] = counts.get(row["phase"], 0) + 1
    protocol = {
        "schema_version": "fml-scientist-experiment-protocol-v1",
        "catalog_commit": catalog["repository_commit"],
        "status": "TEMPLATE" if model == "SET_MODEL" else "FROZEN_READY_FOR_PREFLIGHT",
        "research_questions": [
            "How does research-agent search topology affect normalized improvement under matched budgets?",
            "Which exploration dynamics predict reliability, efficiency, and validation-to-test generalization?",
            "When does regime-adaptive search outperform fixed greedy, tree, MCTS, and evolutionary strategies?",
        ],
        "primary_estimand": "Mean FML normalized improvement on the confirmatory suite.",
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
        },
        "controls": [
            "same model and provider for every arm",
            "same per-task data, pinned task commit, target-file boundary, and metric parser",
            "same maximum agent steps and per-execution timeout",
            "unique workspace per run",
            "validation is the only search feedback",
            "candidate is frozen before one protected-test execution",
        ],
        "execution_platform_contract": {
            "confirmatory_target": "Linux x86_64 with an NVIDIA GPU compatible with the task-pinned CUDA environments",
            "local_apple_silicon_role": "catalog, planning, orchestration, source audit, and report generation only",
            "hardware_must_be_reported": ["host", "CPU", "GPU", "VRAM", "driver", "CUDA", "OS"],
        },
        "analysis_policy": {
            "unit": "one independent agent-task trial",
            "primary_phase": "confirmatory_lite",
            "pilot_excluded_from_primary_claims": True,
            "missing_or_failed_test_credit": "use FML scorer fallback policy; report failure counts separately",
            "uncertainty": "sample SD and two-sided 95% Student-t interval when n >= 2",
            "multiplicity": "report all task-level estimates; label unadjusted exploratory comparisons",
            "selection": "no arm, task, metric, or trial removal after protected-test exposure",
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


def preflight_environment(
    catalog: dict[str, Any], *, provider: str, model: str, repo: Path
) -> dict[str, Any]:
    """Inspect prerequisites without exposing secret values or modifying state."""
    env_tool = next((name for name in ("conda", "mamba", "micromamba") if shutil.which(name)), None)
    required_key = PROVIDER_KEYS.get(provider)
    key_present = bool(required_key and os.environ.get(required_key))
    workspace_tasks = {
        task["task_id"]: (repo / Path(task["repository"]).parts[0] / Path(task["repository"]).parts[1]).is_dir()
        for task in catalog["tasks"]
    }
    blockers = []
    if env_tool is None:
        blockers.append("No conda, mamba, or micromamba executable is available.")
    if required_key is None:
        blockers.append(f"Unknown provider {provider!r}; provider key requirement cannot be verified.")
    elif not key_present:
        blockers.append(f"{required_key} is not configured in this process environment.")
    if model == "SET_MODEL":
        blockers.append("The experiment model is still the SET_MODEL placeholder.")
    system = platform.system()
    machine = platform.machine()
    gpu_tool = shutil.which("nvidia-smi")
    if system != "Linux" or machine not in {"x86_64", "amd64"} or gpu_tool is None:
        blockers.append(
            "The confirmatory suite requires a Linux x86_64 NVIDIA runner for the checked-in CUDA-pinned task environments."
        )
    if not all(workspace_tasks.values()):
        blockers.append("One or more task workspaces have not been bootstrapped by setup.py.")
    return {
        "schema_version": "fml-scientist-preflight-v1",
        "ready": not blockers,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "nvidia_smi": gpu_tool,
        "environment_manager": env_tool,
        "provider": provider,
        "model": model,
        "required_key_name": required_key,
        "required_key_present": key_present,
        "workspace_task_count": sum(workspace_tasks.values()),
        "workspace_task_total": len(workspace_tasks),
        "workspace_tasks": workspace_tasks,
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
) -> dict[str, Any]:
    protocol, rows = build_experiment_protocol(
        catalog,
        model=model,
        provider=provider,
        output_dir=output_dir,
        include_full_extension=include_full_extension,
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
    return protocol
