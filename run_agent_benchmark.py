#!/usr/bin/env python3
"""
Main script for running agents on benchmark tasks.

Usage:
    python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml --model gpt-5 --provider OpenAI
    python run_agent_benchmark.py --agent-config configs/agents/theaiscientist.yaml --task-config configs/tasks/causality_causalml.yaml
    python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml agent.aide.num_drafts=3
"""
import argparse
import hashlib
import json
import os
import random
import signal
import subprocess
import sys
import yaml
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents import AgentConfig, AgentType, AgentRegistry, AgentResult
from benchmark.runner import BenchmarkRunner


# ---------------------------------------------------------------------------
# Config loading & overrides
# ---------------------------------------------------------------------------

def parse_args():
    """Parse command line arguments with support for nested config overrides."""
    parser = argparse.ArgumentParser(
        description="Run AI research agents on benchmark tasks",
        epilog="""
Examples:
  python run_agent_benchmark.py --agent-config configs/agents/aide.yaml --task-config configs/tasks/generalization.yaml --model gpt-5 --provider OpenAI
  python run_agent_benchmark.py --agent-config configs/agents/theaiscientist.yaml --task-config configs/tasks/causality_causalml.yaml agent.theaiscientist.max_runs=10
        """
    )
    parser.add_argument("--agent-config", type=str, required=True,
                        help="Agent config file (e.g., configs/agents/aide.yaml)")
    parser.add_argument("--task-config", type=str, required=True,
                        help="Task config file (e.g., configs/tasks/generalization.yaml)")
    parser.add_argument("--model", type=str,
                        help="Model to use (e.g., gpt-5-2025-08-07, gemini-2.5-pro)")
    parser.add_argument("--provider", type=str,
                        help="Provider to use (e.g., CodexCLI, OpenAI, Google, OpenRouter)")
    parser.add_argument("--workspace-label", type=str, default=None,
                        help="Optional label for workspace copy. A unique ID is always appended.")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Root directory for experiment outputs (default: benchmark_results)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Research-controller seed recorded in summary.json (default: 0)")
    parser.add_argument(
        "--budget-profile", choices=["legacy-unbounded", "matched-v1"],
        default="legacy-unbounded",
        help="Shared resource envelope. Use matched-v1 for post-fix comparisons; "
             "legacy-unbounded preserves diagnostic compatibility.",
    )
    parser.add_argument("--save-code-backup", action="store_true", default=False,
                        help="Back up all git-changed files in the task repo into "
                             "execution_<ts>/code_backup/ before each validation run. "
                             "Off by default: code_backup/ is dominated by large "
                             "regenerable artifacts (checkpoints, dataset caches, ROC "
                             "plots) that scoring/analysis never reads; the agent's code "
                             "edits are already kept in step_snapshots/. Enable to "
                             "retain the full per-step workspace snapshot for debugging.")
    parser.add_argument("--eval-backend", choices=["local", "ssh"], default=None,
                        help="Where validation/test commands run. Defaults to "
                             "FMLBENCH_EVAL_BACKEND, then local.")
    parser.add_argument("--ssh-host", default=None,
                        help="SSH host alias for --eval-backend ssh. Defaults to "
                             "FML_SSH_HOST, then ubuntu-heshi.")
    parser.add_argument("--remote-project-root", default=None,
                        help="Prepared FML project root on the Ubuntu external disk. "
                             "Defaults to FML_SSH_REMOTE_ROOT.")
    parser.add_argument("--keep-remote-workspace", action="store_true", default=False,
                        help="Retain the isolated Ubuntu run workspace for debugging. "
                             "Off by default to control external-disk usage.")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides in format: key=value (e.g., agent.aide.num_drafts=3)")
    return parser.parse_args()


def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)


def _parse_override_value(value: str):
    """Convert an override string value to the appropriate Python type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(config: Dict[str, Any], args) -> Dict[str, Any]:
    """Apply command-line overrides (--model, --provider, positional key=value) to config."""
    overrides = list(args.overrides) if args.overrides else []
    if args.model:
        overrides.append(f"agent.model={args.model}")
    if args.provider:
        overrides.append(f"agent.provider={args.provider}")

    for override_str in overrides:
        if "=" not in override_str:
            print(f"Warning: Invalid override format (expected key=value): {override_str}")
            continue
        key, value = override_str.split("=", 1)
        value = _parse_override_value(value)
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
        print(f"Override: {key} = {value}")

    return config


# ---------------------------------------------------------------------------
# Agent config construction
# ---------------------------------------------------------------------------

AGENT_TYPE_MAP = {
    "theaiscientist": AgentType.THEAISCIENTIST,
    "ai_scientist_v2": AgentType.AI_SCIENTIST_V2,
    "aide": AgentType.AIDE,
    "aira_mcts": AgentType.AIRA_MCTS,
    "openevolve": AgentType.OPENEVOLVE,
    "autoresearch": AgentType.AUTORESEARCH,
    "adaptivesearch": AgentType.ADAPTIVESEARCH,
    "adaptive_pipeline": AgentType.ADAPTIVE_PIPELINE,
}


def get_harness_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_harness_source_manifest(root: str | None = None) -> dict[str, Any]:
    """Hash the effective harness source, including uncommitted new modules.

    A git commit alone is insufficient while the repair branch has tracked and
    untracked implementation files.  This manifest deliberately excludes run
    outputs and workspaces and hashes only executable/configuration sources.
    """
    base = os.path.abspath(root or os.path.dirname(os.path.abspath(__file__)))
    source_roots = ("agents", "benchmark", "ml_scientist", "configs", "scripts", "ml_tasks")
    suffixes = {".py", ".yaml", ".yml", ".json", ".sh"}
    paths: list[str] = []
    top_level = os.path.join(base, "run_agent_benchmark.py")
    if os.path.isfile(top_level):
        paths.append(top_level)
    for relative_root in source_roots:
        absolute_root = os.path.join(base, relative_root)
        if not os.path.isdir(absolute_root):
            continue
        for directory, names, filenames in os.walk(absolute_root):
            names[:] = sorted(
                name for name in names
                if name not in {"__pycache__", ".git", "results_tmp", "baseline_results"}
            )
            for filename in sorted(filenames):
                path = os.path.join(directory, filename)
                if os.path.splitext(filename)[1].lower() in suffixes and os.path.isfile(path):
                    paths.append(path)
    records = []
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        relative = os.path.relpath(path, base).replace(os.sep, "/")
        with open(path, "rb") as handle:
            file_sha256 = hashlib.sha256(handle.read()).hexdigest()
        records.append({"path": relative, "sha256": file_sha256, "size_bytes": os.path.getsize(path)})
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha256.encode("ascii"))
        digest.update(b"\n")
    return {
        "schema_version": "fml-harness-source-manifest-v1",
        "git_commit": get_harness_git_commit() if root is None else "test-root",
        "source_file_count": len(records),
        "content_sha256": digest.hexdigest(),
        "files": records,
    }


def get_agent_config(config: Dict[str, Any]) -> AgentConfig:
    """Create AgentConfig from the loaded config dict."""
    agent_cfg = config.get("agent", {})
    agent_type_str = agent_cfg.get("type", "theaiscientist")

    if agent_type_str not in AGENT_TYPE_MAP:
        raise ValueError(
            f"Unknown agent type: {agent_type_str}. "
            f"Available: {sorted(AGENT_TYPE_MAP.keys())}"
        )

    # Resolve agent-specific params
    agent_params = agent_cfg.get(agent_type_str, {})

    runtime_params = {
        "metrics": config.get("metrics", {}),
        "experiment": config.get("experiment", {}),
        "budget_profile": config.get("budget_profile", "legacy-unbounded"),
        "budget_limits": config.get("budget_limits", {}),
        "reproducibility_contract": config.get("reproducibility_contract", {}),
    }

    return AgentConfig(
        agent_type=AGENT_TYPE_MAP[agent_type_str],
        model=agent_cfg.get("model", "gpt-5"),
        provider=agent_cfg.get("provider", "OpenAI"),
        agent_params=agent_params,
        runtime_params=runtime_params,
    )


# ---------------------------------------------------------------------------
# Result saving & printing
# ---------------------------------------------------------------------------

def save_results(result, config: Dict[str, Any], runner: BenchmarkRunner):
    """Save results to summary.json. Handles both AgentResult and legacy dict."""
    if isinstance(result, AgentResult):
        # Run-level metadata for reproducibility and analysis
        agent_cfg = runner.agent.config
        task_cfg = runner.config  # task's config.json
        baseline_primary = None
        try:
            from benchmark.utils import extract_primary_metric
            include_ds = agent_cfg.runtime_params.get("metrics", {}).get("include_datasets")
            baseline_primary = extract_primary_metric(
                runner.baseline_results, task_cfg.get("metric", ""), include_ds,
            )
        except Exception:
            pass

        summary = {
            "schema_version": "fml-summary-v2",
            "benchmark": runner.benchmark_name,
            "agent": agent_cfg.agent_type.value,
            "model": agent_cfg.model,
            "provider": agent_cfg.provider,
            "experimental_seed": (agent_cfg.runtime_params.get("experiment") or {}).get("seed"),
            "workspace_label": runner.workspace_label,
            "harness_git_commit": get_harness_git_commit(),
            "agent_params": agent_cfg.agent_params,
            "task_config": {
                "metric": task_cfg.get("metric", ""),
                "metric_direction": task_cfg.get("metric_direction", ""),
                "conda_env": task_cfg.get("conda_env", ""),
                "repo_dir": task_cfg.get("repo_dir", ""),
                "pinned_commit": task_cfg.get("pinned_commit", ""),
                "target_files": task_cfg.get("target_files", []),
            },
            "baseline_primary_metric": baseline_primary,
            "total_steps": result.total_steps,
            "total_ideas": result.total_ideas,
            "total_duration_seconds": result.total_duration_seconds,
            "best_val_metric": result.best_step.primary_metric if result.best_step else None,
            "test_result": result.test_result,
            "val_steps": [asdict(s) for s in result.all_steps],
            "token_usage": result.token_usage,
            "parent_workspace": result.parent_workspace,
            "save_code_backup": runner.save_code_backup,
            "execution_backend": {
                "name": runner.eval_backend,
                "ssh_host": runner.ssh_host if runner.eval_backend == "ssh" else None,
                "remote_project_root": (
                    runner.remote_project_root if runner.eval_backend == "ssh" else None
                ),
                "gpu_idle_gate": runner.require_gpu_idle,
            },
            "metadata": result.metadata,
        }
        execution_contracts = result.metadata.get("execution_contracts", {})
        summary["resource_accounting_status"] = "COMPLETE_V2"
        summary["budget_ledger"] = execution_contracts.get("budget_ledger", {})
        summary["execution_counts"] = execution_contracts.get("execution_counts", {})
        summary["candidate_activation"] = execution_contracts.get("activation_records", [])
        summary["reproducibility"] = {
            "contract": (agent_cfg.runtime_params.get("reproducibility_contract") or {}),
            "environment_evidence": execution_contracts.get("reproducibility_records", []),
            "protected_test_comparison": (
                result.test_result.get("reproducibility") if result.test_result else None
            ),
        }
        summary["gpu_telemetry"] = execution_contracts.get("gpu_telemetry_summaries", [])
        summary["gpu_safety_contract"] = execution_contracts.get("gpu_safety_contract")
        save_path = os.path.join(result.parent_workspace, "summary.json")
    elif isinstance(result, dict):
        summary = result
        pw = result.get("parent_workspace", "")
        if pw and os.path.isdir(pw):
            save_path = os.path.join(pw, "summary.json")
        else:
            save_path = os.path.join("benchmark_results", "summary.json")
    else:
        print(f"Warning: Unexpected result type {type(result)}, skipping save.")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    source_manifest = config.get("_harness_source_manifest")
    if isinstance(source_manifest, dict):
        manifest_path = os.path.join(os.path.dirname(save_path), "harness_source_manifest.json")
        with open(manifest_path, "w") as handle:
            json.dump(source_manifest, handle, indent=2)
        with open(manifest_path, "rb") as handle:
            manifest_sha256 = hashlib.sha256(handle.read()).hexdigest()
        summary["harness_source_manifest"] = {
            "path": manifest_path,
            "sha256": manifest_sha256,
            "content_sha256": source_manifest.get("content_sha256"),
            "source_file_count": source_manifest.get("source_file_count"),
        }
    with open(save_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved to: {save_path}")

    # Also save the config used for reproducibility
    config_save_path = os.path.join(os.path.dirname(save_path), "config_used.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Configuration saved to: {config_save_path}")


def print_summary(result):
    """Print a concise summary. Handles both AgentResult and legacy dict."""
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")

    if isinstance(result, AgentResult):
        if result.best_step:
            print(f"Best Val Metric: {result.best_step.primary_metric}"
                  f" (step {result.best_step.step_id}, idea: {result.best_step.idea_description})")
        else:
            print("No successful validation runs.")

        if result.test_result and result.test_result.get("success"):
            print(f"Test Metric: {result.test_result.get('primary_metric')}")
        elif result.test_result:
            print(f"Test Failed: {result.test_result.get('error', 'unknown')}")

        print(f"Total Steps: {result.total_steps}")
        print(f"Total Ideas: {result.total_ideas}")

        # Token usage
        if result.token_usage:
            tu = result.token_usage
            print(f"\nToken Usage: {tu.get('total_tokens', 0):,} total"
                  f" (prompt: {tu.get('prompt_tokens', 0):,},"
                  f" completion: {tu.get('completion_tokens', 0):,})")

    elif isinstance(result, dict):
        # Legacy dict format -- print key fields if present
        print(f"Benchmark: {result.get('benchmark', 'N/A')}")
        print(f"Total ideas tested: {result.get('total_ideas_tested', 0)}")

        if result.get("best_idea"):
            bi = result["best_idea"]
            print(f"Best Idea: {bi.get('Title', 'Untitled')}")

        if "token_usage" in result:
            tu = result["token_usage"]
            print(f"Total LLM calls: {tu.get('total_calls', 0)}")
            print(f"Total tokens: {tu.get('total_tokens', 0):,}")

    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Freeze controller-side stochastic choices and pass the hash seed to every
    # validation/test subprocess. Provider-side determinism is provider-specific,
    # so repeated trials remain the statistical unit even when this seed is fixed.
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    os.environ["FML_EXPERIMENT_SEED"] = str(args.seed)
    try:
        import numpy as np
        np.random.seed(args.seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    # Load config: merge agent config + task config
    try:
        agent_cfg = load_config(args.agent_config)
        task_cfg = load_config(args.task_config)
        config = {
            "agent": agent_cfg.get("agent", {}),
            "benchmark": task_cfg.get("benchmark", {}),
            "metrics": task_cfg.get("metrics", {}),
            "experiment": {"seed": args.seed},
            "budget_profile": args.budget_profile,
            "reproducibility_contract": {
                "mode": "repeat_guarded",
                "metric_absolute_tolerance": 0.01,
                "require_same_constraint_classification": True,
                "no_favourable_tie_breaker": True,
            },
        }
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error: Invalid YAML in configuration file: {e}")
        sys.exit(1)

    apply_overrides(config, args)
    # Freeze before agent initialization so the summary identifies the code
    # that authorized the run, not whatever happens to be present afterward.
    config["_harness_source_manifest"] = build_harness_source_manifest()

    # Validate benchmark name
    benchmark_name = config.get("benchmark", {}).get("name")
    if not benchmark_name:
        print("Error: Benchmark name not specified in task config")
        sys.exit(1)

    # Create agent
    agent_config = get_agent_config(config)
    agent_type_str = config["agent"].get("type", "theaiscientist")
    try:
        agent = AgentRegistry.create(agent_config.agent_type, agent_config)
    except ValueError as e:
        if "not registered" in str(e):
            print(f"Error: Agent type '{agent_type_str}' is not yet implemented.")
            print(f"Available agents: {[a.value for a in AgentRegistry.list_agents()]}")
            sys.exit(1)
        raise

    eval_backend = args.eval_backend or os.environ.get("FMLBENCH_EVAL_BACKEND", "local")
    ssh_host = args.ssh_host or os.environ.get("FML_SSH_HOST", "ubuntu-heshi")
    remote_project_root = args.remote_project_root or os.environ.get(
        "FML_SSH_REMOTE_ROOT", "/media/heshi/game/fml-scientist/repo"
    )
    if agent_config.provider == "CodexCLI":
        os.environ.setdefault(
            "FML_CODEX_AUDIT_DIR",
            os.path.abspath(os.path.join(args.output_dir, "codex_cli_audit")),
        )

    print(f"Initializing {agent_type_str} agent...")
    agent.initialize()

    # Run benchmark
    print(f"\n=== Running {agent_type_str} on {benchmark_name} ===")
    print(f"Model: {agent_config.model} (Provider: {agent_config.provider})")
    print(f"Evaluation backend: {eval_backend}")

    runner = BenchmarkRunner(benchmark_name, agent, workspace_label=args.workspace_label,
                              output_dir=args.output_dir,
                              save_code_backup=args.save_code_backup,
                              eval_backend=eval_backend,
                              ssh_host=ssh_host,
                              remote_project_root=remote_project_root,
                              require_gpu_idle=True,
                              keep_remote_workspace=args.keep_remote_workspace)

    # Register signal handlers to kill experiment subprocesses on external kill
    def _cleanup_on_signal(signum, frame):
        sig_name = signal.Signals(signum).name
        print(f"\nReceived {sig_name}, cleaning up...")
        agent.kill_running_process()
        runner.cleanup_workspace()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _cleanup_on_signal)
    signal.signal(signal.SIGINT, _cleanup_on_signal)

    try:
        result = runner.run()
        save_results(result, config, runner)
        print_summary(result)
        agent.cleanup()
        print("\nDone!")
    finally:
        print("Cleaning up copied workspace...")
        agent.kill_running_process()
        runner.cleanup_workspace()


if __name__ == "__main__":
    main()
