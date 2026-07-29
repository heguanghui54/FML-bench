"""
Main class for managing benchmark execution workflow with agent abstraction support.
"""
import json
import os
import os.path as osp
import shutil
import time
import hashlib
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from agents.base import AgentResult, BaseAgent
from ml_scientist.execution_contracts import (
    BudgetLedger,
    MATCHED_BUDGET_V1,
    activate_budget_ledger,
)


def collect_persisted_execution_contracts(parent_workspace: str) -> dict:
    """Merge every executor phase after agents have already cleaned them up."""
    root = Path(parent_workspace)
    paths = sorted(root.rglob("execution_contract_summary.json")) if root.is_dir() else []
    merged = {
        "schema_version": "fml-merged-execution-contract-summary-v1",
        "execution_counts": {
            "candidate_validation_count": 0,
            "pre_test_validation_count": 0,
            "protected_test_count": 0,
        },
        "activation_records": [],
        "reproducibility_records": [],
        "gpu_telemetry_summaries": [],
        "executor_summaries": [],
    }
    safety_contracts = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, value in payload.get("execution_counts", {}).items():
            merged["execution_counts"][key] = merged["execution_counts"].get(key, 0) + value
        for key in ("activation_records", "reproducibility_records", "gpu_telemetry_summaries"):
            merged[key].extend(payload.get(key, []))
        if payload.get("gpu_safety_contract"):
            safety_contracts.append(payload["gpu_safety_contract"])
        merged["executor_summaries"].append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    # Older post-fix runs can contain all immutable per-phase artifacts but no
    # executor summary because cleanup occurred after the executor reference was
    # released. Recover only when no summary exists, and preserve source hashes.
    if not paths and root.is_dir():
        activation_paths = sorted(root.rglob("runtime_activation_validation.json"))
        telemetry_paths = sorted(root.rglob("gpu_telemetry_summary.json"))
        for path in activation_paths:
            payload = json.loads(path.read_text(encoding="utf-8"))
            merged["activation_records"].append(payload)
            phase = str(payload.get("phase") or "")
            path_text = str(path)
            if "run_pre_test_val" in path_text or phase == "pre_test_val":
                key = "pre_test_validation_count"
            elif "run_final_test" in path_text or phase == "test":
                key = "protected_test_count"
            else:
                key = "candidate_validation_count"
            merged["execution_counts"][key] += 1
            merged["executor_summaries"].append({
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "recovered_from_materialized_artifact": True,
            })
        for path in telemetry_paths:
            merged["gpu_telemetry_summaries"].append(
                json.loads(path.read_text(encoding="utf-8"))
            )
        for path in sorted(root.rglob("runtime_reproducibility.json")):
            merged["reproducibility_records"].append({
                "schema_version": "fml-runtime-reproducibility-artifact-reference-v1",
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    if safety_contracts:
        canonical = json.dumps(safety_contracts[0], sort_keys=True)
        merged["gpu_safety_contract"] = safety_contracts[0]
        merged["gpu_safety_contract_consistent"] = all(
            json.dumps(value, sort_keys=True) == canonical for value in safety_contracts
        )
    return merged


class BenchmarkRunner:
    """
    Simplified BenchmarkRunner that delegates the entire execution pipeline to agents.

    Usage example:
        from agents import AgentConfig, AgentType, AgentRegistry

        config = AgentConfig(
            agent_type=AgentType.THEAISCIENTIST,
            model="gpt-4",
            ...
        )
        agent = AgentRegistry.create(AgentType.THEAISCIENTIST, config)
        agent.initialize()

        runner = BenchmarkRunner("cider", agent)
        results = runner.run()
    """

    def __init__(self, benchmark_name: str, agent: BaseAgent, workspace_label: str = None,
                 output_dir: str = "benchmark_results", save_code_backup: bool = False,
                 eval_backend: str = "local", ssh_host: str = "ubuntu-heshi",
                 remote_project_root: str = "/media/heshi/game/fml-scientist/repo",
                 require_gpu_idle: bool = True, keep_remote_workspace: bool = False):
        self.benchmark_name = benchmark_name
        self.agent = agent
        self.agent_name = agent.config.agent_type.value
        self.workspace_label = workspace_label
        self.output_dir = output_dir
        self.save_code_backup = save_code_backup
        self.eval_backend = eval_backend
        self.ssh_host = ssh_host
        self.remote_project_root = remote_project_root
        self.require_gpu_idle = require_gpu_idle
        self.keep_remote_workspace = keep_remote_workspace
        self._workspace_copy_path = None
        self.config = self._load_config()
        self._setup_workspace(workspace_label)
        try:
            self.task_description = self._load_task_description()
            self.baseline_results = self._load_baseline()
        except Exception:
            self.cleanup_workspace()
            raise

    def run(self) -> dict:
        """Run the benchmark. Passes task info to agent, returns results."""
        print(f"Starting benchmark: {self.benchmark_name}")

        try:
            # 1. Prepare target files with full paths
            target_files = [osp.join(self.config["repo_dir"], f)
                            for f in self.config["target_files"]]

            # 2. Inject metrics filter into benchmark_config so executor can use it
            metrics_cfg = self.agent.config.runtime_params.get("metrics", {})
            if metrics_cfg.get("include_datasets"):
                self.config["include_datasets"] = metrics_cfg["include_datasets"]

            # 2b. Propagate the code-backup toggle into benchmark_config so the
            # executor (constructed by every agent from benchmark_config) can read it.
            self.config["save_code_backup"] = self.save_code_backup
            self.config["_execution_backend"] = {
                "name": self.eval_backend,
                "ssh_host": self.ssh_host,
                "remote_project_root": self.remote_project_root,
                "require_gpu_idle": self.require_gpu_idle,
                "keep_remote_workspace": self.keep_remote_workspace,
            }
            self.config["_experimental_seed"] = (
                self.agent.config.runtime_params.get("experiment") or {}
            ).get("seed")

            # 3. Inject runtime params into agent config
            self.agent.config.runtime_params.update({
                "repo_dir": self.config["repo_dir"],
                "base_dir": osp.join("ml_tasks", self.benchmark_name),
                "benchmark_config": self.config,
                "agent_name": self.agent_name,
                "benchmark_name": self.benchmark_name,
                "task_description": self.task_description,
                "workspace_label": self.workspace_label,
                "output_dir": self.output_dir,
                "eval_backend": self.eval_backend,
            })

            # 3. Run agent (with wall-clock timing)
            t0 = time.monotonic()
            profile_name = str(
                self.agent.config.runtime_params.get("budget_profile")
                or self.agent.config.agent_params.get("budget_profile")
                or "legacy-unbounded"
            )
            if profile_name == "matched-v1":
                overrides = self.agent.config.runtime_params.get("budget_limits", {})
                limits = {**MATCHED_BUDGET_V1, **dict(overrides)}
                ledger = BudgetLedger(profile_name=profile_name, limits=limits)
            else:
                ledger = BudgetLedger.legacy_unbounded()
            self.agent.budget_ledger = ledger
            with activate_budget_ledger(ledger):
                result = self.agent.run(
                    task_description=self.task_description,
                    target_files=target_files,
                    baseline_results=self.baseline_results,
                )
            wall_clock = time.monotonic() - t0

            # Attach wall-clock duration
            if isinstance(result, AgentResult):
                result.total_duration_seconds = wall_clock
                result.metadata.setdefault("execution_contracts", {})["budget_ledger"] = ledger.snapshot()
                if self.agent.executor is not None:
                    result.metadata["execution_contracts"].update(
                        self.agent.executor.execution_contract_summary()
                    )
                persisted = collect_persisted_execution_contracts(result.parent_workspace)
                if persisted["executor_summaries"]:
                    result.metadata["execution_contracts"].update(persisted)
            elif isinstance(result, dict):
                result["total_duration_seconds"] = wall_clock
                result.setdefault("metadata", {}).setdefault("execution_contracts", {})["budget_ledger"] = ledger.snapshot()
                if self.agent.executor is not None:
                    result["metadata"]["execution_contracts"].update(
                        self.agent.executor.execution_contract_summary()
                    )
                parent_workspace = result.get("parent_workspace", "")
                persisted = collect_persisted_execution_contracts(parent_workspace)
                if persisted["executor_summaries"]:
                    result["metadata"]["execution_contracts"].update(persisted)

            # 4. Add benchmark metadata
            if isinstance(result, dict):
                result["benchmark"] = self.benchmark_name
                result["agent"] = self.agent_name

            print(f"Benchmark completed: {self.benchmark_name}")
            return result

        except Exception as e:
            print(f"Error during benchmark execution: {e}")
            raise

    def _load_config(self) -> dict:
        """Load config.json. Support both new 8-field and old format."""
        config_path = osp.join("ml_tasks", self.benchmark_name, "config.json")
        if not osp.exists(config_path):
            raise FileNotFoundError(f"Configuration not found: {config_path}")

        with open(config_path, 'r') as f:
            config = json.load(f)

        # New format: has val_command
        if "val_command" in config:
            required = ["repo_dir", "pinned_commit", "conda_env", "target_files",
                        "val_command", "test_command", "metric", "metric_direction"]
            for field in required:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in {config_path}")
            return config

        # Old format: backward compatibility
        if "execute_commands" in config:
            for field in ["repo_dir", "target_files", "execute_commands"]:
                if field not in config:
                    raise ValueError(f"Missing required field '{field}' in {config_path}")
            return config

        raise ValueError(f"config.json must have either 'val_command' (new) or 'execute_commands' (old)")

    def _load_task_description(self) -> str:
        """Load task description from prompt.json or task_description.txt."""
        # Try prompt.json first
        prompt_file = osp.join("ml_tasks", self.benchmark_name, "prompt.json")
        if osp.exists(prompt_file):
            with open(prompt_file, 'r') as f:
                prompt_json = json.load(f)

            parts = []
            if "system" in prompt_json:
                parts.append(prompt_json["system"])
            if "task_description" in prompt_json:
                parts.append(prompt_json["task_description"])

            if not parts:
                raise ValueError(f"No task description found in {prompt_file}")

            return "\n".join(parts)

        # Fallback to task_description.txt
        txt_file = osp.join("ml_tasks", self.benchmark_name, "task_description.txt")
        if osp.exists(txt_file):
            with open(txt_file, 'r') as f:
                return f.read().strip()

        raise FileNotFoundError(
            f"No prompt.json or task_description.txt in ml_tasks/{self.benchmark_name}/"
        )

    def _load_baseline(self) -> dict:
        """Load baseline results. Try val_info.json first, fallback to final_info.json."""
        # Try new format
        val_path = osp.join("ml_tasks", self.benchmark_name, "baseline_results", "val_info.json")
        if osp.exists(val_path):
            with open(val_path, 'r') as f:
                return json.load(f)

        # Fallback to old format
        old_path = osp.join("ml_tasks", self.benchmark_name, "baseline_results", "final_info.json")
        if osp.exists(old_path):
            with open(old_path, 'r') as f:
                return json.load(f)

        print(f"Warning: No baseline results found for {self.benchmark_name}")
        return {}

    def _setup_workspace(self, label=None):
        """Copy template workspace to a unique directory for this run."""
        repo_dir = self.config["repo_dir"]

        if os.path.isabs(repo_dir):
            raise ValueError(f"repo_dir must be a relative path, got: {repo_dir}")

        # repo_dir format: "workspace/<task_name>/<repo_name>"
        parts = repo_dir.split(os.sep)
        if len(parts) < 3:
            raise ValueError(
                f"repo_dir must have at least 3 parts (workspace/task/repo), got: {repo_dir}"
            )

        template_base = os.path.join(parts[0], parts[1])  # e.g. "workspace/Fairness_and_Bias_aif360"
        repo_subpath = os.sep.join(parts[2:])              # e.g. "AIF360"

        # Generate unique suffix: {optional_label}_{timestamp}_{uuid8}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = uuid.uuid4().hex[:8]
        if label:
            suffix = f"{label}_{timestamp}_{unique_id}"
        else:
            suffix = f"{timestamp}_{unique_id}"

        suffixed_base = f"{template_base}__{suffix}"

        # Safety: error if exists (should never happen with UUID)
        if os.path.exists(suffixed_base):
            raise RuntimeError(f"Workspace copy already exists: {suffixed_base}")

        self._workspace_copy_path = suffixed_base
        try:
            print(f"Preparing workspace: Copying workspace from {template_base} to {suffixed_base}")
            shutil.copytree(template_base, suffixed_base)
        except Exception:
            self.cleanup_workspace()
            raise
        self.config["repo_dir"] = os.path.join(suffixed_base, repo_subpath)
        print(f"Created workspace copy: {suffixed_base}")

    def cleanup_workspace(self):
        """Delete the workspace copy created for this run."""
        if self._workspace_copy_path and os.path.exists(self._workspace_copy_path):
            try:
                shutil.rmtree(self._workspace_copy_path)
                print(f"Cleaned up workspace copy: {self._workspace_copy_path}")
            except Exception as e:
                print(f"Warning: Failed to clean up workspace copy {self._workspace_copy_path}: {e}")
