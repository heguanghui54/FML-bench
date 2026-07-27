"""SSH validation/test backend for an external-disk Ubuntu FML runner.

The research-agent loop and Codex CLI stay on the controller Mac.  Only task
validation/test commands execute on Ubuntu.  A fresh remote workspace is copied
from the prepared task template, controller code edits are overlaid with rsync,
and result JSON is pulled back into the ordinary BenchmarkExecutor artifact
layout.  Formal GPU tasks refuse to start while another compute process exists.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import os.path as osp
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Optional

from benchmark.executor import BenchmarkExecutor, SubprocessResult, TEST_OUTPUT, VAL_OUTPUT


CPU_ONLY_TASKS = {
    "Causality_gcastle",
    "Fairness_fairlearn",
}


class SSHExecutor(BenchmarkExecutor):
    """Run task commands on a named SSH host while retaining local agent state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        backend = self.config.get("_execution_backend", {})
        self.ssh_host = backend.get("ssh_host") or os.environ.get("FML_SSH_HOST", "ubuntu-heshi")
        self.remote_project_root = backend.get("remote_project_root") or os.environ.get(
            "FML_SSH_REMOTE_ROOT", "/media/heshi/game/fml-scientist/repo"
        )
        self.remote_base = str(PurePosixPath(self.remote_project_root).parent)
        self.require_gpu_idle = bool(backend.get("require_gpu_idle", True))
        self.keep_remote_workspace = bool(backend.get("keep_remote_workspace", False))
        self._remote_initialized = False
        self._validate_remote_root()

        parts = Path(self.repo_dir).parts
        if len(parts) < 3 or parts[0] != "workspace" or "__" not in parts[1]:
            raise ValueError(
                "SSHExecutor requires a unique controller workspace path like "
                "workspace/<task>__<run>/<repo>"
            )
        self._local_run_base = Path(parts[0]) / parts[1]
        task_template = parts[1].split("__", 1)[0]
        repo_subpath = PurePosixPath(*parts[2:])
        self.remote_template_base = str(
            PurePosixPath(self.remote_project_root) / "workspace" / task_template
        )
        self.remote_run_base = str(
            PurePosixPath(self.remote_project_root) / "workspace" / parts[1]
        )
        self.remote_repo_dir = str(PurePosixPath(self.remote_run_base) / repo_subpath)
        self.remote_pid_file = str(PurePosixPath(self.remote_run_base) / ".fml_remote_pid")

    @property
    def _ssh_prefix(self) -> list[str]:
        return [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
            self.ssh_host,
        ]

    def _validate_remote_root(self) -> None:
        root = PurePosixPath(self.remote_project_root)
        if not root.is_absolute() or ".." in root.parts or str(root) in {"/", "/media", "/media/heshi/game"}:
            raise ValueError(f"Unsafe remote project root: {root}")
        external_prefix = PurePosixPath("/media/heshi/game")
        if external_prefix not in root.parents:
            raise ValueError(
                f"Remote project root must be below the external disk {external_prefix}: {root}"
            )

    def _ssh_script(self, script: str, *, timeout: Optional[int] = 120) -> subprocess.CompletedProcess:
        return subprocess.run(
            [*self._ssh_prefix, "bash", "-s"],
            input=script,
            text=True,
            capture_output=True,
            timeout=timeout,
        )

    def setup_workspace(self) -> str:
        workspace = super().setup_workspace()
        script = f"""
set -euo pipefail
template={shlex.quote(self.remote_template_base)}
destination={shlex.quote(self.remote_run_base)}
root={shlex.quote(self.remote_project_root)}
test -d "$root/ml_tasks"
test -d "$template"
case "$destination" in "$root"/workspace/*__*) ;; *) echo unsafe-destination >&2; exit 90;; esac
test ! -e "$destination"
cp -a --reflink=auto "$template" "$destination"
"""
        result = self._ssh_script(script, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(
                f"Unable to create remote workspace {self.remote_run_base}: {result.stderr.strip()}"
            )
        self._remote_initialized = True
        return workspace

    def _rsync_command(self, source: str, destination: str) -> list[str]:
        ssh_transport = (
            "ssh -o BatchMode=yes -o ConnectTimeout=8 "
            "-o ServerAliveInterval=30 -o ServerAliveCountMax=3"
        )
        return [
            "rsync",
            "-a",
            "--exclude=.git/",
            "--exclude=results_tmp/",
            "--exclude=__pycache__/",
            "-e",
            ssh_transport,
            source,
            destination,
        ]

    def _sync_controller_workspace(self) -> None:
        source = str(self._local_run_base.resolve()) + "/"
        destination = f"{self.ssh_host}:{self.remote_run_base}/"
        result = subprocess.run(
            self._rsync_command(source, destination),
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Controller-to-Ubuntu rsync failed: {result.stderr.strip()}")

    def _pull_results(self) -> None:
        local_results = Path(self.repo_dir).resolve() / "results_tmp"
        if local_results.exists():
            shutil.rmtree(local_results)
        local_results.mkdir(parents=True, exist_ok=True)
        source = f"{self.ssh_host}:{self.remote_repo_dir}/results_tmp/"
        ssh_transport = (
            "ssh -o BatchMode=yes -o ConnectTimeout=8 "
            "-o ServerAliveInterval=30 -o ServerAliveCountMax=3"
        )
        result = subprocess.run(
            ["rsync", "-a", "-e", ssh_transport, source, str(local_results) + "/"],
            text=True,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Ubuntu-to-controller result rsync failed: {result.stderr.strip()}")

    def _gpu_processes(self) -> str:
        script = """
set -euo pipefail
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null || true
"""
        result = self._ssh_script(script, timeout=30)
        return result.stdout.strip()

    def _assert_gpu_idle(self) -> None:
        if not self.require_gpu_idle or self.benchmark_name in CPU_ONLY_TASKS:
            return
        processes = self._gpu_processes()
        if processes:
            raise RuntimeError(
                "GPU isolation gate failed; stop i4h or other compute jobs before this FML task. "
                f"Active compute processes: {processes}"
            )

    def _platform_snapshot(self) -> dict:
        script = """
set -euo pipefail
printf 'host='; hostname
printf 'os='; . /etc/os-release; printf '%s %s\n' "$NAME" "$VERSION_ID"
printf 'arch='; uname -m
printf 'gpu='; nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu --format=csv,noheader 2>/dev/null || true
printf 'compute='; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || true
"""
        result = self._ssh_script(script, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _remote_command(self, command: str) -> SubprocessResult:
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        conda = str(PurePosixPath(self.remote_base) / "miniforge3" / "bin" / "conda")
        envs = str(PurePosixPath(self.remote_base) / "conda-envs")
        packages = str(PurePosixPath(self.remote_base) / "conda-pkgs")
        cache = str(PurePosixPath(self.remote_base) / "cache")
        tmp = str(PurePosixPath(self.remote_base) / "tmp")
        timeout_prefix = ""
        if self.timeout is not None:
            timeout_prefix = f"timeout --signal=TERM --kill-after=60s {int(self.timeout)}s "
        script = f"""
set -uo pipefail
repo={shlex.quote(self.remote_repo_dir)}
pidfile={shlex.quote(self.remote_pid_file)}
conda={shlex.quote(conda)}
export PATH={shlex.quote(str(PurePosixPath(self.remote_base) / 'miniforge3' / 'bin'))}:"$PATH"
export CONDA_ENVS_PATH={shlex.quote(envs)}
export CONDA_PKGS_DIRS={shlex.quote(packages)}
export HF_HOME={shlex.quote(str(PurePosixPath(cache) / 'huggingface'))}
export TORCH_HOME={shlex.quote(str(PurePosixPath(cache) / 'torch'))}
export XDG_CACHE_HOME={shlex.quote(str(PurePosixPath(cache) / 'xdg'))}
export PIP_CACHE_DIR={shlex.quote(str(PurePosixPath(cache) / 'pip'))}
export TMPDIR={shlex.quote(tmp)}
export PYTHONNOUSERSITE=1
mkdir -p "$TMPDIR"
test -x "$conda"
test -d "$repo/.git"
cd "$repo"
decoded=$(printf %s {shlex.quote(encoded)} | base64 -d)
setsid {timeout_prefix}"$conda" run --no-capture-output -n {shlex.quote(self.conda_env)} bash -c "$decoded" &
pid=$!
printf '%s\n' "$pid" > "$pidfile"
wait "$pid"
rc=$?
rm -f "$pidfile"
exit "$rc"
"""
        print(f"Running on {self.ssh_host}: conda run -n {self.conda_env} bash -c <command>")
        proc = subprocess.Popen(
            [*self._ssh_prefix, "bash", "-s"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setpgrp,
        )
        self._current_proc = proc
        local_timeout = None if self.timeout is None else int(self.timeout) + 120
        try:
            stdout, stderr = proc.communicate(script, timeout=local_timeout)
            self._current_proc = None
            return SubprocessResult(proc.returncode, stdout=stdout, stderr=stderr)
        except subprocess.TimeoutExpired:
            self._kill_remote_process()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
            self._current_proc = None
            return SubprocessResult(1, stderr=f"Remote timeout after {self.timeout} seconds")

    def _remote_integrity_error(self) -> Optional[str]:
        checks = [f"test -d {shlex.quote(self.remote_repo_dir + '/.git')}"]
        for target in self.config.get("target_files", []):
            remote_target = str(PurePosixPath(self.remote_repo_dir) / target)
            checks.append(f"test -e {shlex.quote(remote_target)}")
        result = self._ssh_script("set -e\n" + "\n".join(checks) + "\n", timeout=30)
        if result.returncode != 0:
            return f"remote workspace integrity check failed: {result.stderr.strip()}"
        return None

    def _run_phase(self, run_id: int, command_key: str, output_path: str, backup: bool = True) -> dict:
        command = self.config.get(command_key)
        if not command:
            if "execute_commands" in self.config:
                command = self.config["execute_commands"][0]
            else:
                return {
                    "success": False,
                    "results": None,
                    "primary_metric": None,
                    "error": f"No '{command_key}' or 'execute_commands' specified in config",
                }

        local_results = Path(self.repo_dir).resolve() / "results_tmp"
        if local_results.exists():
            shutil.rmtree(local_results)
        execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_dir = Path(self.workspace_dir) / f"run_{run_id}" / f"execution_{execution_timestamp}"
        execution_dir.mkdir(parents=True, exist_ok=True)
        if backup and self.config.get("save_code_backup", False):
            self._backup_files(run_id, execution_timestamp)

        self._assert_gpu_idle()
        self._sync_controller_workspace()
        before = self._platform_snapshot()
        prep = self._ssh_script(
            f"rm -rf -- {shlex.quote(self.remote_repo_dir + '/results_tmp')}\n"
            f"chmod 555 {shlex.quote(self.remote_repo_dir + '/.git')} 2>/dev/null || true\n",
            timeout=60,
        )
        if prep.returncode != 0:
            return {"success": False, "results": None, "primary_metric": None, "error": prep.stderr}

        result = SubprocessResult(0)
        try:
            if self.config.get("do_preprocess") and self.config.get("preprocess_commands"):
                for pre_cmd in self.config["preprocess_commands"]:
                    result = self._remote_command(pre_cmd)
                    if result.returncode != 0:
                        break
            if result.returncode == 0:
                result = self._remote_command(command)
            if result.returncode == 0 and self.config.get("do_postprocess") and self.config.get("postprocess_commands"):
                for post_cmd in self.config["postprocess_commands"]:
                    result = self._remote_command(post_cmd)
                    if result.returncode != 0:
                        break
        finally:
            self._ssh_script(
                f"chmod 755 {shlex.quote(self.remote_repo_dir + '/.git')} 2>/dev/null || true\n",
                timeout=30,
            )

        integrity_error = self._remote_integrity_error()
        after = self._platform_snapshot()
        evidence = {
            "schema_version": "fml-ssh-execution-v1",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ssh_host": self.ssh_host,
            "remote_project_root": self.remote_project_root,
            "remote_workspace": self.remote_run_base,
            "remote_repo_dir": self.remote_repo_dir,
            "benchmark": self.benchmark_name,
            "agent": self.agent_name,
            "conda_env": self.conda_env,
            "phase": command_key,
            "command": command,
            "command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "platform_before": before,
            "platform_after": after,
            "integrity_error": integrity_error,
        }
        (execution_dir / "ssh_execution.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        if integrity_error:
            return {"success": False, "results": None, "primary_metric": None, "error": integrity_error}
        if result.returncode != 0:
            phase = "val" if command_key == "val_command" else "test"
            self._save_bug_execution_record(run_id, execution_timestamp, phase, result)
            return {"success": False, "results": None, "primary_metric": None, "error": result.stderr}

        try:
            self._pull_results()
        except RuntimeError as exc:
            return {"success": False, "results": None, "primary_metric": None, "error": str(exc)}
        actual_output_path = output_path
        if not (Path(self.repo_dir).resolve() / output_path).exists():
            fallback = Path(self.config.get("exp_running_dir", "results_tmp/")) / "final_info.json"
            if (Path(self.repo_dir).resolve() / fallback).exists():
                actual_output_path = str(fallback)
        return self._collect_results(run_id, execution_timestamp, actual_output_path)

    def run_val(self, run_id: int) -> dict:
        return self._run_phase(run_id, "val_command", VAL_OUTPUT)

    def run_test(self, run_id: int) -> dict:
        return self._run_phase(run_id, "test_command", TEST_OUTPUT, backup=False)

    def _kill_remote_process(self) -> None:
        script = f"""
set -u
pidfile={shlex.quote(self.remote_pid_file)}
if test -f "$pidfile"; then
  pid=$(cat "$pidfile")
  case "$pid" in (*[!0-9]*|'') exit 91;; esac
  kill -TERM -- "-$pid" 2>/dev/null || true
  sleep 2
  kill -KILL -- "-$pid" 2>/dev/null || true
  rm -f "$pidfile"
fi
"""
        try:
            self._ssh_script(script, timeout=15)
        except (OSError, subprocess.SubprocessError):
            pass

    def kill_running_process(self):
        self._kill_remote_process()
        super().kill_running_process()

    def cleanup(self):
        self.kill_running_process()
        super().cleanup()
        if not self._remote_initialized or self.keep_remote_workspace:
            return
        script = f"""
set -euo pipefail
root={shlex.quote(self.remote_project_root)}
target={shlex.quote(self.remote_run_base)}
case "$target" in "$root"/workspace/*__*) ;; *) echo unsafe-cleanup >&2; exit 90;; esac
test ! -L "$target"
rm -rf -- "$target"
"""
        result = self._ssh_script(script, timeout=600)
        if result.returncode != 0:
            print(f"Warning: failed to remove remote workspace: {result.stderr.strip()}")
        else:
            self._remote_initialized = False
