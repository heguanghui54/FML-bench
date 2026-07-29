"""SSH validation/test backend for an external-disk Ubuntu FML runner.

The research-agent loop and Codex CLI stay on the controller Mac.  Only task
validation/test commands execute on Ubuntu.  A fresh remote workspace is copied
from the prepared task template, controller code edits are overlaid with rsync,
and result JSON is pulled back into the ordinary BenchmarkExecutor artifact
layout.  Formal GPU tasks refuse to start while another compute process exists.
"""

from __future__ import annotations

import base64
import csv
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
from benchmark.runtime_probe import (
    finalize_runtime_probe,
    prepare_runtime_probe,
)
from ml_scientist.execution_contracts import (
    BudgetExceeded,
    current_budget_ledger,
    reproducibility_environment,
)


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
        self.telemetry_sample_seconds = int(backend.get("telemetry_sample_seconds", 5))
        self.gpu_warning_temperature_c = float(backend.get("gpu_warning_temperature_c", 82.0))
        self.gpu_abort_temperature_c = float(backend.get("gpu_abort_temperature_c", 88.0))
        self.thermal_throttle_abort_samples = int(backend.get("thermal_throttle_abort_samples", 3))
        self.gpu_startup_timeout_seconds = int(backend.get("gpu_startup_timeout_seconds", 900))
        if self.gpu_startup_timeout_seconds <= 0:
            raise ValueError("gpu_startup_timeout_seconds must be positive")
        # Consumer GPUs can reach their firmware thermal-throttle point well
        # below the emergency abort temperature during long benchmark jobs.
        # Proactively pause the experiment process group, while leaving the
        # harness-owned sampler running, so the scientific computation is
        # preserved without weakening the hard safety gates.
        self.thermal_pacing_enabled = bool(backend.get("thermal_pacing_enabled", True))
        self.thermal_pacing_start_c = float(backend.get("thermal_pacing_start_c", 78.0))
        self.thermal_pacing_resume_c = float(backend.get("thermal_pacing_resume_c", 72.0))
        if self.thermal_pacing_resume_c >= self.thermal_pacing_start_c:
            raise ValueError("thermal_pacing_resume_c must be below thermal_pacing_start_c")
        if self.thermal_pacing_start_c >= self.gpu_abort_temperature_c:
            raise ValueError("thermal_pacing_start_c must be below gpu_abort_temperature_c")
        self._remote_initialized = False
        self._remote_telemetry_path: str | None = None
        self._remote_safety_path: str | None = None
        self._controller_project_root = Path(__file__).resolve().parents[1]
        self._remote_harness_attestation: dict | None = None
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
        self.remote_harness_task_dir = str(
            PurePosixPath(self.remote_run_base)
            / ".fml_harness"
            / "ml_tasks"
            / self.benchmark_name
        )

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
        self._remote_harness_attestation = self._sync_controller_harness()
        return workspace

    @staticmethod
    def _file_manifest(root: Path) -> dict[str, str]:
        excluded = {"__pycache__", "results_tmp", "baseline_results", ".git"}
        manifest: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(root)
            if not path.is_file() or any(part in excluded for part in relative.parts):
                continue
            manifest[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return manifest

    def _sync_controller_harness(self) -> dict:
        """Freeze task harness files inside this run rather than the shared repo."""
        local_root = self._controller_project_root / "ml_tasks" / self.benchmark_name
        if not local_root.is_dir():
            raise RuntimeError(f"Controller task harness is missing: {local_root}")
        local_manifest = self._file_manifest(local_root)
        mkdir = self._ssh_script(
            f"mkdir -p {shlex.quote(self.remote_harness_task_dir)}\n", timeout=30
        )
        if mkdir.returncode != 0:
            raise RuntimeError(f"Unable to create remote harness overlay: {mkdir.stderr.strip()}")
        command = self._rsync_command(
            str(local_root.resolve()) + "/",
            f"{self.ssh_host}:{self.remote_harness_task_dir}/",
        )
        command[-2:-2] = ["--delete", "--exclude=baseline_results/"]
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"Controller harness rsync failed: {result.stderr.strip()}")
        remote = self._ssh_script(
            "set -euo pipefail\n"
            f"cd {shlex.quote(self.remote_harness_task_dir)}\n"
            "find . -type f -not -path '*/__pycache__/*' -not -path './results_tmp/*' "
            "-not -path './baseline_results/*' -print0 | sort -z | "
            "xargs -0 -r sha256sum\n",
            timeout=120,
        )
        if remote.returncode != 0:
            raise RuntimeError(f"Remote harness hashing failed: {remote.stderr.strip()}")
        remote_manifest: dict[str, str] = {}
        for line in remote.stdout.splitlines():
            digest, path = line.split(maxsplit=1)
            remote_manifest[path.removeprefix("./")] = digest
        attestation = {
            "schema_version": "fml-remote-harness-attestation-v1",
            "benchmark": self.benchmark_name,
            "remote_overlay": self.remote_harness_task_dir,
            "file_count": len(local_manifest),
            "local_manifest": local_manifest,
            "remote_manifest": remote_manifest,
            "passed": local_manifest == remote_manifest,
        }
        if not attestation["passed"]:
            raise RuntimeError("Remote harness attestation failed after rsync")
        return attestation

    def _resolve_harness_paths(self, command: str) -> str:
        """Point task-harness references at the immutable per-run overlay."""
        resolved = command
        for depth in range(6, 0, -1):
            token = "../" * depth + f"ml_tasks/{self.benchmark_name}"
            resolved = resolved.replace(token, self.remote_harness_task_dir)
        return resolved

    def _attest_remote_targets(self) -> dict:
        local_manifest: dict[str, str] = {}
        commands: list[str] = ["set -euo pipefail"]
        for relative in self.config.get("target_files", []):
            local = Path(self.repo_dir).resolve() / relative
            local_manifest[relative] = (
                hashlib.sha256(local.read_bytes()).hexdigest() if local.is_file() else "MISSING"
            )
            remote = str(PurePosixPath(self.remote_repo_dir) / relative)
            commands.append(
                f"printf '%s\\t' {shlex.quote(relative)}; "
                f"sha256sum {shlex.quote(remote)} | awk '{{print $1}}'"
            )
        result = self._ssh_script("\n".join(commands) + "\n", timeout=60)
        remote_manifest: dict[str, str] = {}
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                relative, digest = line.split("\t", 1)
                remote_manifest[relative] = digest.strip()
        return {
            "schema_version": "fml-remote-target-attestation-v1",
            "local_manifest": local_manifest,
            "remote_manifest": remote_manifest,
            "passed": result.returncode == 0 and local_manifest == remote_manifest,
            "stderr": result.stderr.strip(),
        }

    @staticmethod
    def _wrap_experiment_environment(
        command: str,
        reproducibility: dict[str, str],
        *,
        runtime_probe: bool,
    ) -> str:
        """Export experiment variables for the complete shell command chain."""
        exports = [
            f"export {key}={shlex.quote(str(value))}"
            for key, value in reproducibility.items()
        ]
        if runtime_probe:
            exports.extend([
                'export PYTHONPATH="$PWD"/.fml_runtime_probe:${PYTHONPATH:-}',
                'export FML_RUNTIME_MARKER_PATH=results_tmp/runtime_activation.jsonl',
            ])
        return "; ".join([*exports, command])

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
        conda = str(PurePosixPath(self.remote_base) / "miniforge3" / "bin" / "conda")
        envs = str(PurePosixPath(self.remote_base) / "conda-envs")
        script = f"""
set -euo pipefail
conda={shlex.quote(conda)}
env_name={shlex.quote(self.conda_env)}
export CONDA_ENVS_PATH={shlex.quote(envs)}
printf 'host='; hostname
printf 'os='; . /etc/os-release; printf '%s %s\n' "$NAME" "$VERSION_ID"
printf 'arch='; uname -m
printf 'gpu='; nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version,temperature.gpu --format=csv,noheader 2>/dev/null || true
printf 'compute='; nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv,noheader 2>/dev/null | tr '\n' ';' || true
printf 'python='; "$conda" run --no-capture-output -n "$env_name" python --version 2>&1 || true
printf 'cuda='; nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 || true
printf 'frameworks='; "$conda" run --no-capture-output -n "$env_name" python - <<'PY' 2>/dev/null || true
import json
try:
    import torch
    row = {{"torch": torch.__version__, "cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version(),
           "deterministic": torch.are_deterministic_algorithms_enabled(),
           "cudnn_benchmark": torch.backends.cudnn.benchmark}}
except Exception as exc:
    row = {{"torch_error": str(exc)}}
print(json.dumps(row, sort_keys=True))
PY
"""
        result = self._ssh_script(script, timeout=30)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def _remote_command(self, command: str) -> SubprocessResult:
        command_started = time.monotonic()
        encoded = base64.b64encode(command.encode("utf-8")).decode("ascii")
        conda = str(PurePosixPath(self.remote_base) / "miniforge3" / "bin" / "conda")
        envs = str(PurePosixPath(self.remote_base) / "conda-envs")
        packages = str(PurePosixPath(self.remote_base) / "conda-pkgs")
        cache = str(PurePosixPath(self.remote_base) / "cache")
        tmp = str(PurePosixPath(self.remote_base) / "tmp")
        effective_timeout = self._effective_command_timeout()
        timeout_prefix = ""
        if effective_timeout is not None:
            timeout_prefix = (
                "timeout --signal=TERM --kill-after=60s "
                f"{max(0.001, effective_timeout):.3f}s "
            )
        telemetry = self._remote_telemetry_path or str(
            PurePosixPath(self.remote_run_base) / ".fml_gpu_telemetry.csv"
        )
        safety = self._remote_safety_path or str(
            PurePosixPath(self.remote_run_base) / ".fml_gpu_safety.json"
        )
        monitor_enabled = self.benchmark_name not in CPU_ONLY_TASKS
        script = f"""
set -uo pipefail
repo={shlex.quote(self.remote_repo_dir)}
pidfile={shlex.quote(self.remote_pid_file)}
conda={shlex.quote(conda)}
telemetry={shlex.quote(telemetry)}
safety={shlex.quote(safety)}
monitor_enabled={1 if monitor_enabled else 0}
sample_seconds={self.telemetry_sample_seconds}
warning_temperature={self.gpu_warning_temperature_c}
abort_temperature={self.gpu_abort_temperature_c}
throttle_limit={self.thermal_throttle_abort_samples}
startup_timeout={self.gpu_startup_timeout_seconds}
pacing_enabled={1 if self.thermal_pacing_enabled else 0}
pacing_start={self.thermal_pacing_start_c}
pacing_resume={self.thermal_pacing_resume_c}
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
monitor_pid=''
if test "$monitor_enabled" -eq 1; then
  mkdir -p "$(dirname "$telemetry")"
  if test ! -f "$telemetry"; then
    printf '%s\n' 'timestamp,temperature_c,power_w,utilization_pct,memory_used_mb,memory_total_mb,thermal_slowdown,thermal_pacing_state,thermal_pacing_pause_seconds,experiment_compute_active' > "$telemetry"
  fi
  monitor_gpu() {{
    throttle_strikes=0
    warning_count=0
    ever_compute_active=0
    monitor_started=$(date +%s)
    pacing_state='RUNNING'
    while kill -0 "$pid" 2>/dev/null; do
      row=$(nvidia-smi --query-gpu=timestamp,temperature.gpu,power.draw,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      throttle=$(nvidia-smi --query-gpu=clocks_event_reasons.sw_thermal_slowdown --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true)
      if test -n "$row"; then
        temperature=$(printf '%s\n' "$row" | awk -F, '{{gsub(/ /,"",$2); print $2}}')
        experiment_compute_active=0
        for compute_pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true); do
          compute_pgid=$(ps -o pgid= -p "$compute_pid" 2>/dev/null | xargs || true)
          if test "$compute_pgid" = "$pid"; then
            experiment_compute_active=1
            break
          fi
        done
        if test "$experiment_compute_active" -eq 1; then
          ever_compute_active=1
        fi
        pause_seconds=0
        if test "$pacing_enabled" -eq 1 && awk "BEGIN {{exit !($temperature >= $pacing_start)}}"; then
          kill -STOP -- "-$pid" 2>/dev/null || true
          pacing_state='PAUSED_RESUMED'
          pause_started=$(date +%s)
          cooldown_temperature=$temperature
          # Keep the scientific process stopped, but poll cooling once per
          # second so a five-second telemetry cadence does not impose a
          # five-second minimum pause on every pacing event.
          while kill -0 "$pid" 2>/dev/null && awk "BEGIN {{exit !($cooldown_temperature > $pacing_resume)}}"; do
            sleep 1
            cooldown_temperature=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | xargs || printf '%s' "$cooldown_temperature")
          done
          kill -CONT -- "-$pid" 2>/dev/null || true
          pause_seconds=$(( $(date +%s) - pause_started ))
        else
          pacing_state='RUNNING'
        fi
        printf '%s,%s,%s,%s,%s\n' "$row" "$throttle" "$pacing_state" "$pause_seconds" "$experiment_compute_active" >> "$telemetry"
        if awk "BEGIN {{exit !($temperature >= $warning_temperature)}}"; then
          warning_count=$((warning_count + 1))
        fi
        # NVIDIA reports the inactive state as the literal string "Not Active".
        # Match only complete affirmative values; a substring match on
        # "active" turns every healthy sample into a false throttle strike.
        throttle_state=$(printf '%s' "$throttle" | tr '[:upper:]' '[:lower:]' | xargs)
        case "$throttle_state" in
          active|yes|true|1) throttle_strikes=$((throttle_strikes + 1)) ;;
          *) throttle_strikes=0 ;;
        esac
        reason=''
        startup_elapsed=$(( $(date +%s) - monitor_started ))
        if awk "BEGIN {{exit !($temperature >= $abort_temperature)}}"; then
          reason='GPU_TEMPERATURE_ABORT'
        elif test "$throttle_strikes" -ge "$throttle_limit"; then
          reason='SUSTAINED_THERMAL_THROTTLING'
        elif test "$ever_compute_active" -eq 0 && test "$startup_elapsed" -ge "$startup_timeout"; then
          reason='GPU_STARTUP_TIMEOUT'
        fi
        if test -n "$reason"; then
          printf '{{"reason":"%s","temperature_c":%s,"warning_count":%s,"startup_elapsed_seconds":%s,"timestamp":"%s"}}\n' \
            "$reason" "$temperature" "$warning_count" "$startup_elapsed" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$safety"
          kill -TERM -- "-$pid" 2>/dev/null || true
          kill -CONT -- "-$pid" 2>/dev/null || true
          sleep 2
          kill -KILL -- "-$pid" 2>/dev/null || true
          break
        fi
      fi
      sleep "$sample_seconds"
    done
  }}
  monitor_gpu &
  monitor_pid=$!
fi
wait "$pid"
rc=$?
if test -n "$monitor_pid"; then
  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
fi
rm -f "$pidfile"
if test -f "$safety"; then rc=86; fi
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
        local_timeout = None if effective_timeout is None else effective_timeout + 120
        try:
            stdout, stderr = proc.communicate(script, timeout=local_timeout)
            self._last_command_duration_seconds = time.monotonic() - command_started
            self._current_proc = None
            return SubprocessResult(proc.returncode, stdout=stdout, stderr=stderr)
        except subprocess.TimeoutExpired:
            self._last_command_duration_seconds = time.monotonic() - command_started
            self._kill_remote_process()
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            proc.wait()
            self._current_proc = None
            return SubprocessResult(1, stderr=f"Remote timeout after {effective_timeout} seconds")

    @staticmethod
    def _number(value: str | None) -> float | None:
        try:
            return float(str(value).strip())
        except (TypeError, ValueError):
            return None

    def _collect_remote_telemetry(self, execution_dir: Path, phase: str) -> dict:
        """Pull harness telemetry even when the experiment was safety-terminated."""
        if self.benchmark_name in CPU_ONLY_TASKS:
            summary = {
                "schema_version": "fml-gpu-telemetry-summary-v1",
                "phase": phase,
                "applicability": "NOT_APPLICABLE_CPU_TASK",
                "sample_count": 0,
                "peak_temperature_c": None,
                "peak_memory_mb": None,
                "mean_power_w": None,
                "peak_power_w": None,
                "gpu_active_seconds": None,
                "warning_count": 0,
                "safety_termination_reason": None,
            }
            self.gpu_telemetry_summaries.append(summary)
            return summary
        local_csv = execution_dir / "gpu_telemetry.csv"
        source = f"{self.ssh_host}:{self._remote_telemetry_path}"
        ssh_transport = (
            "ssh -o BatchMode=yes -o ConnectTimeout=8 "
            "-o ServerAliveInterval=30 -o ServerAliveCountMax=3"
        )
        pulled = subprocess.run(
            ["rsync", "-a", "-e", ssh_transport, source, str(local_csv)],
            text=True,
            capture_output=True,
        )
        rows: list[dict[str, str]] = []
        if pulled.returncode == 0 and local_csv.is_file():
            with local_csv.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        temperatures = [value for row in rows if (value := self._number(row.get("temperature_c"))) is not None]
        powers = [value for row in rows if (value := self._number(row.get("power_w"))) is not None]
        memories = [value for row in rows if (value := self._number(row.get("memory_used_mb"))) is not None]
        utilizations = [value for row in rows if (value := self._number(row.get("utilization_pct"))) is not None]
        pacing_states = [str(row.get("thermal_pacing_state") or "").strip().upper() for row in rows]
        pacing_pause_seconds = [
            value
            for row in rows
            if (value := self._number(row.get("thermal_pacing_pause_seconds"))) is not None
        ]
        explicit_compute_activity = [
            value
            for row in rows
            if (value := self._number(row.get("experiment_compute_active"))) is not None
        ]
        safety_result = self._ssh_script(
            f"test ! -f {shlex.quote(str(self._remote_safety_path))} || cat {shlex.quote(str(self._remote_safety_path))}\n",
            timeout=30,
        )
        safety_payload: dict = {}
        if safety_result.stdout.strip():
            try:
                safety_payload = json.loads(safety_result.stdout)
            except json.JSONDecodeError:
                safety_payload = {"reason": "UNPARSEABLE_SAFETY_TERMINATION"}
            (execution_dir / "gpu_safety_termination.json").write_text(
                json.dumps(safety_payload, indent=2) + "\n", encoding="utf-8"
            )
        if explicit_compute_activity:
            sampled_compute_seconds = sum(
                self.telemetry_sample_seconds for value in explicit_compute_activity if value > 0
            )
            # Cooling happens before the ordinary sample interval. A positive
            # row therefore represents the following running interval; pause
            # duration is already outside this estimate and must not be
            # subtracted a second time.
            active_seconds = sampled_compute_seconds
            active_measurement = "EXPERIMENT_PROCESS_GROUP_COMPUTE_PID_SAMPLED_INTERVALS"
        else:
            # Backward-compatible reduction for telemetry produced before the
            # process-group activity column existed.
            active_seconds = sum(
                self.telemetry_sample_seconds for value in utilizations if value > 0
            )
            active_measurement = "LEGACY_WHOLE_GPU_UTILIZATION_NONZERO"
        summary = {
            "schema_version": "fml-gpu-telemetry-summary-v1",
            "phase": phase,
            "applicability": "APPLICABLE",
            "sample_interval_seconds": self.telemetry_sample_seconds,
            "sample_count": len(rows),
            "peak_temperature_c": max(temperatures) if temperatures else None,
            "peak_memory_mb": max(memories) if memories else None,
            "mean_power_w": sum(powers) / len(powers) if powers else None,
            "peak_power_w": max(powers) if powers else None,
            "gpu_active_seconds": active_seconds,
            "gpu_active_seconds_measurement": active_measurement,
            "warning_count": sum(value >= self.gpu_warning_temperature_c for value in temperatures),
            "thermal_pacing_enabled": self.thermal_pacing_enabled,
            "thermal_pacing_pause_samples": sum(value == "PAUSED_RESUMED" for value in pacing_states),
            "thermal_pacing_pause_seconds": sum(pacing_pause_seconds),
            "safety_termination_reason": safety_payload.get("reason"),
            "telemetry_pull_error": pulled.stderr.strip() if pulled.returncode != 0 else None,
        }
        (execution_dir / "gpu_telemetry_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        self.gpu_telemetry_summaries.append(summary)
        ledger = current_budget_ledger()
        if ledger is not None:
            ledger.add_gpu_active_seconds(active_seconds)
            self._capture_budget_ledger_snapshot()
        return summary

    def _remote_integrity_error(self) -> Optional[str]:
        checks = [f"test -d {shlex.quote(self.remote_repo_dir + '/.git')}"]
        for target in self.config.get("target_files", []):
            remote_target = str(PurePosixPath(self.remote_repo_dir) / target)
            checks.append(f"test -e {shlex.quote(remote_target)}")
        result = self._ssh_script("set -e\n" + "\n".join(checks) + "\n", timeout=30)
        if result.returncode != 0:
            return f"remote workspace integrity check failed: {result.stderr.strip()}"
        return None

    def _sanitize_agent_visible_evaluator_output(
        self,
        *,
        result: SubprocessResult,
        repo_path: Path,
        execution_dir: Path,
        phase_kind: str,
    ) -> tuple[SubprocessResult, dict | None]:
        """Quarantine PrivacyMeter's all-target validation output.

        PrivacyMeter audits all four targets before the validation splitter
        selects its one visible target.  Its raw stdout/stderr can therefore
        contain protected-target AUC/TPR values.  Keep that stream in an
        explicitly operator-only file and return only the postprocessed
        validation split to the research agent and its retry prompts.
        """
        if self.benchmark_name != "Privacy_privacymeter" or phase_kind == "test":
            return result, None

        operator_path = execution_dir / "operator_only_evaluator_output.json"
        operator_payload = {
            "schema_version": "fml-operator-only-evaluator-output-v1",
            "contains_protected_final_metric": True,
            "phase": phase_kind,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        operator_path.write_text(
            json.dumps(operator_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.chmod(operator_path, 0o600)
        operator_sha256 = hashlib.sha256(operator_path.read_bytes()).hexdigest()

        split_path = repo_path / "results_tmp" / "val_info.json"
        visible: dict = {
            "schema_version": "fml-agent-visible-evaluator-output-v1",
            "phase": phase_kind,
            "raw_evaluator_output_redacted": True,
            "operator_output_sha256": operator_sha256,
            "operator_output_size_bytes": operator_path.stat().st_size,
            "returncode": result.returncode,
        }
        if split_path.is_file():
            try:
                payload = json.loads(split_path.read_text(encoding="utf-8"))
                task = payload.get("cifar10", {})
                visible.update({
                    "constraint_contract": payload.get("constraint_contract"),
                    "validation_means": task.get("means", {}),
                    "validation_raw_metrics": task.get("original_metrics"),
                    "validation_components": task.get("final_info_dict", {}),
                })
            except (OSError, json.JSONDecodeError) as exc:
                visible["sanitization_error"] = f"invalid val_info.json: {exc}"
        else:
            visible["execution_error"] = (
                "PrivacyMeter validation failed before a sanitized val_info.json was produced; "
                "raw evaluator output is operator-only."
            )
        visible_path = execution_dir / "agent_visible_evaluator_output.json"
        visible_path.write_text(
            json.dumps(visible, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        safe_text = json.dumps(visible, sort_keys=True, ensure_ascii=False)
        sanitized = SubprocessResult(
            result.returncode,
            stdout=safe_text if result.returncode == 0 else "",
            stderr=safe_text if result.returncode != 0 else "",
        )
        return sanitized, {
            "schema_version": "fml-operator-only-artifact-reference-v1",
            "contains_protected_final_metric": True,
            "sha256": operator_sha256,
            "size_bytes": operator_path.stat().st_size,
            "relative_path": operator_path.name,
            "agent_visible": False,
        }

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

        phase_kind = (
            "test" if command_key == "test_command"
            else "pre_test_val" if str(run_id) == "pre_test_val"
            else "val"
        )
        ledger = current_budget_ledger()
        try:
            if ledger is not None:
                self._admit_phase_budget(phase_kind, ledger)
                ledger.before_execution(phase_kind)
                self._capture_budget_ledger_snapshot()
        except BudgetExceeded as exc:
            return {"success": False, "results": None, "primary_metric": None, "error": str(exc)}
        count_key = {
            "val": "candidate_validation_count",
            "pre_test_val": "pre_test_validation_count",
            "test": "protected_test_count",
        }[phase_kind]
        self.execution_counts[count_key] += 1

        local_results = Path(self.repo_dir).resolve() / "results_tmp"
        if local_results.exists():
            shutil.rmtree(local_results)
        execution_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        execution_dir = Path(self.workspace_dir) / f"run_{run_id}" / f"execution_{execution_timestamp}"
        execution_dir.mkdir(parents=True, exist_ok=True)
        repo_path = Path(self.repo_dir).resolve()
        activation_contract = prepare_runtime_probe(
            repo=repo_path,
            target_files=[] if phase_kind == "test" else list(self.config.get("target_files", [])),
            baseline_sources=self.baseline_target_sources,
            run_id=str(run_id),
            configuration={
                "benchmark": self.benchmark_name,
                "agent": self.agent_name,
                "metric": self.config.get("metric"),
                "experimental_seed": self.config.get("_experimental_seed"),
                "phase": phase_kind,
            },
        )
        if activation_contract is not None and not activation_contract["passed_static_contract"]:
            activation = finalize_runtime_probe(
                repo=repo_path,
                execution_dir=execution_dir,
                contract=activation_contract,
            )
            rejection_reason = activation_contract.get(
                "static_rejection_reason", "INVALID_EXECUTION_STATIC_CONTRACT"
            )
            activation["status"] = rejection_reason
            self._record_activation(activation, phase_kind)
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": rejection_reason,
                "activation": activation,
            }
        if backup and self.config.get("save_code_backup", False):
            self._backup_files(run_id, execution_timestamp)

        remote_evidence_stem = f"{phase_kind}_{run_id}_{execution_timestamp}"
        self._remote_telemetry_path = str(
            PurePosixPath(self.remote_run_base) / f".fml_gpu_telemetry_{remote_evidence_stem}.csv"
        )
        self._remote_safety_path = str(
            PurePosixPath(self.remote_run_base) / f".fml_gpu_safety_{remote_evidence_stem}.json"
        )
        self._assert_gpu_idle()
        self._sync_controller_workspace()
        target_attestation = self._attest_remote_targets()
        (execution_dir / "remote_target_attestation.json").write_text(
            json.dumps(target_attestation, indent=2) + "\n", encoding="utf-8"
        )
        if not target_attestation["passed"]:
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": "REMOTE_SOURCE_MISMATCH: candidate target hashes differ after rsync",
                "target_attestation": target_attestation,
            }
        before = self._platform_snapshot()
        prep = self._ssh_script(
            f"rm -rf -- {shlex.quote(self.remote_repo_dir + '/results_tmp')}\n"
            f"chmod 555 {shlex.quote(self.remote_repo_dir + '/.git')} 2>/dev/null || true\n",
            timeout=60,
        )
        if prep.returncode != 0:
            return {"success": False, "results": None, "primary_metric": None, "error": prep.stderr}

        original_command = command
        command = self._resolve_harness_paths(command)
        resolved_command = command
        repro_env = reproducibility_environment(self.config.get("_experimental_seed"))
        command = self._wrap_experiment_environment(
            command,
            repro_env,
            runtime_probe=True,
        )
        result = SubprocessResult(0)
        try:
            if self.config.get("do_preprocess") and self.config.get("preprocess_commands"):
                for pre_cmd in self.config["preprocess_commands"]:
                    result = self._remote_command(pre_cmd)
                    if result.returncode != 0:
                        break
            if result.returncode == 0:
                result = self._remote_command(command)
                if self._last_command_duration_seconds is not None:
                    self.phase_command_durations[phase_kind].append(
                        self._last_command_duration_seconds
                    )
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
        self.reproducibility_records.append({
            "schema_version": "fml-remote-reproducibility-evidence-v1",
            "phase": phase_kind,
            "experimental_seed": self.config.get("_experimental_seed"),
            "deterministic_environment": repro_env,
            "platform_before": before,
            "platform_after": after,
        })
        telemetry = self._collect_remote_telemetry(execution_dir, phase_kind)
        pull_error: str | None = None
        try:
            self._pull_results()
        except RuntimeError as exc:
            pull_error = str(exc)
        result, operator_only_output = self._sanitize_agent_visible_evaluator_output(
            result=result,
            repo_path=repo_path,
            execution_dir=execution_dir,
            phase_kind=phase_kind,
        )
        evidence = {
            "schema_version": "fml-ssh-execution-v2",
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "ssh_host": self.ssh_host,
            "remote_project_root": self.remote_project_root,
            "remote_workspace": self.remote_run_base,
            "remote_repo_dir": self.remote_repo_dir,
            "benchmark": self.benchmark_name,
            "agent": self.agent_name,
            "conda_env": self.conda_env,
            "phase": command_key,
            "command": original_command,
            "resolved_command": resolved_command,
            "effective_command_sha256": hashlib.sha256(command.encode()).hexdigest(),
            "command_sha256": hashlib.sha256(original_command.encode()).hexdigest(),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "platform_before": before,
            "platform_after": after,
            "integrity_error": integrity_error,
            "activation_contract_sha256": (
                activation_contract.get("candidate_sha256") if activation_contract else None
            ),
            "reproducibility_environment": repro_env,
            "gpu_telemetry": telemetry,
            "operator_only_evaluator_output": operator_only_output,
            "remote_harness_attestation": self._remote_harness_attestation,
            "remote_target_attestation": target_attestation,
        }
        (execution_dir / "ssh_execution.json").write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        activation = finalize_runtime_probe(
            repo=repo_path,
            execution_dir=execution_dir,
            contract=activation_contract,
        )
        self._record_activation(activation, phase_kind)
        if integrity_error:
            return {"success": False, "results": None, "primary_metric": None, "error": integrity_error}
        if result.returncode != 0:
            phase = "val" if command_key == "val_command" else "test"
            self._save_bug_execution_record(run_id, execution_timestamp, phase, result)
            safety_reason = telemetry.get("safety_termination_reason")
            error = f"{safety_reason}: {result.stderr}" if safety_reason else result.stderr
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": error,
                "activation": activation,
                "gpu_telemetry": telemetry,
            }
        if pull_error:
            return {"success": False, "results": None, "primary_metric": None, "error": pull_error}
        if not activation["passed"]:
            return {
                "success": False,
                "results": None,
                "primary_metric": None,
                "error": "INVALID_EXECUTION: required candidate code did not activate at runtime",
                "activation": activation,
                "gpu_telemetry": telemetry,
            }
        actual_output_path = output_path
        if not (Path(self.repo_dir).resolve() / output_path).exists():
            fallback = Path(self.config.get("exp_running_dir", "results_tmp/")) / "final_info.json"
            if (Path(self.repo_dir).resolve() / fallback).exists():
                actual_output_path = str(fallback)
        collected = self._collect_results(run_id, execution_timestamp, actual_output_path)
        collected["activation"] = activation
        collected["gpu_telemetry"] = telemetry
        return collected

    def run_val(self, run_id: int) -> dict:
        return self._run_phase(run_id, "val_command", VAL_OUTPUT)

    def run_test(self, run_id: int) -> dict:
        return self._run_phase(run_id, "test_command", TEST_OUTPUT, backup=False)

    def execution_contract_summary(self) -> dict:
        summary = super().execution_contract_summary()
        summary["gpu_safety_contract"] = {
            "schema_version": "fml-gpu-safety-contract-v1",
            "sample_seconds": self.telemetry_sample_seconds,
            "warning_temperature_c": self.gpu_warning_temperature_c,
            "abort_temperature_c": self.gpu_abort_temperature_c,
            "thermal_throttle_abort_samples": self.thermal_throttle_abort_samples,
            "gpu_startup_timeout_seconds": self.gpu_startup_timeout_seconds,
            "thermal_pacing_enabled": self.thermal_pacing_enabled,
            "thermal_pacing_start_c": self.thermal_pacing_start_c,
            "thermal_pacing_resume_c": self.thermal_pacing_resume_c,
        }
        return summary

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
