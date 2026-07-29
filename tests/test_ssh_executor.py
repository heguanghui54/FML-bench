import importlib.util
import inspect
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_executor_modules():
    package = types.ModuleType("benchmark")
    package.__path__ = [str(ROOT / "benchmark")]
    executor_spec = importlib.util.spec_from_file_location(
        "benchmark.executor", ROOT / "benchmark" / "executor.py"
    )
    executor = importlib.util.module_from_spec(executor_spec)
    assert executor_spec and executor_spec.loader
    with patch.dict(sys.modules, {"benchmark": package, "benchmark.executor": executor}):
        executor_spec.loader.exec_module(executor)
        ssh_spec = importlib.util.spec_from_file_location(
            "benchmark.ssh_executor", ROOT / "benchmark" / "ssh_executor.py"
        )
        ssh_executor = importlib.util.module_from_spec(ssh_spec)
        assert ssh_spec and ssh_spec.loader
        ssh_spec.loader.exec_module(ssh_executor)
    return ssh_executor


class SSHExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_executor_modules()

    def _config(self, task="Causality_gcastle"):
        return {
            "repo_dir": f"workspace/{task}__trial-01/repository",
            "conda_env": "test-env",
            "target_files": ["algorithm.py"],
            "metric": "score",
            "_execution_backend": {
                "name": "ssh",
                "ssh_host": "ubuntu-heshi",
                "remote_project_root": "/media/heshi/game/fml-scientist/repo",
                "require_gpu_idle": True,
            },
        }

    def test_paths_are_derived_under_external_project_root(self):
        executor = self.module.SSHExecutor(
            self._config(), benchmark_name="Causality_gcastle"
        )
        self.assertEqual(
            executor.remote_template_base,
            "/media/heshi/game/fml-scientist/repo/workspace/Causality_gcastle",
        )
        self.assertEqual(
            executor.remote_repo_dir,
            "/media/heshi/game/fml-scientist/repo/workspace/"
            "Causality_gcastle__trial-01/repository",
        )

    def test_system_disk_remote_root_is_rejected(self):
        config = self._config()
        config["_execution_backend"]["remote_project_root"] = "/home/heshi/fml"
        with self.assertRaisesRegex(ValueError, "external disk"):
            self.module.SSHExecutor(config, benchmark_name="Causality_gcastle")

    def test_gpu_gate_blocks_other_compute_but_cpu_task_does_not(self):
        gpu = self.module.SSHExecutor(
            self._config("Privacy_opacus"), benchmark_name="Privacy_opacus"
        )
        with patch.object(gpu, "_gpu_processes", return_value="123, i4h, 512 MiB"):
            with self.assertRaisesRegex(RuntimeError, "stop i4h"):
                gpu._assert_gpu_idle()
        causalml = self.module.SSHExecutor(
            self._config("Causality_causalml"), benchmark_name="Causality_causalml"
        )
        with patch.object(causalml, "_gpu_processes", return_value="123, i4h, 512 MiB"):
            with self.assertRaisesRegex(RuntimeError, "stop i4h"):
                causalml._assert_gpu_idle()
        cpu = self.module.SSHExecutor(
            self._config("Causality_gcastle"), benchmark_name="Causality_gcastle"
        )
        with patch.object(cpu, "_gpu_processes", side_effect=AssertionError("must not query")):
            cpu._assert_gpu_idle()

    def test_telemetry_summary_and_safety_reason_are_preserved(self):
        executor = self.module.SSHExecutor(
            self._config("Privacy_opacus"), benchmark_name="Privacy_opacus"
        )
        executor._remote_telemetry_path = "/media/heshi/game/fml-scientist/repo/workspace/run/.telemetry.csv"
        executor._remote_safety_path = "/media/heshi/game/fml-scientist/repo/workspace/run/.safety.json"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out / "gpu_telemetry.csv").write_text(
                "timestamp,temperature_c,power_w,utilization_pct,memory_used_mb,memory_total_mb,thermal_slowdown,thermal_pacing_state,thermal_pacing_pause_seconds,experiment_compute_active\n"
                "t1,81,90,50,1000,12000,Not Active,PAUSED_RESUMED,2,1\n"
                "t2,89,110,75,2000,12000,Active,PAUSED_RESUMED,3,1\n",
                encoding="utf-8",
            )
            completed = types.SimpleNamespace(returncode=0, stdout="", stderr="")
            safety = types.SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"reason": "GPU_TEMPERATURE_ABORT", "temperature_c": 89}),
                stderr="",
            )
            with patch.object(self.module.subprocess, "run", return_value=completed), patch.object(
                executor, "_ssh_script", return_value=safety
            ):
                summary = executor._collect_remote_telemetry(out, "val")
            self.assertEqual(summary["peak_temperature_c"], 89)
            self.assertEqual(summary["peak_memory_mb"], 2000)
            self.assertEqual(summary["gpu_active_seconds"], 10)
            self.assertEqual(
                summary["gpu_active_seconds_measurement"],
                "EXPERIMENT_PROCESS_GROUP_COMPUTE_PID_SAMPLED_INTERVALS",
            )
            self.assertEqual(summary["warning_count"], 1)
            self.assertEqual(summary["thermal_pacing_pause_samples"], 2)
            self.assertEqual(summary["thermal_pacing_pause_seconds"], 5)
            self.assertEqual(summary["safety_termination_reason"], "GPU_TEMPERATURE_ABORT")
            self.assertTrue((out / "gpu_safety_termination.json").is_file())

    def test_remote_sampler_does_not_treat_not_active_as_active(self):
        source = inspect.getsource(self.module.SSHExecutor._remote_command)
        self.assertIn("active|yes|true|1)", source)
        self.assertNotIn("*active*", source)

    def test_remote_sampler_proactively_paces_without_weakening_abort(self):
        source = inspect.getsource(self.module.SSHExecutor._remote_command)
        self.assertIn('kill -STOP -- "-$pid"', source)
        self.assertIn('kill -CONT -- "-$pid"', source)
        self.assertIn('sleep 1', source)
        self.assertIn("SUSTAINED_THERMAL_THROTTLING", source)
        self.assertIn("GPU_STARTUP_TIMEOUT", source)
        executor = self.module.SSHExecutor(
            self._config("Privacy_opacus"), benchmark_name="Privacy_opacus"
        )
        contract = executor.execution_contract_summary()["gpu_safety_contract"]
        self.assertEqual(contract["thermal_pacing_start_c"], 78.0)
        self.assertEqual(contract["thermal_pacing_resume_c"], 72.0)
        self.assertEqual(contract["gpu_startup_timeout_seconds"], 900)

    def test_invalid_gpu_startup_timeout_is_rejected(self):
        config = self._config("Privacy_opacus")
        config["_execution_backend"]["gpu_startup_timeout_seconds"] = 0
        with self.assertRaisesRegex(ValueError, "must be positive"):
            self.module.SSHExecutor(config, benchmark_name="Privacy_opacus")

    def test_invalid_thermal_pacing_band_is_rejected(self):
        config = self._config("Privacy_opacus")
        config["_execution_backend"]["thermal_pacing_start_c"] = 75
        config["_execution_backend"]["thermal_pacing_resume_c"] = 75
        with self.assertRaisesRegex(ValueError, "must be below"):
            self.module.SSHExecutor(config, benchmark_name="Privacy_opacus")

    def test_privacymeter_validation_quarantines_raw_protected_output(self):
        executor = self.module.SSHExecutor(
            self._config("Privacy_privacymeter"), benchmark_name="Privacy_privacymeter"
        )
        raw_sentinel = "PROTECTED_AUC_SENTINEL_0.9876"
        raw = self.module.SubprocessResult(
            1,
            stdout=f"all-target average {raw_sentinel}",
            stderr=f"target 3 {raw_sentinel}",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            execution = root / "execution"
            (repo / "results_tmp").mkdir(parents=True)
            execution.mkdir()
            (repo / "results_tmp" / "val_info.json").write_text(
                json.dumps({
                    "constraint_contract": {"version": "privacy-v2"},
                    "cifar10": {
                        "means": {
                            "AUC_gap_mean": None,
                            "TPR@0.1%FPR_mean": 0.012,
                            "test_acc_mean": 0.59,
                            "constraint_violated": True,
                        },
                        "original_metrics": {"AUC_gap": 0.3465},
                        "final_info_dict": {"AUC_gap": [0.3465]},
                    },
                }),
                encoding="utf-8",
            )
            sanitized, reference = executor._sanitize_agent_visible_evaluator_output(
                result=raw,
                repo_path=repo,
                execution_dir=execution,
                phase_kind="val",
            )
            self.assertNotIn(raw_sentinel, sanitized.stdout)
            self.assertNotIn(raw_sentinel, sanitized.stderr)
            self.assertIn("0.3465", sanitized.stderr)
            self.assertFalse(reference["agent_visible"])
            operator = (execution / "operator_only_evaluator_output.json").read_text()
            self.assertIn(raw_sentinel, operator)
            visible = (execution / "agent_visible_evaluator_output.json").read_text()
            self.assertNotIn(raw_sentinel, visible)

    def test_privacymeter_runtime_failure_is_redacted_without_val_info(self):
        executor = self.module.SSHExecutor(
            self._config("Privacy_privacymeter"), benchmark_name="Privacy_privacymeter"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            execution = root / "execution"
            repo.mkdir()
            execution.mkdir()
            sanitized, _ = executor._sanitize_agent_visible_evaluator_output(
                result=self.module.SubprocessResult(1, stdout="SECRET", stderr="SECRET"),
                repo_path=repo,
                execution_dir=execution,
                phase_kind="pre_test_val",
            )
            self.assertNotIn("SECRET", sanitized.stderr)
            self.assertIn("operator-only", sanitized.stderr)

    def test_platform_snapshot_uses_task_conda_environment(self):
        executor = self.module.SSHExecutor(
            self._config("Privacy_opacus"), benchmark_name="Privacy_opacus"
        )
        captured = {}

        def fake_script(script, timeout=120):
            captured["script"] = script
            return types.SimpleNamespace(returncode=0, stdout="", stderr="")

        with patch.object(executor, "_ssh_script", side_effect=fake_script):
            executor._platform_snapshot()
        self.assertIn('"$conda" run --no-capture-output', captured["script"])
        self.assertIn("env_name=test-env", captured["script"])
        self.assertNotIn("; python --version", captured["script"])

    def test_relative_task_harness_path_is_rewritten_to_frozen_overlay(self):
        executor = self.module.SSHExecutor(
            self._config("Generalization_domainbed"), benchmark_name="Generalization_domainbed"
        )
        command = "cp ../../../ml_tasks/Generalization_domainbed/train_eval.py ./"

        resolved = executor._resolve_harness_paths(command)

        self.assertNotIn("../../../ml_tasks", resolved)
        self.assertNotIn("../..//", resolved)
        self.assertTrue(resolved.startswith("cp /media/heshi/game/"))
        self.assertIn("/.fml_harness/ml_tasks/Generalization_domainbed/train_eval.py", resolved)

    def test_experiment_environment_applies_to_entire_command_chain(self):
        executor = self.module.SSHExecutor(
            self._config("Generalization_domainbed"), benchmark_name="Generalization_domainbed"
        )

        command = executor._wrap_experiment_environment(
            "cp source target && python train_eval.py",
            {"PYTHONHASHSEED": "4409", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
            runtime_probe=True,
        )

        self.assertTrue(command.startswith("export PYTHONHASHSEED=4409; export CUBLAS_WORKSPACE_CONFIG="))
        self.assertIn('; export PYTHONPATH="$PWD"/.fml_runtime_probe:', command)
        self.assertIn("; cp source target && python train_eval.py", command)
        self.assertNotIn("PYTHONHASHSEED=4409 cp", command)

    def test_remote_target_attestation_rejects_hash_mismatch(self):
        executor = self.module.SSHExecutor(
            self._config("Causality_gcastle"), benchmark_name="Causality_gcastle"
        )
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repository"
            repo.mkdir()
            (repo / "algorithm.py").write_text("local", encoding="utf-8")
            executor.repo_dir = str(repo)
            remote = types.SimpleNamespace(
                returncode=0,
                stdout="algorithm.py\t" + "0" * 64 + "\n",
                stderr="",
            )
            with patch.object(executor, "_ssh_script", return_value=remote):
                attestation = executor._attest_remote_targets()
        self.assertFalse(attestation["passed"])


if __name__ == "__main__":
    unittest.main()
