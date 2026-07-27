import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
