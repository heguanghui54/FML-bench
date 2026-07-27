import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


class ExecutorFactoryTests(unittest.TestCase):
    def _load_modules(self):
        package = types.ModuleType("benchmark")
        package.__path__ = [str(ROOT / "benchmark")]
        executor_spec = importlib.util.spec_from_file_location(
            "benchmark.executor", ROOT / "benchmark" / "executor.py"
        )
        executor = importlib.util.module_from_spec(executor_spec)
        assert executor_spec and executor_spec.loader
        with patch.dict(sys.modules, {"benchmark": package, "benchmark.executor": executor}):
            executor_spec.loader.exec_module(executor)
            factory_spec = importlib.util.spec_from_file_location(
                "benchmark.executor_factory", ROOT / "benchmark" / "executor_factory.py"
            )
            factory = importlib.util.module_from_spec(factory_spec)
            assert factory_spec and factory_spec.loader
            factory_spec.loader.exec_module(factory)
        return executor, factory

    def test_local_backend_does_not_import_ssh_executor(self):
        executor_module, factory = self._load_modules()
        config = {"repo_dir": "workspace/T__r/repo", "conda_env": "x"}
        with patch.dict("sys.modules", {"benchmark.ssh_executor": None}):
            executor = factory.make_executor(config, eval_backend="local")
        self.assertIsInstance(executor, executor_module.BenchmarkExecutor)

    def test_unknown_backend_fails_loudly(self):
        _, factory = self._load_modules()
        config = {"repo_dir": "workspace/T__r/repo", "conda_env": "x"}
        with self.assertRaisesRegex(ValueError, "Unknown eval_backend"):
            factory.make_executor(config, eval_backend="typo")


if __name__ == "__main__":
    unittest.main()
