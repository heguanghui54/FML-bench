import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = (
    ROOT
    / "ml_tasks"
    / "Generalization_domainbed"
    / "original_file_backup"
    / "fast_data_loader.py"
)


class DomainBedLoaderShutdownTests(unittest.TestCase):
    def _load_module(self):
        torch = types.ModuleType("torch")
        torch.utils = types.SimpleNamespace(
            data=types.SimpleNamespace(Sampler=object)
        )
        with mock.patch.dict(sys.modules, {"torch": torch}):
            spec = importlib.util.spec_from_file_location(
                "domainbed_fast_data_loader_test", LOADER_PATH
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        return module

    def test_shutdown_calls_torch_worker_cleanup_when_available(self):
        module = self._load_module()
        iterator = mock.Mock()

        module._shutdown_iterator(iterator)

        iterator._shutdown_workers.assert_called_once_with()

    def test_shutdown_is_compatible_with_single_process_iterator(self):
        module = self._load_module()

        module._shutdown_iterator(object())


if __name__ == "__main__":
    unittest.main()
