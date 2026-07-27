import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fml_setup", ROOT / "setup.py")
SETUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SETUP)


class SetupBootstrapTests(unittest.TestCase):
    def test_incomplete_environment_resumes_and_complete_environment_skips(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / "env"
            prefix.mkdir()
            calls = []

            with patch.object(SETUP, "conda_env_prefix", return_value=prefix), patch.object(
                SETUP, "run", side_effect=lambda command, **kwargs: calls.append(command)
            ):
                SETUP.conda_create("demo", "3.10", ["pip install example==1"])
                self.assertEqual(len(calls), 1)
                self.assertTrue((prefix / ".fml_setup_complete").is_file())

                SETUP.conda_create("demo", "3.10", ["pip install example==1"])
                self.assertEqual(len(calls), 1)

                SETUP.conda_create("demo", "3.10", ["pip install example==2"])
                self.assertEqual(len(calls), 2)

    def test_domainbed_no_longer_contains_unsatisfiable_mkl_pair(self):
        source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertNotIn("mkl==2023.1.0 mkl-service==2.4.0", source)
        self.assertNotIn("mkl-fft==1.3.11 mkl-random==1.2.8", source)
        self.assertIn("conda install mkl==2024.0", source)

    def test_officehome_uses_current_gdown_id_api(self):
        source = (ROOT / "setup.py").read_text(encoding="utf-8")
        self.assertIn("gdown==5.2.0", source)
        self.assertIn("id='1gkbf_Kaxo", source)
        self.assertIn("resume=True", source)

    def test_shared_dataset_link_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shared" / "cifar"
            source.mkdir(parents=True)
            destination = root / "task" / "data" / "cifar"

            SETUP.link_shared_dataset(source, destination)
            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.resolve(), source.resolve())

            SETUP.link_shared_dataset(source, destination)
            self.assertEqual(destination.resolve(), source.resolve())

    def test_shared_dataset_link_preserves_existing_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shared" / "cifar"
            destination = root / "task" / "data" / "cifar"
            source.mkdir(parents=True)
            destination.mkdir(parents=True)
            marker = destination / "existing"
            marker.write_text("keep", encoding="utf-8")

            SETUP.link_shared_dataset(source, destination)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_shared_cifar_link_removes_redundant_local_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shared = root / "shared" / "cifar-100-python"
            shared.mkdir(parents=True)
            task_data = root / "task" / "data"
            task_data.mkdir(parents=True)
            archive = task_data / "cifar-100-python.tar.gz"
            archive.write_bytes(b"partial")

            with patch.object(SETUP, "ensure_shared_cifar", return_value=shared):
                SETUP.link_shared_cifar("cifar100", task_data)

            self.assertTrue((task_data / "cifar-100-python").is_symlink())
            self.assertFalse(archive.exists())

    def test_generated_file_is_replaced_with_shared_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "shared" / "data.pkl"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"verified")
            destination = root / "task" / "data.pkl"
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"duplicate")

            SETUP.replace_with_shared_file(source, destination)

            self.assertTrue(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"verified")


if __name__ == "__main__":
    unittest.main()
