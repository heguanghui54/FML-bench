from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from benchmark.executor import BenchmarkExecutor
from benchmark.runtime_probe import finalize_runtime_probe, prepare_runtime_probe, probe_environment_prefix
from ml_scientist.execution_contracts import (
    BudgetExceeded,
    BudgetLedger,
    ReproducibilityContract,
    build_candidate_activation_contract,
    compare_repeat_validations,
    normalize_summary_contract,
    validate_runtime_markers,
)


class BudgetAndReproducibilityTests(unittest.TestCase):
    def test_budget_stops_before_exceeding_a_frozen_call_count(self):
        ledger = BudgetLedger(limits={
            "token_budget": 100,
            "wall_clock_seconds": 60,
            "proposal_count": 1,
            "review_count": 1,
            "candidate_validation_count": 1,
            "pre_test_validation_count": 1,
            "protected_test_count": 1,
        })
        ledger.before_llm("proposal")
        with self.assertRaises(BudgetExceeded):
            ledger.before_llm("proposal")
        self.assertEqual(ledger.snapshot()["exhausted_dimension"], "proposal_count")

    def test_repeat_disagreement_blocks_protected_test_without_tiebreaker(self):
        contract = ReproducibilityContract(metric_absolute_tolerance=0.01)
        comparison = compare_repeat_validations(
            {"success": True, "primary_metric": 0.80},
            {"success": True, "primary_metric": 0.75},
            contract,
        )
        self.assertEqual(comparison["status"], "STOCHASTIC_UNCERTAIN")
        self.assertFalse(comparison["protected_test_allowed"])
        self.assertTrue(comparison["no_favourable_tie_breaker"])

    def test_wall_budget_exposes_live_remaining_time(self):
        ledger = BudgetLedger(limits={"wall_clock_seconds": 0.05})
        first = ledger.remaining("wall_clock_seconds")
        self.assertIsNotNone(first)
        time.sleep(0.01)
        second = ledger.remaining("wall_clock_seconds")
        self.assertLess(second, first)

    def test_repeat_is_not_started_when_observed_val_duration_cannot_fit(self):
        ledger = BudgetLedger(limits={"wall_clock_seconds": 100.0})
        ledger.started_monotonic -= 30.0
        executor = BenchmarkExecutor({"repo_dir": ".", "conda_env": "unused"})
        executor.phase_command_durations["val"] = [80.0]
        with self.assertRaises(BudgetExceeded):
            executor._admit_phase_budget("pre_test_val", ledger)
        self.assertEqual(ledger.exhausted_dimension, "wall_clock_seconds")

    def test_v1_summary_remains_readable_but_incomplete(self):
        summary = normalize_summary_contract({"benchmark": "task", "test_result": {}})
        self.assertEqual(summary["resource_accounting_status"], "RESOURCE_ACCOUNTING_INCOMPLETE")
        self.assertFalse(summary["resource_matched_comparable"])


class ActivationContractTests(unittest.TestCase):
    def test_generated_probe_defers_cuda_until_candidate_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            source = repo / "model.py"
            source.write_text("def train():\n    return 2\n", encoding="utf-8")
            prepare_runtime_probe(
                repo=repo,
                target_files=["model.py"],
                baseline_sources={"model.py": "def train():\n    return 1\n"},
                run_id="deferred-cuda-proof",
                configuration={"seed": 7},
            )
            generated = (repo / ".fml_runtime_probe" / "sitecustomize.py").read_text(
                encoding="utf-8"
            )
            import_preamble = generated.split("def _enforce_deterministic_flags", 1)[0]
            self.assertNotIn("cuda.is_available", import_preamble)
            self.assertNotIn("cuda.manual_seed_all", import_preamble)
            self.assertIn("_enforce_deterministic_flags(include_cuda=False)", generated)
            self.assertIn("_enforce_deterministic_flags(include_cuda=True)", generated)

    def test_python_whitespace_comments_and_docstrings_are_semantic_noop(self):
        contract = build_candidate_activation_contract(
            before={"model.py": "def train():\n    return 1\n"},
            after={
                "model.py": (
                    "# formatting-only candidate\n\n"
                    "def train():\n"
                    "    \"\"\"New documentation is not an intervention.\"\"\"\n"
                    "    return 1\n"
                )
            },
            run_id="semantic-noop-r1",
        )
        self.assertFalse(contract["passed_static_contract"])
        self.assertEqual(contract["semantic_noop_files"], ["model.py"])
        self.assertEqual(contract["semantic_changed_files"], [])
        self.assertEqual(
            contract["static_rejection_reason"],
            "INVALID_CANDIDATE_NO_SEMANTIC_CHANGE",
        )

    def test_python_module_assignment_change_requires_runtime_assertion(self):
        contract = build_candidate_activation_contract(
            before={"model.py": "LEARNING_RATE = 0.01\n"},
            after={"model.py": "LEARNING_RATE = 0.02\n"},
            run_id="module-value-r1",
        )
        self.assertTrue(contract["passed_static_contract"])
        self.assertEqual(contract["semantic_changed_files"], ["model.py"])
        self.assertEqual(contract["configuration_assertions"][0]["path"], "model.py")

    def test_yaml_formatting_and_comments_are_semantic_noop(self):
        contract = build_candidate_activation_contract(
            before={"configs/train.yaml": "weight_decay: 0.0005\n"},
            after={"configs/train.yaml": "# same value\nweight_decay: 0.000500\n"},
            run_id="yaml-noop-r1",
        )
        self.assertFalse(contract["passed_static_contract"])
        self.assertEqual(contract["semantic_noop_files"], ["configs/train.yaml"])
        self.assertEqual(
            contract["static_rejection_reason"],
            "INVALID_CANDIDATE_NO_SEMANTIC_CHANGE",
        )

    def test_non_python_config_requires_runtime_open_assertion(self):
        contract = build_candidate_activation_contract(
            before={"configs/train.yaml": "weight_decay: 0\n"},
            after={"configs/train.yaml": "weight_decay: 0.0005\n"},
            run_id="config-r1",
        )
        self.assertTrue(contract["passed_static_contract"])
        self.assertEqual(contract["mandatory_activation_targets"], [])
        self.assertEqual(contract["configuration_assertions"][0]["path"], "configs/train.yaml")

    def test_new_unreachable_callable_is_rejected_statically(self):
        contract = build_candidate_activation_contract(
            before={"model.py": "def train():\n    return 1\n"},
            after={"model.py": "def train():\n    return 1\n\ndef new_train():\n    return 2\n"},
            run_id="r1",
        )
        self.assertFalse(contract["passed_static_contract"])
        self.assertEqual(contract["unreachable_new_symbols"][0]["symbol"], "new_train")

    def test_wrong_marker_hash_is_invalid(self):
        contract = build_candidate_activation_contract(
            before={"model.py": "def train():\n    return 1\n"},
            after={"model.py": "def train():\n    return 2\n"},
            run_id="r1",
        )
        validation = validate_runtime_markers(contract, [{
            "run_id": "r1",
            "candidate_sha256": "wrong",
            "configuration_sha256": contract["configuration_sha256"],
            "path": "model.py",
            "symbol": "train",
        }])
        self.assertFalse(validation["passed"])
        self.assertEqual(validation["valid_marker_count"], 0)

    def test_harness_probe_proves_the_changed_callable_ran(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            source = repo / "model.py"
            source.write_text("def train():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "model.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            source.write_text(
                "def train():\n    return 1\n\ndef candidate_train():\n    return 2\n\n"
                "if __name__ == '__main__':\n    candidate_train()\n",
                encoding="utf-8",
            )
            contract = prepare_runtime_probe(
                repo=repo,
                target_files=["model.py"],
                run_id="runtime-proof",
                configuration={"seed": 7},
            )
            self.assertIsNotNone(contract)
            self.assertTrue(contract["passed_static_contract"])
            command = (
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 FML_EXPERIMENTAL_SEED=7 "
                "FML_CUDNN_DETERMINISTIC=1 FML_CUDNN_BENCHMARK=0 "
                + probe_environment_prefix()
                + "python3 model.py"
            )
            completed = subprocess.run(["bash", "-lc", command], cwd=repo, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            execution_dir = repo / "evidence"
            execution_dir.mkdir()
            validation = finalize_runtime_probe(repo=repo, execution_dir=execution_dir, contract=contract)
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["runtime_reproducibility_passed"])
            self.assertEqual(len(validation["runtime_reproducibility_evidence_sha256"]), 64)
            self.assertEqual(validation["requested_experimental_seeds"], [7])
            self.assertFalse(validation["task_seed_override_observed"])
            self.assertTrue((execution_dir / "runtime_reproducibility.json").is_file())
            markers = json.loads((execution_dir / "runtime_activation_markers.json").read_text())
            self.assertEqual(markers[0]["candidate_sha256"], contract["candidate_sha256"])

    def test_harness_probe_proves_candidate_config_was_opened(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            (repo / "configs").mkdir()
            config = repo / "configs" / "train.yaml"
            baseline = "weight_decay: 0\n"
            config.write_text("weight_decay: 0.0005\n", encoding="utf-8")
            contract = prepare_runtime_probe(
                repo=repo,
                target_files=["configs/train.yaml"],
                baseline_sources={"configs/train.yaml": baseline},
                run_id="config-runtime-proof",
                configuration={"seed": 7},
            )
            self.assertTrue(contract["passed_static_contract"])
            command = (
                "CUBLAS_WORKSPACE_CONFIG=:4096:8 FML_EXPERIMENTAL_SEED=7 "
                "FML_CUDNN_DETERMINISTIC=1 FML_CUDNN_BENCHMARK=0 "
                + probe_environment_prefix()
                + "python3 -c \"open('configs/train.yaml', encoding='utf-8').read()\""
            )
            completed = subprocess.run(["bash", "-lc", command], cwd=repo, text=True, capture_output=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            execution_dir = repo / "evidence"
            execution_dir.mkdir()
            validation = finalize_runtime_probe(repo=repo, execution_dir=execution_dir, contract=contract)
            self.assertTrue(validation["passed"])
            self.assertTrue(validation["runtime_reproducibility_passed"])
            markers = json.loads((execution_dir / "runtime_activation_markers.json").read_text())
            self.assertEqual(markers[0]["kind"], "CONFIGURATION_ASSERTION")

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork semantics")
    def test_harness_probe_excludes_forked_dataloader_workers(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            source = repo / "model.py"
            source.write_text("def train():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "model.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            source.write_text("def train():\n    return 2\n", encoding="utf-8")
            contract = prepare_runtime_probe(
                repo=repo,
                target_files=["model.py"],
                run_id="fork-worker-proof",
                configuration={"seed": 7},
            )
            script = repo / "fork_worker.py"
            script.write_text(
                "import os\n"
                "import model\n"
                "pid = os.fork()\n"
                "if pid == 0:\n"
                "    model.train()\n"
                "    os._exit(0)\n"
                "os.waitpid(pid, 0)\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(repo / ".fml_runtime_probe"),
                "FML_RUNTIME_MARKER_PATH": "results_tmp/runtime_activation.jsonl",
                "FML_EXPERIMENTAL_SEED": "7",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "FML_CUDNN_DETERMINISTIC": "1",
                "FML_CUDNN_BENCHMARK": "0",
            })
            completed = subprocess.run(
                ["python3", str(script)], cwd=repo, env=env, text=True, capture_output=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            execution_dir = repo / "evidence"
            execution_dir.mkdir()
            validation = finalize_runtime_probe(
                repo=repo, execution_dir=execution_dir, contract=contract
            )
            self.assertFalse(validation["passed"])
            self.assertEqual(validation["valid_marker_count"], 0)

    def test_harness_probe_attributes_subprocess_launched_trainer(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
            subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
            source = repo / "model.py"
            source.write_text("def train():\n    return 1\n", encoding="utf-8")
            subprocess.run(["git", "add", "model.py"], cwd=repo, check=True)
            subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
            source.write_text("def train():\n    return 2\n", encoding="utf-8")
            contract = prepare_runtime_probe(
                repo=repo,
                target_files=["model.py"],
                run_id="subprocess-trainer-proof",
                configuration={"seed": 7},
            )
            wrapper = repo / "wrapper.py"
            wrapper.write_text(
                "import subprocess, sys\n"
                "subprocess.run([sys.executable, '-c', "
                "'import model; model.train()'], check=True)\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": str(repo / ".fml_runtime_probe"),
                "FML_RUNTIME_MARKER_PATH": "results_tmp/runtime_activation.jsonl",
                "FML_EXPERIMENTAL_SEED": "7",
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                "FML_CUDNN_DETERMINISTIC": "1",
                "FML_CUDNN_BENCHMARK": "0",
            })
            completed = subprocess.run(
                ["python3", str(wrapper)], cwd=repo, env=env, text=True, capture_output=True
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            execution_dir = repo / "evidence"
            execution_dir.mkdir()
            validation = finalize_runtime_probe(
                repo=repo, execution_dir=execution_dir, contract=contract
            )
            self.assertTrue(validation["passed"])
            self.assertEqual(validation["valid_marker_count"], 1)


if __name__ == "__main__":
    unittest.main()
