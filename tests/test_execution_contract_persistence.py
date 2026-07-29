import json
from pathlib import Path
import tempfile
import unittest

from benchmark.executor import BenchmarkExecutor
from benchmark.runner import collect_persisted_execution_contracts
from ml_scientist.execution_contracts import BudgetLedger, activate_budget_ledger


class ExecutionContractPersistenceTests(unittest.TestCase):
    def _executor(self, workspace, counts, marker):
        executor = BenchmarkExecutor({
            "repo_dir": str(workspace),
            "conda_env": "unused",
            "_experimental_seed": 4409,
        })
        executor.workspace_dir = str(workspace)
        executor.execution_counts.update(counts)
        executor.activation_records.append({"phase": marker})
        executor.gpu_telemetry_summaries.append({"phase": marker})
        return executor

    def test_cleanup_boundary_evidence_can_be_merged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            search = self._executor(
                root / "search",
                {"candidate_validation_count": 1},
                "val",
            )
            final = self._executor(
                root / "final",
                {"pre_test_validation_count": 1, "protected_test_count": 1},
                "pre_test_val",
            )
            search._persist_execution_contract_summary()
            final._persist_execution_contract_summary()

            merged = collect_persisted_execution_contracts(str(root))

        self.assertEqual(merged["execution_counts"]["candidate_validation_count"], 1)
        self.assertEqual(merged["execution_counts"]["pre_test_validation_count"], 1)
        self.assertEqual(merged["execution_counts"]["protected_test_count"], 1)
        self.assertEqual([row["phase"] for row in merged["activation_records"]], ["pre_test_val", "val"])
        self.assertEqual(len(merged["executor_summaries"]), 2)

    def test_budget_snapshot_survives_context_deactivation(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = self._executor(Path(tmp), {}, "val")
            ledger = BudgetLedger()
            with activate_budget_ledger(ledger):
                ledger.before_execution("val")
                executor._capture_budget_ledger_snapshot()
            summary = executor.execution_contract_summary()

        self.assertIsNotNone(summary["budget_ledger"])
        self.assertEqual(summary["budget_ledger"]["usage"]["candidate_validation_count"], 1)

    def test_materialized_phase_artifacts_recover_missing_executor_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "search" / "run_0" / "execution_1"
            repeat = root / "final" / "run_pre_test_val" / "execution_2"
            test = root / "final" / "run_final_test" / "execution_3"
            for path, phase in ((val, "val"), (repeat, "pre_test_val"), (test, "test")):
                path.mkdir(parents=True)
                path.joinpath("runtime_activation_validation.json").write_text(
                    json.dumps({"phase": phase, "passed": True}), encoding="utf-8"
                )
                path.joinpath("gpu_telemetry_summary.json").write_text(
                    json.dumps({"phase": phase, "gpu_active_seconds": 5}), encoding="utf-8"
                )
            merged = collect_persisted_execution_contracts(str(root))

        self.assertEqual(merged["execution_counts"]["candidate_validation_count"], 1)
        self.assertEqual(merged["execution_counts"]["pre_test_validation_count"], 1)
        self.assertEqual(merged["execution_counts"]["protected_test_count"], 1)
        self.assertEqual(len(merged["gpu_telemetry_summaries"]), 3)


if __name__ == "__main__":
    unittest.main()
