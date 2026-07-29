from __future__ import annotations

import unittest
from types import SimpleNamespace

from agents.base import BaseAgent


class _Agent(BaseAgent):
    def initialize(self) -> None:
        pass

    def run(self, task_description=None, target_files=None, baseline_results=None):
        raise NotImplementedError


class _Executor:
    def __init__(self, pre_test_success: bool):
        self.timeout = 10
        self.pre_test_success = pre_test_success
        self.val_calls = []
        self.test_calls = []

    def run_val(self, run_id):
        self.val_calls.append(run_id)
        return {
            "success": self.pre_test_success,
            "results": {} if self.pre_test_success else None,
            "primary_metric": 1.0 if self.pre_test_success else None,
            "error": None if self.pre_test_success else "constraint failed",
        }

    def run_test(self, run_id):
        self.test_calls.append(run_id)
        return {"success": True, "primary_metric": 2.0}


class ProtectedTestGateTests(unittest.TestCase):
    @staticmethod
    def _agent(executor):
        agent = object.__new__(_Agent)
        agent.executor = executor
        agent.best_code_snapshot = None
        agent.config = SimpleNamespace(runtime_params={"reproducibility_contract": {}})
        agent._selected_validation_result = {
            "success": True,
            "results": {},
            "primary_metric": 1.0,
            "error": None,
        }
        return agent

    def test_failed_pre_test_validation_blocks_protected_test(self):
        executor = _Executor(pre_test_success=False)
        result = self._agent(executor)._execute_test()
        self.assertFalse(result["success"])
        self.assertTrue(result["error"].startswith("PRE_TEST_VAL_FAILED:"))
        self.assertEqual(executor.val_calls, ["pre_test_val"])
        self.assertEqual(executor.test_calls, [])
        self.assertEqual(executor.timeout, 10)

    def test_successful_pre_test_validation_allows_protected_test(self):
        executor = _Executor(pre_test_success=True)
        result = self._agent(executor)._execute_test()
        self.assertTrue(result["success"])
        self.assertEqual(executor.test_calls, ["final_test"])
        self.assertEqual(executor.timeout, 10)


if __name__ == "__main__":
    unittest.main()
