from __future__ import annotations

import json
import hashlib
import tempfile
import unittest
from pathlib import Path

from agents.adaptive_pipeline.agent import AdaptivePipelineAgent
from agents.base import AgentConfig, AgentType

from ml_scientist.adaptive_fml_executor import (
    audit_modified_code_reachability,
    build_adaptive_fml_operator,
    enforce_resource_budget,
    validate_post_execution_policy_evaluations,
)
from ml_scientist.strategy_operator import compile_policy_assessment_operator


ROOT = Path(__file__).resolve().parents[1]


class AdaptiveFmlExecutorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.graph = json.loads((ROOT / "artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json").read_text(encoding="utf-8"))
        cls.plan = json.loads((ROOT / "artifacts/ml_scientist/bootstrap/plans/Privacy_privacymeter.json").read_text(encoding="utf-8"))

    def test_builds_distinct_atomic_policy_executor_contract(self):
        payload = build_adaptive_fml_operator(
            plan=self.plan,
            graph=self.graph,
            run_id="adaptive-pilot-1",
            node_id="hypothesis-frontier",
            visible_evidence=[{
                "metric_name": "AUC_gap_mean", "value": 0.5,
                "observability": "ONLINE_VISIBLE", "split": "validation",
            }],
            budget_before={"remaining_fraction": 1.0, "candidate_steps": 1, "token_budget": 100000},
        )
        self.assertEqual(payload["output_label"], "adaptive_pipeline")
        self.assertTrue(payload["baseline_alias_forbidden"])
        selected = payload["decision"]["selected_policy_by_role"]
        self.assertEqual(set(selected), {"state", "proposal", "selection", "memory", "diversity"})
        self.assertGreater(len({policy_id.split(":")[1] for policy_id in selected.values()}), 1)
        self.assertIn("known_bundle_bonus=0", payload["decision"]["rationale"])
        self.assertTrue(payload["operator"]["policy_trace_required"])
        self.assertEqual(payload["operator"]["policy_trace_phase"], "POST_EXECUTION_ONLY")
        self.assertNotIn("CREATE|PATCH", payload["operator"]["instruction"])
        self.assertEqual(len(payload["contract_sha256"]), 64)

    def test_learning_signal_is_compiled_only_after_real_artifacts_exist(self):
        payload = build_adaptive_fml_operator(
            plan=self.plan, graph=self.graph, run_id="adaptive-pilot-assess",
            node_id="hypothesis-frontier", visible_evidence=[],
            budget_before={"remaining_fraction": 1.0, "candidate_steps": 1},
        )
        node = next(row for row in self.plan["nodes"] if row["node_id"] == "hypothesis-frontier")
        assessment = compile_policy_assessment_operator(
            graph=self.graph,
            selected_policy_by_role=payload["decision"]["selected_policy_by_role"],
            node=node,
            planned_policy_use=[{"role": "proposal", "planned_use": "generate one candidate"}],
            outcome="PASSED_GATE",
            evidence_artifacts=[{"path": "validation.json", "sha256": "abc"}],
            state_before={"baseline_metric": 0.5},
            next_state={"validation_metric": 0.4},
            resource_telemetry={"tokens_consumed": 100},
        )
        self.assertEqual(assessment["learning_signal_phase"], "POST_EXECUTION_ONLY")
        self.assertIn("CREATE|PATCH", assessment["instruction"])
        self.assertEqual(assessment["evidence_artifact_count"], 1)

    def test_protected_test_evidence_is_filtered_before_prompt(self):
        payload = build_adaptive_fml_operator(
            plan=self.plan,
            graph=self.graph,
            run_id="adaptive-pilot-2",
            node_id="hypothesis-frontier",
            visible_evidence=[{
                "metric_name": "AUC_gap_mean", "value": 0.1,
                "observability": "PROTECTED_FINAL_ONLY", "split": "test",
            }],
            budget_before={"remaining_fraction": 1.0, "candidate_steps": 1},
        )
        self.assertEqual(len(payload["decision"]["visible_evidence"]), 0)
        self.assertFalse(payload["operator"]["protected_evidence_included"])

    def test_static_gate_catches_unreachable_new_training_function(self):
        before = {"models/utils.py": "def train():\n    return 1\n\ntrain()\n"}
        after = {"models/utils.py": "def train():\n    return 1\n\ndef crossfit_margin_train():\n    return 2\n\ntrain()\n"}
        report = audit_modified_code_reachability(before, after)
        self.assertFalse(report["passed"])
        self.assertEqual(report["uncalled_new_functions"], ["crossfit_margin_train"])

    def test_static_gate_accepts_direct_existing_entrypoint_modification(self):
        before = {"train.py": "def train():\n    return 1\n\ntrain()\n"}
        after = {"train.py": "def train():\n    return 1 + 1\n\ntrain()\n"}
        self.assertTrue(audit_modified_code_reachability(before, after)["passed"])

    def test_budget_gate_stops_on_any_exhausted_dimension(self):
        report = enforce_resource_budget(
            tokens_consumed=100, wall_clock_seconds=5, candidate_steps_consumed=0,
            budget={"token_budget": 100, "wall_clock_seconds": 10, "candidate_steps": 1},
        )
        self.assertFalse(report["passed"])
        self.assertEqual(report["route"], "STOP_BUDGET_EXHAUSTED")

    def test_post_execution_credit_requires_exact_roles_and_real_artifacts(self):
        selected = {"proposal": "policy:aide:proposal"}
        artifacts = [{"path": "validation.json", "sha256": "abc"}]
        rows = validate_post_execution_policy_evaluations(
            payload={"policy_evaluations": [{
                "policy_id": "policy:aide:proposal", "role": "proposal",
                "applied": True, "status": "PASSED",
                "evidence_artifact_refs": ["validation.json"],
                "gate_checks": ["candidate_executed"],
                "learning_signal": {"intent": "NONE"},
            }]},
            selected_policy_by_role=selected,
            evidence_artifacts=artifacts,
        )
        self.assertEqual(rows[0]["status"], "PASSED")
        with self.assertRaises(ValueError):
            validate_post_execution_policy_evaluations(
                payload={"policy_evaluations": [{
                    "policy_id": "policy:aide:proposal", "role": "proposal",
                    "applied": True, "status": "PASSED",
                    "evidence_artifact_refs": ["invented.json"],
                    "gate_checks": ["candidate_executed"],
                    "learning_signal": {"intent": "NONE"},
                }]},
                selected_policy_by_role=selected,
                evidence_artifacts=artifacts,
            )

    def test_agent_loads_task_plan_after_runner_runtime_injection(self):
        agent = AdaptivePipelineAgent(AgentConfig(
            agent_type=AgentType.ADAPTIVE_PIPELINE,
            agent_params={
                "graph_path": str(ROOT / "artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json"),
                "plan_root": str(ROOT / "artifacts/ml_scientist/bootstrap/plans"),
            },
            runtime_params={"benchmark_name": "Generalization_domainbed"},
        ))
        agent._load_adaptive_context()
        self.assertEqual(agent.plan["task_id"], "Generalization_domainbed")
        self.assertTrue(agent.graph.get("nodes"))

    def test_artifact_descriptor_supplies_hash_and_auditable_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val_info.json"
            path.write_text('{"metric": 0.7}\n', encoding="utf-8")
            row = AdaptivePipelineAgent._artifact_descriptor(
                path, "validation_result", {"primary_metric": 0.7},
            )
        self.assertEqual(row["sha256"], hashlib.sha256(b'{"metric": 0.7}\n').hexdigest())
        self.assertEqual(row["content_summary"]["primary_metric"], 0.7)
        self.assertGreater(row["size_bytes"], 0)

    def test_constraint_failure_is_not_classified_as_execution_crash(self):
        error = (
            "Average result: AUC 0.8465, TPR@0.1%FPR of 0.0065, TPR@0.0%FPR of 0.0000\n"
            "ValueError: Invalid results: TPR constraint violated: TPR@0.1%=0.012, "
            "TPR@0.0%=0.0 AND Accuracy constraint violated: 0.59276 < 0.61\n"
        )
        failure = AdaptivePipelineAgent._classify_validation_failure(error)
        self.assertEqual(failure["kind"], "CONSTRAINT_FAILED")
        self.assertAlmostEqual(failure["observed"]["auc_gap_mean"], 0.3465)
        self.assertEqual(failure["observed"]["tpr_at_0_1_fpr_gate_value"], 0.012)
        self.assertEqual(failure["observed"]["test_acc_mean"], 0.59276)
        self.assertIn("Accuracy constraint violated", failure["violation_reason"])

    def test_ordinary_runtime_error_remains_execution_failure(self):
        failure = AdaptivePipelineAgent._classify_validation_failure("CUDA out of memory")
        self.assertEqual(failure["kind"], "EXECUTION_FAILED")


if __name__ == "__main__":
    unittest.main()
