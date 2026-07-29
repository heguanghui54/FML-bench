from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.catalog import build_catalog
from ml_scientist.fml_episode_bridge import import_completed_summaries, summary_to_adaptive_episode, summary_to_observational_episode
from ml_scientist.live_pipeline_report import _current_validation_preview, write_live_pipeline_report


ROOT = Path(__file__).resolve().parents[1]


def _summary() -> dict:
    return {
        "benchmark": "Generalization_domainbed",
        "agent": "autoresearch",
        "model": "test-model",
        "provider": "CodexCLI",
        "workspace_label": "pilot-autoresearch-Generalization_domainbed-trial01",
        "baseline_primary_metric": 0.27,
        "best_val_metric": 0.40,
        "total_steps": 1,
        "total_ideas": 1,
        "total_duration_seconds": 12.0,
        "token_usage": {"prompt_tokens": 90, "completion_tokens": 10, "total_tokens": 100},
        "agent_params": {"max_steps": 1},
        "val_steps": [{"step_id": 1, "primary_metric": 0.40, "val_result": {"success": True}}],
        "test_result": {"success": True, "primary_metric": 0.42},
    }


class FmlEpisodeBridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_legacy_summary_is_observational_and_never_gets_atomic_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "results" / "pilot" / "trial_01" / "autoresearch" / "Generalization_domainbed" / "run" / "summary.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
            episode = summary_to_observational_episode(
                summary_path, graph_snapshot_sha256="frozen", catalog=self.catalog,
            )
            self.assertEqual(episode["selected_strategy_id"], "strategy:autoresearch")
            self.assertEqual(episode["selected_policy_ids"], [])
            self.assertEqual(episode["policy_evaluations"], [])
            self.assertEqual(episode["learning_signals"], [])
            self.assertFalse(episode["protected_metric_used_for_routing"])
            self.assertNotIn("protected_test_metric", episode["next_state"]["observations"])

    def test_import_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary_path = root / "results" / "summary.json"
            summary_path.parent.mkdir()
            summary_path.write_text(json.dumps(_summary()), encoding="utf-8")
            graph = root / "graph.json"
            graph.write_text(json.dumps({"content_sha256": "frozen", "nodes": [], "edges": []}), encoding="utf-8")
            episodes = root / "episodes.jsonl"
            first = import_completed_summaries(
                results_root=root / "results", episodes_path=episodes, graph_path=graph, catalog=self.catalog,
            )
            second = import_completed_summaries(
                results_root=root / "results", episodes_path=episodes, graph_path=graph, catalog=self.catalog,
            )
            self.assertEqual(first["imported_count"], 1)
            self.assertEqual(second["imported_count"], 0)
            self.assertEqual(second["total_episode_count"], 1)

    def test_adaptive_summary_preserves_explicit_policy_trace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = _summary()
            payload["agent"] = "adaptive_pipeline"
            payload["metadata"] = {"adaptive_pipeline": {
                "graph_content_sha256": "frozen",
                "decision_contracts": [{
                    "decision": {
                        "decision_id": "decision:adaptive",
                        "selected_policy_ids": ["policy:aide:proposal"],
                        "selected_policy_by_role": {"proposal": "policy:aide:proposal"},
                    },
                    "planner_output": {"action": "change train"},
                    "operator": {"budget": {"candidate_steps": 1}},
                }],
                "post_execution_assessments": [{
                    "artifact_path": "assessment.json",
                    "policy_evaluations": [{
                        "policy_id": "policy:aide:proposal", "role": "proposal",
                        "applied": True, "status": "PASSED",
                        "evidence_artifact_refs": ["validation.json"],
                        "gate_checks": ["candidate_executed"],
                        "learning_signal": {"intent": "NONE"},
                    }],
                }],
                "reachability_audits": [{"passed": True, "artifact_path": "reachability.json"}],
            }}
            summary_path = root / "summary.json"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            episode = summary_to_adaptive_episode(
                summary_path, graph_snapshot_sha256="frozen", catalog=self.catalog,
            )
            self.assertEqual(episode["selected_policy_ids"], ["policy:aide:proposal"])
            self.assertEqual(episode["policy_evaluations"][0]["status"], "PASSED")
            self.assertEqual(episode["learning_signals"], [{"intent": "NONE"}])
            self.assertEqual(episode["attribution_scope"], "explicit_atomic_policy_trace_only")


class LivePipelineReportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_report_labels_fallbacks_and_unexecuted_adaptive_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results" / "pilot" / "trial_01" / "autoresearch" / "Generalization_domainbed" / "run"
            results.mkdir(parents=True)
            results.joinpath("summary.json").write_text(json.dumps(_summary()), encoding="utf-8")
            campaign = root / "campaign"
            campaign.mkdir()
            campaign.joinpath("campaign_state.json").write_text(json.dumps({
                "processed_run_count": 1, "selected_run_count": 2,
                "remaining_run_count": 1, "current_run_id": "next", "complete": False,
            }), encoding="utf-8")
            campaign.joinpath("campaign_events.jsonl").write_text("", encoding="utf-8")
            matrix = root / "matrix.csv"
            with matrix.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["phase"])
                writer.writeheader(); writer.writerows([{"phase": "pilot"}, {"phase": "pilot"}])
            graph = root / "graph.json"
            graph.write_text(json.dumps({
                "content_sha256": "frozen", "nodes": [
                    {"node_type": "Policy"}, {"node_type": "Strategy"}, {"node_type": "Skill"},
                ], "edges": [{}],
            }), encoding="utf-8")
            issues = root / "issues.json"
            issues.write_text(json.dumps({"issues": [{
                "id": "P001", "scope": "current_pipeline", "status": "OPEN", "severity": "critical",
                "problem": "missing executor", "evidence": "none", "recommended_change": "add bridge",
            }]}), encoding="utf-8")
            episodes = root / "episodes.jsonl"
            episodes.write_text("{}\n", encoding="utf-8")
            out = root / "out"
            adaptive = root / "adaptive-missing"
            adaptive.mkdir()
            adaptive.joinpath("validation_milestone.json").write_text(json.dumps({
                "status": "VALIDATION_COMPLETE_REPEAT_BUDGET_ADMISSION_BLOCKED",
                "paper_claim_eligible": False,
                "claim_boundary": "visible validation only",
                "visible_validation": {"score": 0.5},
            }), encoding="utf-8")
            report = write_live_pipeline_report(
                results_root=root / "results", campaign_dir=campaign, matrix_path=matrix,
                graph_path=graph, issue_registry_path=issues, out_dir=out, catalog=self.catalog,
                new_pipeline_results_root=adaptive, episode_buffer_path=episodes,
            )
            self.assertEqual(report["new_pipeline"]["status"], "NOT_RUN_EXECUTOR_READY_AWAITING_FROZEN_BASELINE_GATE")
            self.assertTrue(report["new_pipeline"]["executor_adapter_present"])
            self.assertEqual(report["memory"]["policies"], 1)
            self.assertEqual(report["memory"]["skills"], 1)
            self.assertEqual(report["memory"]["buffered_observational_episodes"], 1)
            self.assertTrue((out / "live-dashboard.html").is_file())
            self.assertTrue((out / "manifest.json").is_file())
            self.assertIn("不能把 AdaptiveSearch", (out / "live-dashboard.html").read_text(encoding="utf-8"))
            self.assertEqual(report["validation_milestones"]["count"], 1)
            self.assertIn(
                "VALIDATION_COMPLETE_REPEAT_BUDGET_ADMISSION_BLOCKED",
                (out / "live-dashboard.html").read_text(encoding="utf-8"),
            )

    def test_current_validation_preview_is_explicitly_provisional(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val = root / "pilot" / "trial_01" / "adaptivesearch" / "Privacy_privacymeter" / "run" / "val_info.json"
            val.parent.mkdir(parents=True)
            val.write_text(json.dumps({
                "cifar10": {"means": {
                    "AUC_gap_mean": 0.2134,
                    "test_acc_mean": 0.65292,
                    "constraint_violated": False,
                }},
            }), encoding="utf-8")
            preview = _current_validation_preview(
                root,
                "pilot-adaptivesearch-Privacy_privacymeter-trial01",
                self.catalog,
            )
            self.assertEqual(preview["status"], "PROVISIONAL_VALIDATION_ONLY")
            self.assertEqual(preview["metrics"]["AUC_gap_mean"], 0.2134)
            self.assertIn("not a protected-test", preview["note"])


if __name__ == "__main__":
    unittest.main()
