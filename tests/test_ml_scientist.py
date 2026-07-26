from __future__ import annotations

import tempfile
import unittest
import csv
import json
from pathlib import Path

from ml_scientist.catalog import build_catalog, load_simple_yaml
from ml_scientist.campaign import CampaignError, load_run_matrix, run_campaign
from ml_scientist.experiment_design import build_experiment_protocol, preflight_environment, write_experiment_protocol
from ml_scientist.governance import EvidenceGateError, EvidenceGatedSkillRegistry, initialize_governance_artifacts
from ml_scientist.planner import build_research_paper_plan, validate_plan
from ml_scientist.published_prior import (
    agent_summary_rows,
    agent_task_rows,
    process_rows,
    task_card_rows,
    write_published_prior,
)
from ml_scientist.reporting import paired_agent_comparisons, summarize_agent_performance, write_catalog_artifacts, write_experiment_artifacts


ROOT = Path(__file__).resolve().parents[1]


class CatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_learns_complete_repository_inventory(self):
        self.assertEqual(self.catalog["counts"]["agents"], 7)
        self.assertEqual(self.catalog["counts"]["tasks"], 18)
        self.assertEqual(self.catalog["counts"]["lite_tasks"], 8)
        self.assertEqual(self.catalog["counts"]["process_metrics"], 12)
        self.assertEqual(len(self.catalog["metrics"]), 30)

    def test_agent_profiles_are_source_grounded(self):
        for agent in self.catalog["agents"]:
            self.assertTrue((ROOT / agent["source"]).is_file())
            self.assertEqual(len(agent["source_sha256"]), 64)
            self.assertTrue(agent["selection_policy"])
            self.assertTrue(agent["memory_policy"])

    def test_task_contracts_have_baselines_and_test_boundary(self):
        for task in self.catalog["tasks"]:
            self.assertIsNotNone(task["baseline_validation"], task["task_id"])
            self.assertIsNotNone(task["baseline_test"], task["task_id"])
            self.assertTrue(task["validation_command"])
            self.assertTrue(task["protected_test_command"])
            self.assertTrue((ROOT / task["task_config"]).is_file())

    def test_simple_yaml_parser_preserves_agent_parameters(self):
        parsed = load_simple_yaml(ROOT / "configs/agents/ai_scientist_v2.yaml")
        params = parsed["agent"]["ai_scientist_v2"]
        self.assertEqual(params["num_ideas"], 3)
        self.assertEqual(params["stage_budgets"], [0.1, 0.2, 0.5, 0.2])


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)
        cls.plan = build_research_paper_plan(cls.catalog["tasks"][0], cls.catalog["agents"])

    def test_plan_is_valid_and_paper_is_evidence_locked(self):
        validate_plan(self.plan)
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        self.assertEqual(nodes["final-paper-writing"]["status"], "LOCKED_DEPENDENCIES")
        self.assertIn("final-evidence-bundle", nodes["final-paper-writing"]["depends_on"])
        self.assertFalse(nodes["protected-final-test"]["metadata"]["feedback_to_search"])

    def test_all_nodes_have_inner_review_loops(self):
        for node in self.plan["nodes"]:
            self.assertTrue(node["inner_loop"]["reviewers"], node["node_id"])
            self.assertTrue(node["inner_loop"]["hard_gates"], node["node_id"])

    def test_four_skill_channels_are_governed(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        self.assertEqual(
            set(nodes["skill-evolution"]["metadata"]["skill_channels"]),
            {"research_execution", "research_review", "paper_writing", "paper_review"},
        )

    def test_review_can_trigger_versioned_upstream_amendments(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        triage = nodes["review-amendment-triage"]
        self.assertIn("code-modification", triage["metadata"]["amendment_targets"])
        self.assertIn("visible-validation", triage["metadata"]["amendment_targets"])
        self.assertIn("new hidden evaluation", triage["metadata"]["protected_test_rule"])

    def test_publication_has_two_integrity_gates_and_rereview(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        self.assertIn("paper-integrity", nodes)
        self.assertIn("paper-re-review", nodes)
        self.assertIn("final-paper-integrity", nodes)
        self.assertEqual(nodes["paper-finalize"]["depends_on"], ["final-paper-integrity"])
        self.assertTrue(self.plan["academic_publication_contract"]["final_integrity_restarts_from_scratch"])


class ReportingTests(unittest.TestCase):
    def test_catalog_figures_and_empty_experiment_tables_are_honest(self):
        catalog = build_catalog(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            manifest = write_catalog_artifacts(catalog, out / "catalog")
            self.assertEqual(len(manifest["figures"]), 3)
            for figure in manifest["figures"]:
                self.assertTrue((out / "catalog" / figure["path"]).is_file())
            status = write_experiment_artifacts(out / "missing-results", out / "experiments")
            self.assertEqual(status["record_count"], 0)
            self.assertFalse(status["experimental_figures_emitted"])

    def test_real_replicates_get_fml_normalization_uncertainty_and_figure(self):
        catalog = build_catalog(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results = root / "results"
            metric_reports = root / "metric_reports"
            for trial in range(1, 4):
                run = results / "confirmatory_lite" / f"trial_{trial:02d}" / "autoresearch" / "Fairness_fairlearn" / "run"
                run.mkdir(parents=True)
                payload = {
                    "benchmark": "Fairness_fairlearn",
                    "agent": "autoresearch",
                    "model": "test-model",
                    "provider": "OpenAI",
                    "experimental_seed": trial,
                    "best_val_metric": 0.05,
                    "test_result": {"success": True, "primary_metric": 0.04 + trial * 0.001},
                    "total_steps": 10,
                    "total_ideas": 10,
                    "total_duration_seconds": 100,
                    "token_usage": {"total_tokens": 1000},
                }
                (run / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
                process = metric_reports / "confirmatory_lite" / f"trial_{trial:02d}" / "autoresearch" / "table_process_metrics.csv"
                process.parent.mkdir(parents=True)
                with process.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["task", "Exploration Spread", "Wall-clock time (h)"])
                    writer.writerow(["Fairness_fairlearn", 0.1 * trial, 1.0 + trial])
                    writer.writerow(["MEAN", 0.1 * trial, 1.0 + trial])
            status = write_experiment_artifacts(results, root / "report", catalog, metric_reports)
            self.assertEqual(status["record_count"], 3)
            self.assertEqual(status["replicated_group_count"], 1)
            self.assertEqual(status["process_metric_record_count"], 6)
            self.assertEqual(status["process_metric_figure_count"], 2)
            self.assertTrue(status["experimental_figures_emitted"])
            self.assertTrue((root / "report" / "figures" / "replicated_normalized_improvement.svg").is_file())

    def test_overall_agent_estimates_require_complete_task_blocks(self):
        records = []
        for agent, offset in (("a", 0.1), ("b", 0.0)):
            for trial in range(1, 4):
                for task_index in range(2):
                    records.append(
                        {
                            "phase": "pilot",
                            "agent": agent,
                            "model": "m",
                            "provider": "p",
                            "trial": trial,
                            "task": f"task-{task_index}",
                            "normalized_improvement": offset + 0.01 * trial + 0.001 * task_index,
                        }
                    )
        trial_rows, overall = summarize_agent_performance(records)
        self.assertEqual(sum(row["complete_task_block"] for row in trial_rows), 6)
        self.assertEqual(len(overall), 2)
        self.assertTrue(all(row["complete_trial_n"] == 3 for row in overall))
        pairs = paired_agent_comparisons(records)
        self.assertEqual(len(pairs), 1)
        self.assertAlmostEqual(pairs[0]["mean_difference_left_minus_right"], 0.1)


class PublishedPriorTests(unittest.TestCase):
    def test_complete_published_tables_are_transcribed(self):
        catalog = build_catalog(ROOT)
        self.assertEqual(len(agent_task_rows()), 18 * 6)
        self.assertEqual(len(agent_summary_rows()), 7)
        self.assertEqual(len(process_rows()), 12)
        self.assertEqual(len(task_card_rows()), 18)
        adaptive = next(row for row in agent_summary_rows() if row["agent_id"] == "adaptivesearch")
        self.assertEqual(adaptive["mean_normalized_test_improvement"], 0.208)
        self.assertEqual(adaptive["pairwise_win_rate_percent"], 58.6)
        auc = next(row for row in process_rows() if row["metric"] == "AUC-over-steps")
        self.assertEqual(auc["pooled_spearman_rho"], 0.784)
        self.assertTrue(auc["significant_unadjusted_p_lt_0_05"])
        self.assertEqual({row["task_id"] for row in task_card_rows()}, {task["task_id"] for task in catalog["tasks"]})

    def test_published_prior_is_labeled_and_separated_from_new_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_published_prior(root)
            self.assertEqual(manifest["status"], "published_prior_only")
            self.assertFalse(manifest["separation_contract"]["merge_with_new_experiment_rows"])
            self.assertEqual(manifest["row_counts"]["agent_task"], 108)
            self.assertTrue((root / "figures" / "published_agent_task_heatmap.svg").is_file())
            self.assertTrue((root / "published_evaluation_contract.json").is_file())
            self.assertTrue((root / "paper_version_lineage.csv").is_file())
            self.assertTrue((root / "provenance_manifest.json").is_file())


class ExperimentDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_protocol_is_balanced_and_separates_trials(self):
        protocol, rows = build_experiment_protocol(self.catalog, model="fixed-model")
        self.assertEqual(protocol["phase_run_counts"]["pilot"], 14)
        self.assertEqual(protocol["phase_run_counts"]["confirmatory_lite"], 168)
        self.assertEqual(protocol["total_planned_runs"], 182)
        confirmatory = [row for row in rows if row["phase"] == "confirmatory_lite"]
        self.assertEqual(len({(row["agent"], row["task"], row["trial"]) for row in confirmatory}), 168)
        self.assertTrue(all("--seed" in row["command"] for row in rows))
        self.assertTrue(all(f"trial_{row['trial']:02d}" in row["result_root"] for row in rows))

    def test_preflight_reports_key_presence_without_key_value(self):
        report = preflight_environment(self.catalog, provider="OpenAI", model="SET_MODEL", repo=ROOT)
        self.assertIn("required_key_present", report)
        self.assertNotIn("key_value", report)
        self.assertFalse(report["ready"])

    def test_llm_calls_use_the_frozen_trial_seed(self):
        source = (ROOT / "agents" / "llm.py").read_text(encoding="utf-8")
        self.assertNotIn("seed=0", source)
        self.assertIn("seed=experiment_seed()", source)

    def test_campaign_is_resumable_and_refuses_placeholder_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_experiment_protocol(self.catalog, root / "template")
            with self.assertRaises(CampaignError):
                load_run_matrix(root / "template" / "run_matrix.csv")
            write_experiment_protocol(self.catalog, root / "frozen", model="fixed-model")
            state = run_campaign(
                matrix_path=root / "frozen" / "run_matrix.csv",
                repo=ROOT,
                log_dir=root / "logs",
                phases={"pilot"},
                max_runs=2,
                dry_run=True,
            )
            self.assertEqual(state["selected_run_count"], 2)
            self.assertEqual(state["status_counts"], {"DRY_RUN_READY": 2})
            self.assertFalse(state["complete"])


class SkillGovernanceTests(unittest.TestCase):
    def test_promotion_requires_real_downstream_evidence_and_distinct_contexts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.json"
            source.write_text('{"kind":"source audit"}', encoding="utf-8")
            outcome_a = root / "outcome-a.json"
            outcome_a.write_text('{"passed":true}', encoding="utf-8")
            outcome_b = root / "outcome-b.json"
            outcome_b.write_text('{"passed":true}', encoding="utf-8")
            registry = EvidenceGatedSkillRegistry(root / "registry", "commit")
            registry.propose(
                skill_id="hs-test-rule",
                channel="research_execution",
                rule="A testable rule",
                evidence_paths=[source],
                acceptance_gates=["downstream_pass"],
                rollback_condition="comparable regression",
            )
            with self.assertRaises(EvidenceGateError):
                registry.promote("hs-test-rule")
            registry.record_outcome(
                skill_id="hs-test-rule", evidence_path=outcome_a, context_id="task-a",
                real_downstream_complete=True, no_regression=True, metrics={"score": 1},
            )
            self.assertEqual(registry.promote("hs-test-rule"), "provisional_success")
            with self.assertRaises(EvidenceGateError):
                registry.promote("hs-test-rule")
            registry.record_outcome(
                skill_id="hs-test-rule", evidence_path=outcome_b, context_id="task-b",
                real_downstream_complete=True, no_regression=True, metrics={"score": 2},
            )
            self.assertEqual(registry.promote("hs-test-rule"), "repeated_success")
            snapshot = registry.snapshot(root / "snapshot.json")
            self.assertFalse(snapshot["pending_writes_visible"])
            self.assertEqual(snapshot["active_versions"]["hs-test-rule"], 1)

    def test_bootstrap_does_not_invent_skills(self):
        catalog = build_catalog(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            status = initialize_governance_artifacts(catalog, Path(tmp))
            self.assertEqual(status["candidate_skill_count"], 0)
            self.assertEqual(status["active_skill_count"], 0)


if __name__ == "__main__":
    unittest.main()
