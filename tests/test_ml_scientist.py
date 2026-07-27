from __future__ import annotations

import tempfile
import unittest
import csv
import hashlib
import json
import sys
from pathlib import Path

from ml_scientist.catalog import build_catalog, load_simple_yaml
from ml_scientist.campaign import CampaignError, _command_argv, load_run_matrix, run_campaign
from ml_scientist.experiment_design import build_experiment_protocol, preflight_environment, write_experiment_protocol
from ml_scientist.governance import EvidenceGateError, EvidenceGatedSkillRegistry, initialize_governance_artifacts
from ml_scientist.knowledge_base import build_agent_dossiers, build_metric_implementation_audit, build_task_dossiers, write_knowledge_base
from ml_scientist.planner import build_research_paper_plan, validate_plan
from ml_scientist.paper_evaluation import build_paper_evaluation_protocol, evaluate_paper_evaluation, write_paper_evaluation_protocol
from ml_scientist.paper_package import write_paper_package
from ml_scientist.published_prior import (
    agent_summary_rows,
    agent_task_rows,
    derived_agent_dispersion_rows,
    derived_task_discrimination_rows,
    process_rows,
    published_guidance_for_task,
    task_card_rows,
    write_published_prior,
)
from ml_scientist.reporting import (
    adaptive_opportunity_interactions,
    collect_scorer_semantics_sensitivity,
    paired_agent_comparisons,
    paired_agent_task_effects,
    summarize_agent_performance,
    write_catalog_artifacts,
    write_experiment_artifacts,
)
from ml_scientist.statistical_analysis import build_statistical_analysis_protocol


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
        metrics = {row["name"]: row for row in self.catalog["metrics"] if row["category"] != "task_performance"}
        self.assertIn("every persisted step-snapshot", metrics["Exploration Spread"]["definition"])
        self.assertIn("Last successful recorded step", metrics["Best-improvement step"]["definition"])

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

    def test_published_prior_guides_planning_without_unlocking_claims(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        guidance = nodes["method-frontier"]["metadata"]["published_prior_guidance"]
        self.assertEqual(guidance["task_id"], self.catalog["tasks"][0]["task_id"])
        self.assertFalse(guidance["claim_unlock_eligible"])
        self.assertIn("published and local evidence classes are not merged", nodes["paper-peer-review"]["metadata"]["published_prior_checks"])

    def test_review_can_trigger_versioned_upstream_amendments(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        triage = nodes["review-amendment-triage"]
        self.assertIn("code-modification", triage["metadata"]["amendment_targets"])
        self.assertIn("visible-validation", triage["metadata"]["amendment_targets"])
        self.assertIn("confirmatory-analysis", triage["metadata"]["amendment_targets"])
        self.assertIn("new hidden evaluation", triage["metadata"]["protected_test_rule"])

    def test_confirmatory_analysis_unlocks_final_evidence_not_search(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        analysis = nodes["confirmatory-analysis"]
        self.assertEqual(analysis["depends_on"], ["protected-final-test"])
        self.assertIn("holm_pairwise_adjustment", analysis["unlock_conditions"])
        self.assertEqual(nodes["final-evidence-bundle"]["depends_on"], ["confirmatory-analysis"])

    def test_publication_has_two_integrity_gates_and_rereview(self):
        nodes = {node["node_id"]: node for node in self.plan["nodes"]}
        self.assertIn("paper-integrity", nodes)
        self.assertIn("paper-re-review", nodes)
        self.assertIn("final-paper-integrity", nodes)
        self.assertEqual(nodes["paper-finalize"]["depends_on"], ["final-paper-integrity"])
        self.assertTrue(self.plan["academic_publication_contract"]["final_integrity_restarts_from_scratch"])
        self.assertIn("separate local extension", nodes["paper-integrity"]["metadata"]["paper_metric_boundary"])


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
        self.assertEqual(pairs[0]["complete_paired_trial_n"], 3)
        self.assertEqual(pairs[0]["seed_block_wins_left"], 3)
        self.assertEqual(pairs[0]["exact_sign_p_seed_blocks"], 0.25)
        self.assertEqual(pairs[0]["holm_p_seed_blocks"], 0.25)
        self.assertIsNone(pairs[0]["cohen_dz"])
        task_effects = paired_agent_task_effects(records)
        self.assertEqual(len(task_effects), 2)
        self.assertTrue(all(row["matched_seed_n"] == 3 for row in task_effects))
        duplicated = records + [dict(records[0])]
        duplicate_pairs = paired_agent_comparisons(duplicated)
        self.assertEqual(duplicate_pairs[0]["complete_paired_trial_n"], 2)
        self.assertEqual(duplicate_pairs[0]["excluded_incomplete_paired_trial_n"], 1)

    def test_scorer_sensitivity_keeps_official_and_corrected_semantics_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp) / "run"
            snapshots = run / "step_snapshots"
            snapshots.mkdir(parents=True)
            payload = {
                "best_val_metric": 0.8,
                "val_steps": [
                    {"step_id": 1, "primary_metric": 0.8, "val_result": {"success": True}},
                    {"step_id": 2, "primary_metric": None, "val_result": {"success": False}},
                    {"step_id": 3, "primary_metric": 0.8, "val_result": {"success": True}},
                ],
            }
            summary_path = run / "summary.json"
            summary_path.write_text(json.dumps(payload), encoding="utf-8")
            for step in (1, 2, 3):
                (snapshots / f"step_{step:04d}_code.json").write_text("{}", encoding="utf-8")
            rows = collect_scorer_semantics_sensitivity(
                [{"phase": "confirmatory_lite", "trial": 1, "agent": "a", "task": "t", "summary_path": str(summary_path), "summary_sha256": "hash"}]
            )
            self.assertEqual(rows[0]["official_last_matching_best_step"], 3)
            self.assertEqual(rows[0]["sensitivity_first_matching_best_step"], 1)
            self.assertEqual(rows[0]["invalid_persisted_snapshot_n"], 1)
            self.assertEqual(rows[0]["valid_only_exploration_status"], "RECOMPUTE_GRAPHCODEBERT_SENSITIVITY")
            self.assertFalse(rows[0]["official_metrics_replaced"])

    def test_statistical_protocol_blocks_pseudoreplication_and_unadjusted_claims(self):
        protocol = build_statistical_analysis_protocol()
        self.assertIn("matched seed trial", protocol["estimand"]["primary_uncertainty_unit"])
        self.assertIn("Holm", protocol["primary_analysis"]["multiplicity"])
        self.assertFalse(protocol["sensitivity_analyses"][0]["replacement_allowed"])

    def test_adaptive_opportunity_interaction_uses_new_outcomes_in_frozen_strata(self):
        tasks = [
            "Continual_Learning_pycil", "Data_Efficiency_usb",
            "Generalization_domainbed", "Generalization_domainbed_officehome",
            "Robustness_openood", "Privacy_opacus", "Privacy_privacymeter",
            "Robustness_and_Reliability_art",
        ]
        agents = ["theaiscientist", "ai_scientist_v2", "aide", "aira_mcts", "autoresearch", "openevolve", "adaptivesearch"]
        dense = {"Privacy_opacus", "Privacy_privacymeter", "Robustness_and_Reliability_art"}
        records = []
        for agent_index, agent in enumerate(agents):
            for trial in range(1, 4):
                for task in tasks:
                    baseline_value = 0.01 * agent_index + 0.001 * trial
                    if agent == "adaptivesearch":
                        baseline_value += 0.20 if task in dense else 0.05
                    records.append({"phase": "confirmatory_lite", "model": "m", "provider": "p", "trial": trial, "task": task, "agent": agent, "normalized_improvement": baseline_value})
        rows = adaptive_opportunity_interactions(records)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(row["dense_task_n"] == 3 and row["sparse_task_n"] == 5 for row in rows))
        self.assertTrue(all(row["complete_block_gate_passed"] for row in rows))
        self.assertTrue(all(row["holm_family_size"] == 6 for row in rows))
        self.assertTrue(all(abs(row["mean_dense_minus_sparse_adaptive_advantage"] - 0.15) < 1e-12 for row in rows))


class PublishedPriorTests(unittest.TestCase):
    def test_complete_published_tables_are_transcribed(self):
        catalog = build_catalog(ROOT)
        self.assertEqual(len(agent_task_rows()), 18 * 6)
        self.assertEqual(len(agent_summary_rows()), 7)
        self.assertEqual(len(process_rows()), 12)
        self.assertEqual(len(task_card_rows()), 18)
        self.assertEqual(len(derived_agent_dispersion_rows()), 6)
        self.assertEqual(len(derived_task_discrimination_rows()), 18)
        adaptive = next(row for row in agent_summary_rows() if row["agent_id"] == "adaptivesearch")
        self.assertEqual(adaptive["mean_normalized_test_improvement"], 0.208)
        self.assertEqual(adaptive["pairwise_win_rate_percent"], 58.6)
        auc = next(row for row in process_rows() if row["metric"] == "AUC-over-steps")
        self.assertEqual(auc["pooled_spearman_rho"], 0.784)
        self.assertTrue(auc["significant_unadjusted_p_lt_0_05"])
        autoresearch = next(row for row in derived_agent_dispersion_rows() if row["agent_id"] == "autoresearch")
        self.assertAlmostEqual(autoresearch["sample_sd_across_task_means"], 0.243, places=3)
        most_discriminating = max(derived_task_discrimination_rows(), key=lambda row: row["range_across_agents"])
        self.assertEqual(most_discriminating["paper_task"], "PrivacyMeter")
        self.assertEqual({row["task_id"] for row in task_card_rows()}, {task["task_id"] for task in catalog["tasks"]})

    def test_published_prior_is_labeled_and_separated_from_new_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = write_published_prior(root)
            self.assertEqual(manifest["status"], "published_prior_only")
            self.assertFalse(manifest["separation_contract"]["merge_with_new_experiment_rows"])
            self.assertEqual(manifest["row_counts"]["agent_task"], 108)
            self.assertTrue((root / "figures" / "published_agent_task_heatmap.svg").is_file())
            self.assertTrue((root / "figures" / "derived_agent_mean_vs_cross_task_sd.svg").is_file())
            self.assertTrue((root / "figures" / "derived_task_strategy_discrimination.svg").is_file())
            self.assertTrue((root / "published_evaluation_contract.json").is_file())
            self.assertTrue((root / "paper_version_lineage.csv").is_file())
            synthesis = (root / "paper_ready_prior_evidence_brief.md").read_text(encoding="utf-8")
            self.assertIn("Claims that remain locked", synthesis)
            self.assertIn("Do not call these differences statistically significant", synthesis)
            self.assertTrue((root / "provenance_manifest.json").is_file())

    def test_post_hoc_guidance_changes_search_mode_but_cannot_unlock_claims(self):
        dense = published_guidance_for_task("Unlearning_open_unlearning")
        sparse = published_guidance_for_task("Data_Efficiency_easyfsl")
        self.assertEqual(dense["strategy_hypothesis"]["initial_mode"], "greedy_exploitation")
        self.assertEqual(sparse["strategy_hypothesis"]["initial_mode"], "multi_branch_exploration")
        self.assertFalse(dense["claim_unlock_eligible"])
        self.assertIn("autoresearch", sparse["strategy_hypothesis"]["required_counterfactual_operators"])


class PaperEvaluationTests(unittest.TestCase):
    def test_paper_quality_is_separate_from_fml_and_hard_gated(self):
        protocol = build_paper_evaluation_protocol()
        self.assertEqual(protocol["benchmark_status"], "local_extension_not_part_of_fml_bench")
        self.assertTrue(protocol["separation_from_fml"]["scores_must_not_be_averaged_together"])
        self.assertEqual(len(protocol["hard_gates"]), 6)
        self.assertEqual(len(protocol["reviewer_roles"]), 5)
        self.assertEqual(sum(row["weight"] for row in protocol["rubric"]), 1.0)
        self.assertEqual(protocol["matched_writing_ablation"]["total_planned_reviews"], 45)

    def test_paper_evaluation_writes_empty_score_templates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            status = write_paper_evaluation_protocol(root)
            self.assertEqual(status["status"], "TEMPLATE_NO_PAPER_SCORES")
            self.assertFalse(status["fml_metric_merge_allowed"])
            self.assertEqual(status["planned_review_row_n"], 3 * 3 * 5 * 7)
            self.assertEqual(status["planned_hard_gate_row_n"], 3 * 3 * 6)
            self.assertTrue((root / "paper_evaluation_protocol.json").is_file())
            self.assertTrue((root / "paper_review_score_template.csv").is_file())
            self.assertTrue((root / "issue_disposition_template.csv").is_file())
            self.assertTrue((root / "paper_release_decision_template.json").is_file())

    def test_completed_blinded_reviews_reduce_to_matched_writing_arm_effects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            template = root / "template"
            data = root / "data"
            report = root / "report"
            data.mkdir()
            write_paper_evaluation_protocol(template)
            with (template / "paper_review_score_template.csv").open(encoding="utf-8") as handle:
                score_rows = list(csv.DictReader(handle))
                score_fields = list(score_rows[0])
            arm_scores = {"W0_one_shot": 3, "W1_gated_no_peer_revision": 4, "W2_full_pipeline": 5}
            arm_index = {arm: index for index, arm in enumerate(arm_scores)}
            reviewer_roles = ["editor_in_chief", "methodology", "statistics", "reproducibility", "devils_advocate"]
            reports_dir = data / "reports"
            reports_dir.mkdir()
            for row in score_rows:
                row["manuscript_blind_id"] = f"blind-{arm_index[row['arm_private']]}-{row['draft']}"
                order = arm_index[row["arm_private"]] * 15 + (int(row["draft"]) - 1) * 5 + reviewer_roles.index(row["reviewer_role"]) + 1
                report_path = reports_dir / f"review-{order:02d}.md"
                if not report_path.exists():
                    report_path.write_text(f"synthetic independent review {order}\n", encoding="utf-8")
                row["reviewer_run_id"] = f"run-{order:02d}"
                row["reviewer_seed"] = str(9000 + order)
                row["review_order"] = str(order)
                row["reviewer_model"] = "fixed-review-model"
                row["report_path"] = str(report_path)
                row["report_sha256"] = hashlib.sha256(report_path.read_bytes()).hexdigest()
                row["score_1_to_5"] = str(arm_scores[row["arm_private"]])
                row["major_issue"] = "false"
                row["review_complete"] = "true"
            with (data / "review_scores.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=score_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(score_rows)
            with (template / "hard_gate_template.csv").open(encoding="utf-8") as handle:
                hard_rows = list(csv.DictReader(handle))
                hard_fields = list(hard_rows[0])
            for row in hard_rows:
                row["manuscript_blind_id"] = f"blind-{arm_index[row['arm_private']]}-{row['draft']}"
                row["pass"] = "true"
            with (data / "hard_gate_results.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=hard_fields, lineterminator="\n")
                writer.writeheader()
                writer.writerows(hard_rows)
            (data / "review_reports.json").write_text('{"reports": "synthetic test fixture"}', encoding="utf-8")
            (data / "claim_evidence_map.json").write_text('{"claims": "synthetic test fixture"}', encoding="utf-8")
            (data / "reproducibility_manifest.json").write_text('{"manifest": "synthetic test fixture"}', encoding="utf-8")
            (data / "issue_dispositions.csv").write_text("manuscript_blind_id,issue_id,severity,status,owner,evidence_path,rationale\n", encoding="utf-8")
            summary = evaluate_paper_evaluation(data, report)
            self.assertEqual(summary["status"], "COMPLETE_EVALUATION")
            self.assertEqual(summary["evaluated_manuscript_n"], 9)
            self.assertEqual(summary["paper_threshold_pass_n"], 6)
            self.assertAlmostEqual(summary["primary_contrast"]["mean_difference_left_minus_right"], 2.0)
            self.assertEqual(summary["primary_contrast"]["matched_draft_n"], 3)
            self.assertTrue((report / "manuscript_criterion_statistics.csv").is_file())
            self.assertTrue((report / "writing_arm_paired_comparisons.csv").is_file())


class PaperPackageTests(unittest.TestCase):
    def test_methods_are_substantive_but_empirical_claims_stay_locked(self):
        catalog = build_catalog(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_catalog_artifacts(catalog, root / "catalog")
            write_experiment_protocol(catalog, root / "protocol")
            write_experiment_artifacts(root / "missing-results", root / "experiments", catalog)
            write_paper_evaluation_protocol(root / "paper_evaluation")
            (root / "knowledge_base").mkdir()
            (root / "knowledge_base" / "metric_implementation_audit.json").write_text("{}", encoding="utf-8")
            (root / "published_prior").mkdir()
            (root / "published_prior" / "provenance_manifest.json").write_text("{}", encoding="utf-8")
            (root / "plans").mkdir()
            for task in catalog["tasks"]:
                (root / "plans" / f"{task['task_id']}.json").write_text("{}", encoding="utf-8")
            readiness = write_paper_package(
                catalog,
                root / "paper",
                catalog_dir=root / "catalog",
                knowledge_base_dir=root / "knowledge_base",
                plans_dir=root / "plans",
                published_prior_dir=root / "published_prior",
                protocol_dir=root / "protocol",
                experiments_dir=root / "experiments",
                paper_evaluation_dir=root / "paper_evaluation",
            )
            self.assertEqual(readiness["stage"], "METHODS_ONLY_EMPIRICAL_SECTIONS_LOCKED")
            self.assertFalse(readiness["ready_for_numerical_results"])
            config = json.loads((root / "paper" / "paper_configuration.json").read_text(encoding="utf-8"))
            self.assertEqual(config["confirmation_status"], "AWAITING_HUMAN_CONFIRMATION")
            manuscript = (root / "paper" / "evidence_locked_manuscript.md").read_text(encoding="utf-8")
            self.assertIn("## 4. Experimental protocol", manuscript)
            self.assertIn("[LOCKED", manuscript)
            with (root / "paper" / "claim_evidence_registry.csv").open(encoding="utf-8") as handle:
                claims = list(csv.DictReader(handle))
            self.assertEqual(len(claims), 10)
            self.assertEqual(next(row for row in claims if row["claim_id"] == "K1")["status"], "READY_STATIC")
            self.assertTrue(next(row for row in claims if row["claim_id"] == "E1")["status"].startswith("LOCKED"))
            manifest = json.loads((root / "paper" / "paper_package_manifest.json").read_text(encoding="utf-8"))
            self.assertFalse(manifest["new_empirical_results_present"])
            self.assertFalse(manifest["published_prior_merged_into_new_results"])


class KnowledgeBaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_all_agent_dossiers_resolve_audited_control_flow(self):
        dossiers = build_agent_dossiers(ROOT, self.catalog)
        self.assertEqual(len(dossiers), 7)
        self.assertEqual({row["agent_id"] for row in dossiers}, {row["agent_id"] for row in self.catalog["agents"]})
        for dossier in dossiers:
            self.assertTrue(dossier["control_flow_symbols"], dossier["agent_id"])
            self.assertTrue(all(symbol["line_start"] > 0 for symbol in dossier["control_flow_symbols"]))
            self.assertTrue(all(len(module["sha256"]) == 64 for module in dossier["source_modules"]))
            self.assertTrue(dossier["known_failure_modes_to_test"])

    def test_all_task_dossiers_capture_metrics_baselines_and_boundaries(self):
        dossiers = build_task_dossiers(ROOT, self.catalog)
        self.assertEqual(len(dossiers), 18)
        for dossier in dossiers:
            metric = dossier["metric_contract"]
            self.assertIsNotNone(metric["baseline_validation_raw"], dossier["task_id"])
            self.assertIsNotNone(metric["baseline_test_raw"], dossier["task_id"])
            self.assertTrue(metric["all_declared_metric_directions"], dossier["task_id"])
            self.assertTrue(dossier["execution"]["editable_target_files"])
            self.assertEqual(len(dossier["published_agent_results"]), 6)
        unlearning = next(row for row in dossiers if row["task_id"] == "Unlearning_open_unlearning")["metric_contract"]
        self.assertEqual(unlearning["raw_native_direction"], "higher")
        self.assertEqual(unlearning["fml_display_direction"], "lower")
        self.assertEqual(unlearning["fml_display_transform"], "-log10(raw forget_quality)")
        self.assertAlmostEqual(unlearning["baseline_test_display"], 166.8, places=1)
        self.assertAlmostEqual(unlearning["normalization"]["worst"], 166.8, places=1)

    def test_knowledge_base_reports_pending_real_evidence_honestly(self):
        with tempfile.TemporaryDirectory() as tmp:
            status = write_knowledge_base(ROOT, self.catalog, Path(tmp))
            self.assertEqual(status["agent_dossier_n"], 7)
            self.assertEqual(status["task_dossier_n"], 18)
            self.assertTrue(status["all_agent_control_symbols_resolved"])
            self.assertTrue(status["all_task_baselines_present"])
            self.assertEqual(status["real_new_experiment_record_n"], 0)
            audit = json.loads((Path(tmp) / "objective_completion_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["overall_status"], "PARTIAL_REAL_EXPERIMENTS_AND_MANUSCRIPT_EVALUATION_PENDING")
            self.assertEqual([row["status"] for row in audit["requirements"]].count("NOT_YET_PROVEN"), 2)
            experiment_row = next(row for row in audit["requirements"] if row["requirement"] == "run sound new experiments")
            self.assertNotIn("preflight is not ready", experiment_row["proof"])
            self.assertIn("reported separately", experiment_row["proof"])

    def test_metric_audit_exposes_paper_scorer_semantic_differences(self):
        audit = build_metric_implementation_audit(ROOT, self.catalog)
        findings = {row["finding_id"]: row for row in audit["semantic_findings"]}
        self.assertEqual(findings["EXPLORATION_VALID_STEP_SCOPE"]["status"], "PAPER_SCORER_SEMANTIC_DIFFERENCE")
        self.assertEqual(findings["BEST_IMPROVEMENT_FIRST_VS_LAST"]["status"], "PAPER_SCORER_SEMANTIC_DIFFERENCE")
        self.assertIn("Do not change metric semantics", audit["study_rule"])


class ExperimentDesignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def test_protocol_is_balanced_and_separates_trials(self):
        protocol, rows = build_experiment_protocol(self.catalog, model="fixed-model")
        self.assertEqual(protocol["phase_run_counts"]["pilot"], 14)
        self.assertEqual(protocol["phase_run_counts"]["confirmatory_lite"], 168)
        self.assertEqual(protocol["total_planned_runs"], 182)
        self.assertEqual({row["max_steps"] for row in rows if row["phase"] == "pilot"}, {1})
        self.assertEqual({row["max_steps"] for row in rows if row["phase"] == "confirmatory_lite"}, {3})
        self.assertIn("cannot activate", protocol["resource_budget_amendment"]["adaptive_search_limitation"])
        confirmatory = [row for row in rows if row["phase"] == "confirmatory_lite"]
        self.assertEqual(len({(row["agent"], row["task"], row["trial"]) for row in confirmatory}), 168)
        self.assertTrue(all("--seed" in row["command"] for row in rows))
        self.assertTrue(all(f"trial_{row['trial']:02d}" in row["result_root"] for row in rows))
        pilot_tasks = {row["task"] for row in rows if row["phase"] == "pilot"}
        self.assertEqual(pilot_tasks, {"Privacy_privacymeter", "Generalization_domainbed"})
        self.assertTrue(all(not row["paper_claim_eligible"] for row in rows if row["phase"] == "pilot"))
        self.assertTrue(all(row["paper_claim_eligible"] for row in confirmatory))
        self.assertEqual([row["execution_order"] for row in rows], list(range(1, 183)))
        self.assertEqual({row["task"] for row in rows[:7]}, {"Generalization_domainbed"})
        self.assertEqual(protocol["analysis_policy"]["execution_order"]["order_seed"], 20260727)
        self.assertIn("anti_cherry_pick_rule", protocol["pilot_selection_basis"])
        self.assertIn("matched seed block", protocol["analysis_policy"]["primary_uncertainty_unit"])
        self.assertIn("Holm", protocol["analysis_policy"]["multiplicity"])
        with self.assertRaises(ValueError):
            build_experiment_protocol(self.catalog, model="fixed-model", pilot_steps=0)

    def test_codex_cli_ssh_condition_is_explicit_in_every_run(self):
        protocol, rows = build_experiment_protocol(
            self.catalog,
            model="gpt-5.6-sol",
            provider="CodexCLI",
            eval_backend="ssh",
            ssh_host="ubuntu-heshi",
            remote_project_root="/media/heshi/game/fml-scientist/repo",
        )
        self.assertEqual(protocol["factors"]["provider"], "CodexCLI")
        self.assertEqual(protocol["factors"]["execution_backend"], "ssh")
        self.assertEqual(protocol["factors"]["adaptive_search_embedding"]["device"], "cpu")
        self.assertTrue(protocol["factors"]["adaptive_search_embedding"]["local_files_only"])
        self.assertIn("not an exact reproduction", protocol["execution_platform_contract"]["controller_model_condition"])
        self.assertTrue(all("--provider CodexCLI" in row["command"] for row in rows))
        self.assertTrue(all("--eval-backend ssh" in row["command"] for row in rows))
        self.assertTrue(all("--remote-project-root /media/heshi/game/fml-scientist/repo" in row["command"] for row in rows))

    def test_preflight_reports_key_presence_without_key_value(self):
        report = preflight_environment(self.catalog, provider="OpenAI", model="SET_MODEL", repo=ROOT)
        self.assertIn("required_key_present", report)
        self.assertNotIn("key_value", report)
        self.assertEqual(report["lite_workspace_task_total"], 8)
        self.assertEqual(report["lite_environment_task_total"], 8)
        self.assertIn("ready_for_full_extension", report)
        self.assertIn("adaptivesearch_controller", report)
        self.assertIn("weight_sha256", report["adaptivesearch_controller"])
        self.assertFalse(report["ready"])

    def test_llm_calls_use_the_frozen_trial_seed(self):
        source = (ROOT / "agents" / "llm.py").read_text(encoding="utf-8")
        self.assertNotIn("seed=0", source)
        self.assertIn("seed=experiment_seed()", source)

    def test_campaign_is_resumable_and_refuses_placeholder_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_experiment_protocol(self.catalog, root / "template")
            self.assertTrue((root / "template" / "statistical_analysis_protocol.json").is_file())
            self.assertTrue((root / "template" / "statistical_analysis_protocol.md").is_file())
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

    def test_campaign_reuses_the_active_controller_python(self):
        argv = _command_argv("python run_agent_benchmark.py --model fixed")
        self.assertEqual(argv[0], sys.executable)
        self.assertEqual(argv[1:], ["run_agent_benchmark.py", "--model", "fixed"])


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
