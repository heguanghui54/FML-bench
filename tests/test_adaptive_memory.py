from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.adaptive_controller import (
    assess_negative_memory_candidate,
    build_review_amendment,
    filter_visible_evidence,
    retrieve_negative_memory,
    select_policy_composition,
    select_strategy,
)
from ml_scientist.adaptive_runtime import AdaptiveResearchRuntime
from ml_scientist.architecture_report import write_architecture_checkpoint_report
from ml_scientist.benchmark_learning import build_fml_benchmark_contract, learn_manifest, write_benchmark_contract
from ml_scientist.catalog import build_catalog
from ml_scientist.governance import EvidenceGatedSkillRegistry, initialize_governance_artifacts
from ml_scientist.knowledge_base import write_knowledge_base
from ml_scientist.knowledge_graph import (
    append_episode,
    build_project_knowledge_graph,
    query_knowledge_graph,
    validate_knowledge_graph,
    write_knowledge_graph,
)
from ml_scientist.paper_evaluation import write_paper_evaluation_protocol
from ml_scientist.planner import build_research_paper_plan, validate_plan
from ml_scientist.reporting import write_catalog_artifacts


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)
        cls.contract = build_fml_benchmark_contract(cls.catalog)

    def test_fml_contract_separates_visible_and_protected_metric_views(self):
        self.assertTrue(self.contract["validation"]["passed"])
        task_metrics = [row for row in self.contract["metrics"] if row["category"] == "task_performance"]
        self.assertEqual(len(task_metrics), 36)
        for task in self.catalog["tasks"]:
            rows = [row for row in task_metrics if row["task_id"] == task["task_id"]]
            self.assertEqual({row["split"] for row in rows}, {"validation", "test"})
            protected = next(row for row in rows if row["split"] == "test")
            self.assertEqual(protected["observability"], "PROTECTED_FINAL_ONLY")
            self.assertTrue(protected["protected"])

    def test_val_test_gap_is_not_online_visible(self):
        metric = next(row for row in self.contract["metrics"] if row["name"] == "Val-test |gap|")
        self.assertEqual(metric["observability"], "PROTECTED_FINAL_ONLY")
        self.assertNotIn("search_trajectory", metric["stages"])

    def test_external_manifest_cannot_activate_without_audit_and_boundary_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "benchmark.json"
            source.write_text(
                json.dumps(
                    {
                        "benchmark_id": "new-benchmark",
                        "tasks": [],
                        "metrics": [
                            {
                                "metric_id": "m1", "name": "score", "category": "task_performance",
                                "direction": "higher", "observability": "ONLINE_VISIBLE", "stages": ["experiment"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            learned = learn_manifest(source)
            self.assertEqual(learned["maturity"], "observation")
            self.assertFalse(learned["activation_eligible"])
            with self.assertRaises(ValueError):
                learn_manifest(source, activate=True)


class KnowledgeGraphTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)

    def _artifact_root(self, root: Path) -> Path:
        artifact = root / "artifacts"
        write_catalog_artifacts(self.catalog, artifact / "catalog")
        initialize_governance_artifacts(self.catalog, artifact / "governance")
        write_knowledge_base(ROOT, self.catalog, artifact / "knowledge_base")
        write_paper_evaluation_protocol(artifact / "paper_evaluation")
        write_benchmark_contract(build_fml_benchmark_contract(self.catalog), artifact / "benchmark_contracts")
        return artifact

    def test_current_memory_files_migrate_into_typed_graph(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._artifact_root(root)
            graph, migration = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            report = validate_knowledge_graph(graph)
            self.assertTrue(report["passed"], report["errors"])
            self.assertGreater(report["node_type_counts"]["Metric"], 30)
            self.assertEqual(report["node_type_counts"]["Agent"], 7)
            self.assertEqual(report["node_type_counts"]["Policy"], 42)
            self.assertEqual(report["node_type_counts"]["Strategy"], 11)
            self.assertGreaterEqual(report["node_type_counts"]["Memory"], 1)
            self.assertGreater(migration["source_file_count"], 20)
            status = write_knowledge_graph(graph, migration, artifact / "knowledge_graph", governance_dir=artifact / "governance")
            self.assertTrue(status["passed"])
            legacy = json.loads((artifact / "governance" / "memory_registry.json").read_text(encoding="utf-8"))
            self.assertTrue(legacy["legacy_entries_migrated"])
            self.assertTrue(legacy["legacy_compatibility_view"])
            self.assertTrue(Path(legacy["canonical_store"]).is_file())

    def test_graph_queries_strategies_and_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = self._artifact_root(Path(tmp))
            graph_a, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            graph_b, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            self.assertEqual(graph_a["content_sha256"], graph_b["content_sha256"])
            rows = query_knowledge_graph(graph_a, "privacy diverse search", node_types={"Strategy"}, limit=10)
            self.assertTrue(rows)
            self.assertTrue(all(row["node_type"] == "Strategy" for row in rows))
            related = query_knowledge_graph(graph_a, "Privacy_privacymeter", node_types={"Strategy"}, limit=10)
            self.assertGreaterEqual(len(related), 7)
            policies = query_knowledge_graph(
                graph_a, "privacy diverse search", node_types={"Policy"}, stage="method_search", limit=100
            )
            self.assertTrue(policies)
            self.assertTrue(all(row["attributes"]["selectable_unit"] for row in policies))
            self.assertTrue(all("method_search" in row["attributes"]["eligible_stages"] for row in policies))
            visible_metrics = query_knowledge_graph(
                graph_a, "Privacy_privacymeter AUC_gap_mean", node_types={"Metric"}, stage="experiment", limit=20
            )
            self.assertTrue(visible_metrics)
            self.assertTrue(all(not row["attributes"].get("protected") for row in visible_metrics))

    def test_only_promoted_atomic_policy_skills_are_retrievable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._artifact_root(root)
            source = root / "source.json"
            source.write_text('{"rule":"multi proposal"}', encoding="utf-8")
            outcome = root / "outcome.json"
            outcome.write_text('{"passed":true}', encoding="utf-8")
            registry = EvidenceGatedSkillRegistry(artifact / "governance", "commit")
            registry.propose(
                skill_id="hs-policy-test", channel="research_execution", rule="Generate two distinct proposals",
                evidence_paths=[source], acceptance_gates=["proposal_diversity"],
                rollback_condition="diversity regression", policy_role="proposal",
                eligible_stages=["method_search"], requires_capabilities=["candidate_state"],
                provides_capabilities=["candidate_pool", "candidate_artifact"],
            )
            observation_graph, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            self.assertFalse(query_knowledge_graph(
                observation_graph, "hs policy test", node_types={"Skill"}, stage="method_search"
            ))
            registry.record_outcome(
                skill_id="hs-policy-test", evidence_path=outcome, context_id="task-a",
                real_downstream_complete=True, no_regression=True,
                metrics={
                    "spread": 1.0,
                    "policy_trace": {
                        "role": "proposal", "applied": True, "status": "PASSED",
                        "evidence_artifact_refs": [str(outcome)],
                    },
                },
                held_out_validation=True,
            )
            registry.promote("hs-policy-test")
            active_graph, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            rows = query_knowledge_graph(
                active_graph, "hs policy test", node_types={"Skill"}, stage="method_search"
            )
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["attributes"]["policy_role"], "proposal")
            self.assertTrue(rows[0]["attributes"]["active"])

    def test_episode_append_rejects_protected_routing(self):
        episode = {
            "episode_id": "e1", "run_id": "r1", "node_id": "n1",
            "selected_strategy_id": "strategy:aide", "visible_evidence": [],
            "outcome": "PASSED_GATE", "snapshot_sha256": "abc",
            "protected_metric_used_for_routing": True,
        }
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                append_episode(Path(tmp) / "episodes.jsonl", episode)

    def test_final_graph_replay_blocks_known_failed_label_smoothing(self):
        graph = json.loads((
            ROOT / "artifacts/ml_scientist/post_arm/knowledge_graph/"
            "final_graph_snapshot/knowledge_graph.json"
        ).read_text(encoding="utf-8"))
        memories = retrieve_negative_memory(graph, task_id="Privacy_privacymeter")
        self.assertTrue(memories)
        replay = assess_negative_memory_candidate(
            {"modification": "use CrossEntropyLoss label_smoothing=0.1"}, memories
        )
        self.assertFalse(replay["passed"])
        self.assertEqual(replay["route"], "REJECT_NEGATIVE_MEMORY_DUPLICATE")
        override = assess_negative_memory_candidate(
            {"modification": "use CrossEntropyLoss label_smoothing=0.1"},
            memories,
            counterfactual_distinction="new artifact proves a different activation target and control",
        )
        self.assertTrue(override["passed"])
        self.assertEqual(override["route"], "ALLOW_JUSTIFIED_OVERRIDE")

        distinct = assess_negative_memory_candidate(
            {
                "hypothesis_id": "non_dp_weight_decay_5e_4",
                "modification": "set optimizer weight_decay to 0.0005",
                "comparison": [
                    {"idea": "label smoothing", "selected": False},
                    {"idea": "mixup", "selected": False},
                ],
            },
            memories,
        )
        self.assertTrue(distinct["passed"])
        self.assertEqual(distinct["route"], "ALLOW")

    def test_completed_episode_becomes_episode_decision_and_metric_edges(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._artifact_root(root)
            episodes = root / "episodes.jsonl"
            append_episode(
                episodes,
                {
                    "episode_id": "e1", "run_id": "r1", "task_id": "Privacy_privacymeter",
                    "node_id": "code-modification", "selected_strategy_id": "strategy:aide",
                    "visible_evidence": [{"metric_id": "process::Valid step ratio", "observability": "ONLINE_VISIBLE", "value": 1.0}],
                    "outcome": "FAILED_GATE", "snapshot_sha256": "abc",
                },
            )
            graph, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact, episode_paths=[episodes])
            counts = validate_knowledge_graph(graph)["node_type_counts"]
            self.assertEqual(counts["Episode"], 1)
            self.assertEqual(counts["Decision"], 1)
            self.assertEqual(counts["Failure"], 1)
            self.assertTrue(any(edge["relation"] == "USES_METRIC" for edge in graph["edges"]))

    def test_explicit_policy_trace_updates_only_credited_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._artifact_root(root)
            episodes = root / "episodes.jsonl"
            append_episode(
                episodes,
                {
                    "episode_id": "policy-e1", "run_id": "r1", "task_id": "Privacy_privacymeter",
                    "node_id": "hypothesis-frontier", "node_type": "method_search",
                    "selected_policy_ids": ["policy:autoresearch:state", "policy:theaiscientist:proposal"],
                    "selected_policy_by_role": {
                        "state": "policy:autoresearch:state",
                        "proposal": "policy:theaiscientist:proposal",
                    },
                    "visible_evidence": [], "outcome": "PASSED_GATE", "snapshot_sha256": "abc",
                    "policy_evaluations": [
                        {
                            "policy_id": "policy:autoresearch:state", "role": "state", "applied": True,
                            "status": "PASSED", "evidence_artifact_refs": ["state.json"],
                        },
                        {
                            "policy_id": "policy:theaiscientist:proposal", "role": "proposal", "applied": False,
                            "status": "NOT_APPLIED", "evidence_artifact_refs": [],
                        },
                    ],
                },
            )
            graph, _ = build_project_knowledge_graph(ROOT, self.catalog, artifact, episode_paths=[episodes])
            nodes = {row["node_id"]: row for row in graph["nodes"]}
            self.assertEqual(nodes["policy:autoresearch:state"]["attributes"]["successful_contexts"], 1)
            self.assertEqual(
                nodes["policy:autoresearch:state"]["attributes"]["successful_contexts_by_stage"],
                {"method_search": 1},
            )
            self.assertNotIn("successful_contexts", nodes["policy:theaiscientist:proposal"]["attributes"])
            credited = [edge for edge in graph["edges"] if edge["relation"] == "CREDITED_TO"]
            self.assertEqual(len(credited), 2)

    def test_html_checkpoint_reports_graph_bound_plans_and_pause(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = self._artifact_root(root)
            graph, migration = build_project_knowledge_graph(ROOT, self.catalog, artifact)
            write_knowledge_graph(graph, migration, artifact / "knowledge_graph", governance_dir=artifact / "governance")
            task = self.catalog["tasks"][0]
            plan = build_research_paper_plan(task, self.catalog["agents"], knowledge_graph=graph)
            (artifact / "plans").mkdir()
            (artifact / "plans" / f"{task['task_id']}.json").write_text(json.dumps(plan), encoding="utf-8")
            status = write_architecture_checkpoint_report(
                graph_path=artifact / "knowledge_graph" / "knowledge_graph.json",
                benchmark_contract_path=artifact / "benchmark_contracts" / "fml-bench.json",
                plans_dir=artifact / "plans",
                out_path=artifact / "reports" / "checkpoint.html",
            )
            self.assertEqual(status["graph_bound_plan_count"], 1)
            report = (artifact / "reports" / "checkpoint.html").read_text(encoding="utf-8")
            self.assertIn("PAUSED_BY_USER", report)
            self.assertIn("Protected final metrics never return to search", report)


class AdaptivePlannerControllerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = build_catalog(ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifact = root / "artifacts"
            write_catalog_artifacts(cls.catalog, artifact / "catalog")
            initialize_governance_artifacts(cls.catalog, artifact / "governance")
            write_benchmark_contract(build_fml_benchmark_contract(cls.catalog), artifact / "benchmark_contracts")
            cls.graph, _ = build_project_knowledge_graph(ROOT, cls.catalog, artifact)

    def test_stage3_plan_is_graph_bound_and_node_local(self):
        task = next(row for row in self.catalog["tasks"] if row["task_id"] == "Privacy_privacymeter")
        plan = build_research_paper_plan(
            task,
            self.catalog["agents"],
            knowledge_graph=self.graph,
            research_request="privacy attack evaluation with limited compute",
            budget={"candidate_steps": 5, "token_budget": 100000, "wall_clock_hours": 6},
        )
        validate_plan(plan)
        self.assertEqual(plan["plan_mode"], "task_conditioned_atomic_policy_composition")
        self.assertEqual(plan["knowledge_graph_snapshot"]["content_sha256"], self.graph["content_sha256"])
        self.assertFalse(plan["controller_contract"]["cartesian_full_pipeline_expansion"])
        nodes = {row["node_id"]: row for row in plan["nodes"]}
        code_metadata = nodes["code-modification"]["metadata"]
        self.assertEqual(code_metadata["selection_unit"], "atomic_policy_composition")
        self.assertEqual(
            set(code_metadata["policy_candidates_by_role"]),
            {"state", "proposal", "selection", "debug", "memory", "diversity"},
        )
        self.assertTrue(all(code_metadata["policy_candidates_by_role"].values()))
        self.assertNotIn("strategy_candidates", code_metadata)
        self.assertEqual(
            nodes["baseline-reproduction"]["metadata"]["strategy_candidates"],
            ["strategy:deterministic_contract_validation"],
        )
        self.assertIn("proposal_reachable_call_path", nodes["code-modification"]["inner_loop"]["hard_gates"])
        self.assertIn("metric:process::Exploration Spread", nodes["hypothesis-frontier"]["metadata"]["metric_bindings"])
        self.assertFalse(nodes["protected-final-test"]["metadata"]["controller_routing_allowed"])
        self.assertIn("PROTECTED_FINAL_ONLY", nodes["protected-final-test"]["metadata"]["allowed_metric_observability"])

    def test_controller_filters_protected_evidence_and_records_choice(self):
        evidence = [
            {"metric_name": "Exploration Spread", "observability": "ONLINE_VISIBLE", "status": "low", "value": 0.1},
            {"metric_name": "test score", "observability": "PROTECTED_FINAL_ONLY", "value": 0.99},
        ]
        self.assertEqual(len(filter_visible_evidence(evidence, stage_complete=False)), 1)
        decision = select_strategy(
            run_id="r1", node_id="code-modification", stage=4, snapshot_sha256="sha",
            candidates=[
                {"strategy_id": "strategy:a", "maturity": "verified", "retrieval_score": 2, "expected_cost": 0.2, "diversity_contribution": 1.0},
                {"strategy_id": "strategy:b", "maturity": "observation", "retrieval_score": 2, "expected_cost": 0.2, "diversity_contribution": 0.0},
            ],
            evidence=evidence,
            budget_before={"remaining_fraction": 0.8, "tokens": 1000},
        )
        self.assertEqual(decision["selected_strategy_id"], "strategy:a")
        self.assertFalse(decision["protected_metric_visible"])
        self.assertEqual(len(decision["visible_evidence"]), 1)

    def test_controller_can_compose_atomic_policies_from_different_sources(self):
        graph_nodes = {row["node_id"]: row for row in self.graph["nodes"]}
        state = graph_nodes["policy:autoresearch:state"]
        proposal = graph_nodes["policy:theaiscientist:proposal"]
        decision = select_policy_composition(
            run_id="r1", node_id="hypothesis-frontier", node_type="method_search", stage=4,
            snapshot_sha256="sha",
            candidates_by_role={
                "state": [{
                    "policy_id": state["node_id"], "maturity": "observation", "retrieval_score": 0,
                    "expected_cost": 0.01, "attributes": state["attributes"],
                }],
                "proposal": [{
                    "policy_id": proposal["node_id"], "maturity": "observation", "retrieval_score": 1,
                    "expected_cost": 0.01, "attributes": proposal["attributes"],
                }],
            },
            required_roles=["state", "proposal"], evidence=[],
            budget_before={"remaining_fraction": 1.0},
        )
        self.assertEqual(decision["selected_policy_by_role"]["state"], "policy:autoresearch:state")
        self.assertEqual(decision["selected_policy_by_role"]["proposal"], "policy:theaiscientist:proposal")
        self.assertTrue(decision["compatibility"]["passed"])

    def test_paper_review_can_route_to_code_and_require_new_hidden_evaluation(self):
        amendment = build_review_amendment(
            issue_id="review-1", issue_type="implementation", source_paper_node="paper-peer-review",
            protected_test_exposed=True,
        )
        self.assertEqual(amendment["target_node"], "code-modification")
        self.assertTrue(amendment["invalidates_descendants"])
        self.assertTrue(amendment["requires_new_hidden_evaluation"])

    def test_persisted_runtime_schedules_without_launching_experiments(self):
        task = self.catalog["tasks"][0]
        plan = build_research_paper_plan(task, self.catalog["agents"], knowledge_graph=self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AdaptiveResearchRuntime(
                plan=plan, graph=self.graph, run_id="runtime-test",
                state_path=Path(tmp) / "state.json",
            )
            self.assertFalse(runtime.state["experiment_execution_enabled"])
            self.assertEqual(runtime.ready_nodes(), ["memory-snapshot"])
            decision = runtime.choose_strategy(
                "memory-snapshot", evidence=[], budget_before={"remaining_fraction": 1.0}
            )
            self.assertEqual(decision["selected_strategy_id"], "strategy:deterministic_contract_validation")
            self.assertFalse(decision["operator_contract"]["protected_evidence_included"])
            self.assertTrue(decision["operator_contract"]["execution_requires_explicit_executor_call"])
            self.assertIn("Required output", decision["operator_contract"]["instruction"])
            runtime.record_outcome(
                "memory-snapshot", outcome="PASSED_GATE", evidence_artifacts=[{"path": "snapshot.json"}],
                budget_remaining=True,
            )
            self.assertIn("research-contract", runtime.ready_nodes())
            restored = AdaptiveResearchRuntime(
                plan=plan, graph=self.graph, run_id="runtime-test",
                state_path=Path(tmp) / "state.json",
            )
            self.assertEqual(restored.state["node_status"]["memory-snapshot"], "COMPLETE")

    def test_runtime_rejects_snapshot_drift_and_protected_routing(self):
        task = self.catalog["tasks"][0]
        plan = build_research_paper_plan(task, self.catalog["agents"], knowledge_graph=self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AdaptiveResearchRuntime(
                plan=plan, graph=self.graph, run_id="runtime-test",
                state_path=Path(tmp) / "state.json",
            )
            runtime.state["node_status"]["protected-final-test"] = "READY"
            with self.assertRaises(ValueError):
                runtime.choose_strategy(
                    "protected-final-test", evidence=[], budget_before={"remaining_fraction": 1.0}
                )
            drifted = dict(self.graph)
            drifted["content_sha256"] = "different"
            with self.assertRaises(ValueError):
                AdaptiveResearchRuntime(
                    plan=plan, graph=drifted, run_id="runtime-test-2",
                    state_path=Path(tmp) / "other.json",
                )

    def test_atomic_runtime_requires_role_level_policy_trace(self):
        task = self.catalog["tasks"][0]
        plan = build_research_paper_plan(task, self.catalog["agents"], knowledge_graph=self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            runtime = AdaptiveResearchRuntime(
                plan=plan, graph=self.graph, run_id="policy-runtime-test",
                state_path=Path(tmp) / "state.json",
            )
            runtime.state["node_status"]["hypothesis-frontier"] = "READY"
            decision = runtime.choose_strategy(
                "hypothesis-frontier",
                evidence=[{"metric_name": "Exploration Spread", "observability": "ONLINE_VISIBLE", "value": 0.2}],
                budget_before={"remaining_fraction": 1.0, "tokens_remaining": 1000},
            )
            self.assertEqual(decision["decision_kind"], "atomic_policy_composition")
            self.assertTrue(decision["operator_contract"]["policy_trace_required"])
            self.assertNotIn("selected_strategy_id", decision)
            with self.assertRaises(ValueError):
                runtime.record_outcome(
                    "hypothesis-frontier", outcome="PASSED_GATE",
                    evidence_artifacts=[{"path": "hypotheses.json"}], budget_remaining=True,
                )
            evaluations = [
                {
                    "policy_id": policy_id, "role": role, "applied": True, "status": "PASSED",
                    "evidence_artifact_refs": [f"trace/{role}.json"], "gate_checks": ["role_gate_passed"],
                }
                for role, policy_id in decision["selected_policy_by_role"].items()
            ]
            evaluations[0]["learning_signal"] = {
                "intent": "CREATE",
                "branch": "task_specific",
                "reusable_pattern": "Switch representation after measured low diversity",
                "trigger": {"exploration_spread_lt": 0.3},
                "procedure": ["Freeze the current candidate set", "Generate a mechanism-distinct representation"],
                "eligible_stages": ["method_search"],
                "expected_effect": {"Exploration Spread": "increase"},
                "acceptance_gates": ["held_out_diversity_improves"],
                "rollback_condition": "held-out task regresses",
            }
            runtime.record_outcome(
                "hypothesis-frontier", outcome="PASSED_GATE",
                evidence_artifacts=[{"path": "hypotheses.json"}], budget_remaining=True,
                policy_evaluations=evaluations,
                budget_after={"remaining_fraction": 0.9, "tokens_remaining": 700},
                telemetry={"wall_clock_seconds": 12.5},
                state_observations={
                    "metrics": [{"metric_name": "Exploration Spread", "value": 0.5}],
                    "hypothesis_count": 4,
                },
            )
            episode = runtime.export_episode("hypothesis-frontier")
            self.assertEqual(set(episode["selected_policy_ids"]), set(decision["selected_policy_ids"]))
            self.assertEqual(len(episode["policy_evaluations"]), len(decision["selected_policy_ids"]))
            self.assertEqual(episode["schema_version"], "ml-scientist-adaptive-episode-v2")
            self.assertEqual(episode["research_context"]["task_id"], task["task_id"])
            self.assertEqual(episode["next_state"]["route"]["action"], "ADVANCE")
            self.assertEqual(episode["resource_telemetry"]["tokens_consumed"], 300)
            self.assertAlmostEqual(episode["resource_telemetry"]["diversity_delta"], 0.3)
            self.assertEqual(episode["learning_signals"][0]["intent"], "CREATE")

    def test_stage7_runtime_invokes_skill_lifecycle_without_experiment(self):
        task = self.catalog["tasks"][0]
        plan = build_research_paper_plan(task, self.catalog["agents"], knowledge_graph=self.graph)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            governance = root / "governance"
            initialize_governance_artifacts(self.catalog, governance)
            episodes = root / "episodes.jsonl"
            episodes.write_text(json.dumps({
                "schema_version": "ml-scientist-adaptive-episode-v2",
                "episode_id": "prior:contract:1",
                "run_id": "prior",
                "task_id": task["task_id"],
                "node_id": "research-contract",
                "node_type": "contract",
                "decision_id": "decision:prior",
                "selected_strategy_id": "strategy:deterministic_contract_validation",
                "selected_policy_ids": [],
                "visible_evidence": [],
                "research_context": {"task_id": task["task_id"], "research_request": ""},
                "state_before": {"node_status": "READY"},
                "action": {"decision_id": "decision:prior"},
                "outcome": "PASSED_GATE",
                "next_state": {"node_status": "COMPLETE", "route": {"action": "ADVANCE"}},
                "resource_telemetry": {},
                "evidence_artifacts": [{"path": "contract.json"}],
                "policy_evaluations": [],
                "snapshot_sha256": self.graph["content_sha256"],
                "protected_metric_used_for_routing": False,
            }) + "\n", encoding="utf-8")
            runtime = AdaptiveResearchRuntime(
                plan=plan, graph=self.graph, run_id="stage7-runtime-test",
                state_path=root / "state.json",
            )
            runtime.state["node_status"]["skill-evolution"] = "READY"
            result = runtime.run_skill_evolution(
                episodes_path=episodes,
                governance_root=governance,
                out_dir=root / "skill-evolution",
            )
            self.assertFalse(result["experiment_execution_started"])
            self.assertEqual(result["outcome_record"]["outcome"], "PASSED_GATE")
            self.assertEqual(runtime.state["node_status"]["skill-evolution"], "COMPLETE")
            self.assertTrue((root / "skill-evolution" / "skill_governance_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
