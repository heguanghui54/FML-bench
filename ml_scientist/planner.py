"""Stage-3 planner for a unified experiment-to-paper dependency graph."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

from .contracts import InnerLoopContract, NodeStatus, PipelinePlan, ResearchNode
from .knowledge_graph import query_knowledge_graph
from .policy_model import policy_id, policy_roles_for_node
from .published_prior import published_guidance_for_task


STAGES = [
    {"id": 0, "name": "context_memory_snapshot", "role": "Freeze retrievable memory and skill versions."},
    {"id": 1, "name": "goal_evidence_contract", "role": "Freeze research, evaluation, budget, and paper-claim contracts."},
    {"id": 2, "name": "method_skill_search", "role": "Select ML methods, baseline search operators, reviewers, and skills."},
    {"id": 3, "name": "unified_task_graph_planning", "role": "Plan all experiment and paper nodes plus nested OR trees."},
    {"id": 4, "name": "execute_ready_candidates", "role": "Execute dependency-ready code, experiment, analysis, or writing candidates."},
    {"id": 5, "name": "type_specific_review", "role": "Apply frozen node-specific metrics, reviewers, and integrity gates."},
    {"id": 6, "name": "select_repair_iterate", "role": "Advance, refine, replicate, debug, backtrack, or amend the graph."},
    {"id": 7, "name": "read_write_assess_govern", "role": "Gate completed trajectories, distill skills, assess contextual utility, and govern future snapshots."},
]


COMMON_OUTCOMES = (
    "INVALID_EXECUTION",
    "VALID_REGRESSION",
    "STOCHASTIC_UNCERTAIN",
    "VALID_NONPROMOTABLE",
    "VALID_CANDIDATE",
    "PASSED_GATE",
    "FAILED_GATE",
    "BLOCKED_BUDGET",
    "BLOCKED_HUMAN_REVIEW",
)


def _loop(node_type: str) -> InnerLoopContract:
    if node_type in {"code_modification", "baseline_reproduction"}:
        return InnerLoopContract(
            ("draft", "refine", "debug", "revert"),
            ("execution", "constraint", "performance", "code_diff"),
            (
                "allowed_files", "valid_execution", "metric_parseable", "no_test_access",
                "proposal_reachable_call_path", "runtime_activation_marker",
            ),
            COMMON_OUTCOMES,
            {"candidate_steps": 8, "debug_depth": 3},
        )
    if node_type in {"experiment", "replication", "ablation", "statistical_analysis"}:
        return InnerLoopContract(
            ("run", "replicate", "refine_protocol", "analyze"),
            ("execution", "performance", "stability", "experimental_design", "cost"),
            ("frozen_split", "matched_budget", "complete_logs", "no_test_feedback"),
            COMMON_OUTCOMES,
            {"candidate_steps": 8, "minimum_seeds": 3},
        )
    if node_type in {"paper_outline", "paper_methods", "paper_writing"}:
        return InnerLoopContract(
            ("draft", "restructure", "revise", "remove_unsupported_claim"),
            ("claim_evidence", "statistics", "citation", "reproducibility", "clarity"),
            ("no_invented_results", "all_numbers_traceable", "evidence_version_bound"),
            COMMON_OUTCOMES,
            {"candidate_steps": 20, "review_rounds": 2},
        )
    if node_type in {"paper_integrity", "paper_peer_review"}:
        return InnerLoopContract(
            ("audit", "adversarial_review", "trace_upstream", "request_amendment"),
            ("claim_evidence", "statistics", "citation", "methodology", "reproducibility"),
            ("zero_untraceable_numbers", "zero_known_fabrications", "all_issues_accounted"),
            COMMON_OUTCOMES,
            {"review_rounds": 3},
        )
    return InnerLoopContract(
        ("inspect", "freeze", "validate"),
        ("contract", "provenance", "integrity"),
        ("complete_artifacts", "hashes_present"),
        COMMON_OUTCOMES,
        {"candidate_steps": 4},
    )


def _node(
    node_id: str,
    title: str,
    node_type: str,
    dependencies: list[str],
    outputs: list[str],
    gates: list[str],
    **metadata: Any,
) -> ResearchNode:
    return ResearchNode(
        node_id=node_id,
        title=title,
        node_type=node_type,
        depends_on=dependencies,
        output_artifacts=outputs,
        unlock_conditions=gates,
        inner_loop=_loop(node_type),
        metadata=metadata,
        status=NodeStatus.READY if not dependencies else NodeStatus.LOCKED,
    )


def _policy_candidates_by_role(
    task: dict[str, Any],
    agents: list[dict[str, Any]],
    knowledge_graph: dict[str, Any] | None,
    research_request: str,
    node_type: str,
) -> dict[str, list[str]]:
    roles = policy_roles_for_node(node_type)
    fallback = {
        role: [policy_id(agent["agent_id"], role) for agent in agents]
        for role in roles
    }
    if not knowledge_graph or not roles:
        return fallback
    context = " ".join(
        value for value in (
            task.get("task_id", ""), task.get("domain", ""), task.get("task_description", ""), research_request,
        ) if value
    )
    retrieved = query_knowledge_graph(
        knowledge_graph,
        context,
        node_types={"Policy", "Skill"},
        stage=node_type,
        limit=128,
    )
    ranked: dict[str, list[str]] = {role: [] for role in roles}
    for row in retrieved:
        attributes = row.get("attributes", {})
        if row.get("maturity") in {"contradiction", "superseded"}:
            continue
        if row.get("node_type") == "Skill" and not attributes.get("active"):
            continue
        role = attributes.get("policy_role")
        if role in ranked:
            ranked[role].append(row["node_id"])
    return {
        role: list(dict.fromkeys(ranked[role] + fallback[role]))
        for role in roles
    }


def _bind_adaptive_contracts(
    nodes: list[ResearchNode],
    *,
    task: dict[str, Any],
    agents: list[dict[str, Any]],
    knowledge_graph: dict[str, Any] | None,
    research_request: str,
    budget: dict[str, float | int],
) -> None:
    task_validation_metric = f"metric:task::{task['task_id']}::{task['canonical_metric']}::validation"
    task_test_metric = f"metric:task::{task['task_id']}::{task['canonical_metric']}::protected_test"
    process_metrics = [
        "metric:process::Exploration Spread", "metric:process::Exploration Uniqueness",
        "metric:process::Exploration Reach", "metric:process::Valid step ratio",
        "metric:process::AUC-over-steps", "metric:process::Token cost (M)",
        "metric:process::Wall-clock time (h)",
    ]
    adaptive_policy_types = {"method_search", "code_modification", "experiment", "replication", "ablation"}
    paper_types = {"paper_outline", "paper_methods", "paper_writing", "paper_integrity", "paper_peer_review"}
    for node in nodes:
        if node.node_type in adaptive_policy_types and node.node_id != "protected-final-test":
            roles = list(policy_roles_for_node(node.node_type))
            node.metadata["policy_roles"] = roles
            node.metadata["policy_candidates_by_role"] = _policy_candidates_by_role(
                task,
                agents,
                knowledge_graph,
                research_request,
                node.node_type,
            )
            node.metadata["selection_unit"] = "atomic_policy_composition"
            node.metadata["source_strategy_bundles"] = [f"strategy:{agent['agent_id']}" for agent in agents]
            node.metadata["strategy_selection_time"] = "compose_compatible_policies_when_dependency_ready"
        elif node.node_type in paper_types:
            node.metadata["strategy_candidates"] = (
                ["strategy:evidence_gated_paper_writing"]
                if node.node_type in {"paper_outline", "paper_methods", "paper_writing"}
                else ["strategy:multi_role_paper_review", "strategy:claim_evidence_audit"]
            )
            node.metadata["strategy_selection_time"] = "when_dependency_ready_using_current_evidence_bundle"
        else:
            node.metadata["strategy_candidates"] = ["strategy:deterministic_contract_validation"]
            node.metadata["strategy_selection_time"] = "fixed_governance_operator"
        node.metadata["controller_policy"] = {
            "candidate_scope": "node_local_atomic_policies_not_full_pipeline_bundles",
            "selection": "deterministic_compatible_composition_over_frozen_graph_snapshot",
            "record_visible_evidence": True,
            "record_rejected_candidates": True,
            "policy_credit_requires_explicit_trace": True,
            "remaining_budget": budget,
        }
        if node.node_id in {"method-frontier", "hypothesis-frontier"}:
            node.metadata["metric_bindings"] = [
                "metric:process::Exploration Spread", "metric:process::Exploration Uniqueness",
                "metric:process::Exploration Reach", "metric:process::Token cost (M)",
                "metric:process::Wall-clock time (h)",
            ]
            node.metadata["allowed_metric_observability"] = ["ONLINE_VISIBLE", "POST_STAGE_VISIBLE"]
        elif node.node_id == "code-modification":
            node.metadata["metric_bindings"] = [
                "metric:process::Valid step ratio", "metric:process::Token cost (M)",
                "metric:process::Wall-clock time (h)",
            ]
            node.metadata["learned_failure_gate"] = "proposal -> diff -> reachable call path -> runtime marker"
            node.metadata["allowed_metric_observability"] = ["ONLINE_VISIBLE", "POST_STAGE_VISIBLE"]
        elif node.node_id in {"visible-validation", "baseline-reproduction", "replication", "ablation"}:
            node.metadata["metric_bindings"] = [
                task_validation_metric, "metric:process::Valid step ratio",
                "metric:process::Token cost (M)", "metric:process::Wall-clock time (h)",
            ]
            node.metadata["allowed_metric_observability"] = ["ONLINE_VISIBLE", "POST_STAGE_VISIBLE"]
        elif node.node_id == "protected-final-test":
            node.metadata["metric_bindings"] = [task_test_metric]
            node.metadata["allowed_metric_observability"] = ["PROTECTED_FINAL_ONLY"]
            node.metadata["controller_routing_allowed"] = False
        elif node.node_id in {"process-evaluation", "confirmatory-analysis"}:
            node.metadata["metric_bindings"] = process_metrics + ([task_test_metric] if node.node_id == "confirmatory-analysis" else [])
            node.metadata["allowed_metric_observability"] = (
                ["ONLINE_VISIBLE", "POST_STAGE_VISIBLE"]
                if node.node_id == "process-evaluation"
                else ["POST_STAGE_VISIBLE", "PROTECTED_FINAL_ONLY"]
            )
        elif node.node_type in paper_types:
            node.metadata["metric_bindings"] = ["paper-review:hard-gates", "paper-review:seven-criterion-rubric"]
            node.metadata["allowed_metric_observability"] = ["PAPER_REVIEW_ONLY"]
        else:
            node.metadata["metric_bindings"] = []
            node.metadata["allowed_metric_observability"] = ["ONLINE_VISIBLE", "POST_STAGE_VISIBLE"]


def build_research_paper_plan(
    task: dict[str, Any],
    agents: list[dict[str, Any]],
    *,
    knowledge_graph: dict[str, Any] | None = None,
    research_request: str = "",
    budget: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    task_id = task["task_id"]
    prior_guidance = published_guidance_for_task(task_id)
    frozen_budget = dict(budget or {"candidate_steps": 8, "token_budget": 0, "wall_clock_hours": 0})
    nodes = [
        _node("memory-snapshot", "Freeze memory and skill snapshot", "governance", [], ["memory_snapshot.json"], ["versioned", "retrieval_boundary_frozen"]),
        _node("research-contract", "Freeze research and paper claim contracts", "contract", ["memory-snapshot"], ["research_contract.json", "claim_contract.json"], ["task_metric_frozen", "budgets_frozen", "test_boundary_frozen"]),
        _node("literature-frontier", "Retrieve and verify the primary-source literature frontier", "literature_search", ["research-contract"], ["literature_evidence.json", "related_work_notes.md", "literature_sources.json"], ["primary_sources_only", "source_manifest_hashed", "no_fabricated_citations"]),
        _node("method-frontier", "Learn and select baseline-agent search operators", "method_search", ["research-contract"], ["method_frontier.json"], ["all_agent_cards_loaded", "published_prior_labeled_nonclaim_evidence"], agent_ids=[agent["agent_id"] for agent in agents], published_prior_guidance=prior_guidance),
        _node("baseline-reproduction", "Reproduce the task baseline", "baseline_reproduction", ["research-contract"], ["baseline_reproduction.json"], ["baseline_metric_matches_contract"], baseline_validation=task["baseline_validation"]),
        _node("paper-outline", "Prepare provisional paper outline", "paper_outline", ["research-contract"], ["paper_outline.md"], ["provisional_no_results_claims"], provisional=True),
        _node("paper-methods", "Prepare provisional method and system description", "paper_methods", ["method-frontier"], ["methods_skeleton.md"], ["provisional_no_results_claims"], provisional=True),
        _node("hypothesis-frontier", "Generate and review ML hypotheses", "method_search", ["literature-frontier", "method-frontier", "baseline-reproduction"], ["hypothesis_frontier.json"], ["hypotheses_falsifiable", "baseline_grounded", "literature_grounded"]),
        _node("experiment-design", "Freeze candidates, controls, attribution, metrics, seeds, and budgets", "experiment_design", ["hypothesis-frontier"], ["preregistered_experiment_contract.json", "candidate_matrix.json"], ["controls_frozen", "activation_targets_declared", "reproducibility_frozen", "resource_profile_frozen"]),
        _node("code-modification", "Implement competing method candidates", "code_modification", ["experiment-design"], ["candidate_code_snapshots.json"], ["allowed_files_only", "source_hashes_recorded", "activation_contract_materialized"]),
        _node("visible-validation", "Run visible FML validation experiments", "experiment", ["code-modification"], ["validation_trajectory.jsonl"], ["validation_only", "complete_metric_output"], metric=task["canonical_metric"], direction=task["canonical_direction"]),
        _node("replication", "Replicate selected improvements", "replication", ["visible-validation"], ["replication_results.json"], ["minimum_three_seeds", "same_protocol"]),
        _node("ablation", "Run required mechanism ablations", "ablation", ["visible-validation"], ["ablation_results.json"], ["single_mechanism_changes", "matched_budget"]),
        _node("process-evaluation", "Compute FML process-level research metrics", "statistical_analysis", ["visible-validation"], ["process_metrics.csv"], ["all_steps_in_denominator", "cost_complete", "metric_semantic_audit_acknowledged"], metric_semantic_audit="knowledge_base/metric_implementation_audit.json", scorer_version_frozen=True),
        _node("evidence-freeze", "Freeze validation evidence and search history", "governance", ["replication", "ablation", "process-evaluation"], ["evidence_bundle_validation.json"], ["all_required_reviews_pass", "bundle_hashed"]),
        _node("protected-final-test", "Run protected FML test once", "experiment", ["evidence-freeze"], ["sealed_final_test.json"], ["best_candidate_frozen", "result_not_returned_to_search"], feedback_to_search=False),
        _node("confirmatory-analysis", "Run matched confirmatory statistics and scorer sensitivities", "statistical_analysis", ["protected-final-test"], ["paired_agent_comparisons.csv", "paired_agent_task_effects.csv", "adaptive_opportunity_interactions.csv", "scorer_semantics_sensitivity.csv"], ["complete_seed_blocks_only", "task_repeated_measures_not_pseudoreplicated", "holm_pairwise_adjustment", "published_partition_labels_frozen_before_new_outcomes", "official_scorer_columns_unchanged"], statistical_contract="protocol/statistical_analysis_protocol.json", primary_unit="matched complete seed block"),
        _node("final-evidence-bundle", "Integrate frozen paper evidence", "governance", ["confirmatory-analysis"], ["paper_evidence_bundle.json"], ["claim_evidence_graph_complete", "supersession_chain_valid", "statistical_claims_traceable"]),
        _node("final-paper-writing", "Write evidence-grounded final paper", "paper_writing", ["paper-outline", "paper-methods", "final-evidence-bundle"], ["manuscript.tex", "claim_evidence_map.json"], ["all_research_dependencies_pass", "all_numbers_traceable", "no_stale_evidence"]),
        _node("paper-integrity", "Run blocking pre-review integrity and AI-failure audit", "paper_integrity", ["final-paper-writing"], ["pre_review_integrity_report.json", "ai_failure_mode_audit.json"], ["zero_untraceable_numbers", "zero_known_fabrications", "failure_modes_cleared_or_human_acknowledged"], paper_evaluation_contract="paper_evaluation/paper_evaluation_protocol.json", paper_metric_boundary="paper quality is a separate local extension and is never averaged with FML metrics", mandatory_checkpoint=True),
        _node("paper-peer-review", "Run full multi-role peer review", "paper_peer_review", ["paper-integrity"], ["editorial_decision.json", "review_reports.json", "revision_roadmap.json"], ["all_review_issues_accounted"], reviewer_roles=["editor_in_chief", "methodology", "statistics", "reproducibility", "devils_advocate"], published_prior_checks=["published and local evidence classes are not merged", "opportunity partition is labeled post-hoc", "pooled process p-values are not treated as multiplicity-corrected", "AUC overlap with final performance is disclosed", "paper-versus-scorer process-metric semantic differences are disclosed", "primary comparisons use matched complete seed blocks", "Holm-adjusted and raw p-values are distinguished", "non-significance is not described as equivalence"], mandatory_checkpoint=True),
        _node("review-amendment-triage", "Trace every review issue to writing or upstream evidence", "paper_integrity", ["paper-peer-review"], ["review_issue_trace.json", "versioned_amendment_plan.json"], ["every_issue_has_owner", "upstream_defects_create_new_graph_version"], amendment_targets=["code-modification", "visible-validation", "replication", "ablation", "process-evaluation", "confirmatory-analysis"], protected_test_rule="post-exposure empirical changes require a new hidden evaluation"),
        _node("paper-revision", "Revise manuscript and answer every review issue", "paper_writing", ["review-amendment-triage"], ["revised_manuscript.tex", "response_to_reviewers.md"], ["all_review_items_addressed_or_explained", "evidence_versions_current"], mandatory_checkpoint=True),
        _node("paper-re-review", "Verify revision responses and residual issues", "paper_peer_review", ["paper-revision"], ["rereview_report.json", "residual_issue_matrix.json"], ["revision_claims_verified", "residual_issues_classified"], mandatory_checkpoint=True),
        _node("paper-re-revision", "Resolve major residual issues or record limitations", "paper_writing", ["paper-re-review"], ["second_revised_manuscript.tex", "residual_response.md"], ["major_residuals_resolved", "unresolved_items_acknowledged_as_limitations"], conditional="execute substantive revision only for major residual issues"),
        _node("final-paper-integrity", "Re-verify the full paper independently from scratch", "paper_integrity", ["paper-re-revision"], ["final_integrity_report.json", "final_failure_mode_audit.json"], ["zero_integrity_issues", "all_claims_resolve_to_current_evidence"], mandatory_checkpoint=True),
        _node("paper-finalize", "Finalize reproducible paper package", "paper_writing", ["final-paper-integrity"], ["paper.pdf", "reproducibility_manifest.json"], ["final_integrity_pass", "review_cycle_complete"], mandatory_checkpoint=True),
        _node("paper-process-summary", "Record the complete research and paper creation process", "governance", ["paper-finalize"], ["paper_creation_process_record.md", "audit_timeline.json"], ["all_stage_transitions_recorded", "all_overrides_attributed"]),
        _node(
            "skill-evolution",
            "Run post-arm Read-Write-Assess-Govern skill evolution",
            "governance",
            ["paper-process-summary"],
            [
                "trajectory_buffer_report.json", "skill_distillation_report.json",
                "utility_assessment_report.json", "skill_governance_report.json",
            ],
            [
                "informative_trajectory_gate", "create_patch_none_decision", "structured_trigger_and_procedure",
                "contextual_advantage_assessed", "held_out_validation", "authoring_episode_not_reused_for_validation",
                "no_mid_arm_mutation", "rollback_available",
            ],
            skill_channels=["research_execution", "research_review", "paper_writing", "paper_review"],
            skill_branches=["general", "task_specific", "action"],
            lifecycle=["READ", "WRITE", "ASSESS", "GOVERN"],
            execution_command="python3 -m ml_scientist.cli adaptive-runtime-evolve",
            runtime_action="AdaptiveResearchRuntime.run_skill_evolution",
        ),
    ]
    _bind_adaptive_contracts(
        nodes,
        task=task,
        agents=agents,
        knowledge_graph=knowledge_graph,
        research_request=research_request,
        budget=frozen_budget,
    )
    edges = [
        {"source": dependency, "target": node.node_id, "relation": "depends_on"}
        for node in nodes
        for dependency in node.depends_on
    ]
    plan = PipelinePlan(
        task_id=task_id,
        stages=STAGES,
        nodes=nodes,
        edges=edges,
        memory_contract={
            "snapshot_frozen_per_arm": True,
            "pending_writes_visible_to_current_arm": False,
            "registries": ["research_execution", "research_review", "paper_writing", "paper_review"],
            "skill_branches": ["general", "task_specific", "action"],
            "lifecycle": ["READ", "WRITE", "ASSESS", "GOVERN"],
            "contextual_utility": True,
            "protected_test_retrievable": False,
        },
        protected_test_contract={
            "runs_after_search_freeze": True,
            "maximum_runs": 1,
            "feedback_to_search": False,
            "post_exposure_code_changes_require_new_hidden_evaluation": True,
        },
    ).to_dict()
    plan["schema_version"] = "fml-scientist-conditional-dag-v5"
    plan["plan_mode"] = "task_conditioned_atomic_policy_composition"
    plan["research_request"] = research_request
    plan["research_context"] = {
        "task_id": task_id,
        "task_domain": task.get("domain"),
        "task_description": task.get("task_description"),
        "canonical_metric": task.get("canonical_metric"),
        "canonical_direction": task.get("canonical_direction"),
        "user_research_request": research_request,
    }
    plan["knowledge_graph_snapshot"] = {
        "required": True,
        "content_sha256": knowledge_graph.get("content_sha256") if knowledge_graph else None,
        "status": "FROZEN_GRAPH_BOUND" if knowledge_graph else "GRAPH_NOT_SUPPLIED_COMPATIBILITY_MODE",
    }
    plan["controller_contract"] = {
        "one_conditional_pipeline_per_research_job": True,
        "node_local_strategy_candidates": False,
        "node_local_atomic_policy_candidates": True,
        "full_baseline_bundles_are_provenance_only": True,
        "policy_composition_requires_capability_validation": True,
        "policy_credit_requires_explicit_trace": True,
        "cartesian_full_pipeline_expansion": False,
        "full_pipeline_variants_only_for_preregistered_ablation": True,
        "selection_uses_visible_evidence_only": True,
        "protected_final_metrics_may_route_search": False,
        "decision_record_required": True,
        "enriched_transition_record_required": True,
        "transition_fields": [
            "research_context", "state_before", "action", "next_state", "resource_telemetry",
            "policy_evaluations", "evidence_artifacts",
        ],
        "budget": frozen_budget,
        "transitions": {
            "PASSED_GATE": "ADVANCE",
            "VALID_CANDIDATE": "ADVANCE_OR_REPLICATE",
            "INVALID_EXECUTION": "DEBUG_RETRY_OR_SELECT_ALTERNATIVE",
            "VALID_REGRESSION": "SELECT_ALTERNATIVE_OR_BACKTRACK",
            "STOCHASTIC_UNCERTAIN": "REPLICATE_IF_BUDGET_ALLOWS",
            "FAILED_GATE": "BACKTRACK_OR_CREATE_VERSIONED_AMENDMENT",
            "BLOCKED_BUDGET": "STOP_AND_DOCUMENT_LIMITATION",
        },
    }
    plan["skill_evolution_contract"] = {
        "lifecycle": ["READ", "WRITE", "ASSESS", "GOVERN"],
        "write_intents": ["CREATE", "PATCH", "NONE"],
        "branches": ["general", "task_specific", "action"],
        "retrieve_at": "major_stage_boundary_and_keep_fixed_inside_node_inner_loop",
        "candidate_visibility": "next_frozen_snapshot_only",
        "authoring_episode_may_validate_candidate": False,
        "utility": "contextual_advantage_from_explicit_environment_feedback",
        "promotion": "later_held_out_complete_downstream_no_regression_only",
        "harmful_comparable_evidence": "rollback_and_retain_contradiction",
        "protected_final_metrics_may_evolve_search_skills": False,
    }
    plan["academic_publication_contract"] = {
        "workflow": [
            "research",
            "write",
            "pre_review_integrity",
            "full_review",
            "revise",
            "re_review",
            "conditional_re_revise",
            "final_integrity",
            "finalize",
            "process_summary",
        ],
        "mandatory_human_checkpoints": [
            "paper-integrity",
            "paper-peer-review",
            "paper-revision",
            "paper-re-review",
            "final-paper-integrity",
            "paper-finalize",
        ],
        "integrity_checks_are_blocking": True,
        "final_integrity_restarts_from_scratch": True,
        "maximum_full_revision_rounds": 2,
        "all_review_issues_must_be_accounted_for": True,
    }
    validate_plan(plan)
    return plan


def validate_plan(plan: dict[str, Any]) -> None:
    nodes = {node["node_id"]: node for node in plan["nodes"]}
    if len(nodes) != len(plan["nodes"]):
        raise ValueError("Duplicate node IDs")
    indegree = {node_id: 0 for node_id in nodes}
    outgoing: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        if node.get("planned_at_stage") != 3:
            raise ValueError(f"{node['node_id']} was not planned at Stage 3")
        for dependency in node["depends_on"]:
            if dependency not in nodes:
                raise ValueError(f"Unknown dependency {dependency}")
            indegree[node["node_id"]] += 1
            outgoing[dependency].append(node["node_id"])
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for target in outgoing[current]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    if visited != len(nodes):
        raise ValueError("Task graph contains a cycle; amendments must create graph versions")
    paper = nodes["final-paper-writing"]
    required = {"paper-outline", "paper-methods", "final-evidence-bundle"}
    if not required.issubset(set(paper["depends_on"])):
        raise ValueError("Final paper is not locked behind all evidence dependencies")
    protected = nodes["protected-final-test"]
    if protected["metadata"].get("feedback_to_search") is not False:
        raise ValueError("Protected test must not feed the search loop")
    for node in nodes.values():
        metadata = node.get("metadata", {})
        if metadata.get("selection_unit") != "atomic_policy_composition":
            continue
        roles = metadata.get("policy_roles", [])
        candidates = metadata.get("policy_candidates_by_role", {})
        if not roles:
            raise ValueError(f"Atomic policy node has no roles: {node['node_id']}")
        missing = [role for role in roles if not candidates.get(role)]
        if missing:
            raise ValueError(f"Atomic policy node has empty candidate roles {missing}: {node['node_id']}")
        if metadata.get("strategy_candidates"):
            raise ValueError(f"Atomic policy node still selects full strategies: {node['node_id']}")
    required_publication_chain = [
        "final-paper-writing",
        "paper-integrity",
        "paper-peer-review",
        "review-amendment-triage",
        "paper-revision",
        "paper-re-review",
        "paper-re-revision",
        "final-paper-integrity",
        "paper-finalize",
        "paper-process-summary",
    ]
    for predecessor, successor in zip(required_publication_chain, required_publication_chain[1:]):
        if predecessor not in nodes[successor]["depends_on"]:
            raise ValueError(f"Academic publication chain is broken at {predecessor} -> {successor}")
    triage = nodes["review-amendment-triage"]["metadata"]
    if "code-modification" not in triage.get("amendment_targets", []):
        raise ValueError("Paper review cannot trace empirical errors back to code")
    if not plan.get("academic_publication_contract", {}).get("final_integrity_restarts_from_scratch"):
        raise ValueError("Final paper integrity must be independently rerun from scratch")
