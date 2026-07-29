"""Typed, versioned knowledge graph for benchmark, strategy, episode, and claim memory."""

from __future__ import annotations

import hashlib
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .policy_model import build_agent_policy_specs


NODE_TYPES = {
    "Benchmark", "Task", "Metric", "Constraint", "Agent", "Strategy", "Policy", "Skill",
    "PipelineRecipe", "PipelineStage", "Episode", "Decision", "EvidenceArtifact",
    "Failure", "Claim", "PaperCriterion", "Memory", "Dataset", "ResourceBudget",
    "ResearchContext",
}

RELATION_TYPES = {
    "HAS_TASK", "EVALUATED_BY", "HAS_VIEW", "AVAILABLE_AT", "IMPLEMENTS",
    "APPLICABLE_TO", "DERIVED_FROM", "USES_SKILL", "HAS_STAGE", "DEPENDS_ON",
    "PRODUCED", "SUPPORTS", "CONTRADICTS", "VALIDATED_ON", "SELECTED",
    "OBSERVED", "VIOLATED", "AMENDS", "INVALIDATES", "SUPERSEDES", "COSTS",
    "DESCRIBED_BY", "REVIEWED_BY", "MIGRATED_FROM", "USES_METRIC",
    "HAS_POLICY", "CREDITED_TO", "REQUIRES_POLICY", "COMPATIBLE_WITH", "CONFLICTS_WITH",
    "OCCURRED_IN", "ASSESSED_IN",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_file(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _sha_payload(payload: Any) -> str:
    return _sha_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def _source(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = str(resolved.relative_to(root.resolve()))
    except ValueError:
        display = str(resolved)
    return {"path": display, "sha256": _sha_file(resolved)}


def _safe_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _tokens(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9_]+", value.lower()) if len(token) > 1}


class GraphBuilder:
    def __init__(self, repository_commit: str):
        self.repository_commit = repository_commit
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}

    def add_node(
        self,
        node_id: str,
        node_type: str,
        label: str,
        *,
        maturity: str = "observation",
        provenance: Iterable[dict[str, Any]] = (),
        attributes: dict[str, Any] | None = None,
    ) -> str:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown knowledge node type: {node_type}")
        incoming = {
            "node_id": node_id,
            "node_type": node_type,
            "label": label,
            "maturity": maturity,
            "provenance": list(provenance),
            "attributes": attributes or {},
        }
        existing = self.nodes.get(node_id)
        if existing is not None and existing != incoming:
            raise ValueError(f"Conflicting duplicate knowledge node: {node_id}")
        self.nodes[node_id] = incoming
        return node_id

    def add_edge(
        self,
        source: str,
        target: str,
        relation: str,
        *,
        provenance: Iterable[dict[str, Any]] = (),
        attributes: dict[str, Any] | None = None,
    ) -> str:
        if relation not in RELATION_TYPES:
            raise ValueError(f"Unknown knowledge relation type: {relation}")
        edge_key = f"{source}|{relation}|{target}"
        edge_id = "edge:" + _sha_bytes(edge_key.encode())[:20]
        incoming = {
            "edge_id": edge_id,
            "source": source,
            "target": target,
            "relation": relation,
            "provenance": list(provenance),
            "attributes": attributes or {},
        }
        existing = self.edges.get(edge_id)
        if existing is not None and existing != incoming:
            raise ValueError(f"Conflicting duplicate knowledge edge: {edge_id}")
        self.edges[edge_id] = incoming
        return edge_id

    def to_dict(self) -> dict[str, Any]:
        graph = {
            "schema_version": "ml-scientist-knowledge-graph-v1",
            "repository_commit": self.repository_commit,
            "canonical_memory": True,
            "memory_policy": {
                "frozen_snapshot_per_arm": True,
                "mid_arm_mutation_allowed": False,
                "protected_final_metrics_retrievable_by_search": False,
                "negative_and_contradictory_evidence_retained": True,
                "skill_evolution_lifecycle": ["READ", "WRITE", "ASSESS", "GOVERN"],
                "authoring_episode_may_validate_candidate": False,
                "context_dependent_utility": True,
            },
            "nodes": sorted(self.nodes.values(), key=lambda row: row["node_id"]),
            "edges": sorted(self.edges.values(), key=lambda row: row["edge_id"]),
        }
        graph["content_sha256"] = _sha_payload(graph)
        return graph


def _add_catalog(
    builder: GraphBuilder,
    catalog: dict[str, Any],
    source: dict[str, Any],
    policy_source: dict[str, Any],
) -> None:
    benchmark_id = builder.add_node(
        "benchmark:fml-bench", "Benchmark", "FML-bench",
        maturity="verified", provenance=[source],
        attributes={"version": catalog["repository_commit"], "evaluation_boundary": catalog["evaluation_boundary"]},
    )
    for agent in catalog["agents"]:
        agent_id = builder.add_node(
            f"agent:{agent['agent_id']}", "Agent", agent["display_name"],
            maturity="verified", provenance=[source],
            attributes={"strategy_family": agent["strategy_family"], "config": agent["config"]},
        )
        strategy_id = builder.add_node(
            f"strategy:{agent['agent_id']}", "Strategy", agent["display_name"] + " search strategy",
            maturity="observation", provenance=[source],
            attributes={
                "agent_id": agent["agent_id"],
                "strategy_family": agent["strategy_family"],
                "selection_policy": agent["selection_policy"],
                "proposal_operator": agent["proposal_operator"],
                "debug_policy": agent["debug_policy"],
                "memory_policy": agent["memory_policy"],
                "diversity_mechanism": agent["diversity_mechanism"],
                "state_representation": agent["state_representation"],
                "entity_semantics": "source_pipeline_bundle_for_provenance_and_reference_composition",
                "selectable_unit": False,
            },
        )
        builder.add_edge(agent_id, strategy_id, "IMPLEMENTS", provenance=[source])
        builder.add_edge(strategy_id, benchmark_id, "APPLICABLE_TO", provenance=[source])
        for policy in build_agent_policy_specs(agent):
            policy_node_id = policy.pop("policy_id")
            builder.add_node(
                policy_node_id,
                "Policy",
                f"{agent['display_name']} {policy['policy_role']} policy",
                maturity="observation",
                provenance=[source, policy_source],
                attributes=policy,
            )
            builder.add_edge(strategy_id, policy_node_id, "HAS_POLICY", provenance=[source, policy_source])
            builder.add_edge(policy_node_id, strategy_id, "DERIVED_FROM", provenance=[source, policy_source])
    for task in catalog["tasks"]:
        task_id = builder.add_node(
            f"task:{task['task_id']}", "Task", task["task_id"], maturity="verified", provenance=[source],
            attributes={
                "domain": task["domain"], "dataset": task["canonical_dataset"],
                "canonical_metric": task["canonical_metric"], "direction": task["canonical_direction"],
                "validation_command": task["validation_command"],
                "protected_test_command": task["protected_test_command"],
                "target_files": task["target_files"],
            },
        )
        builder.add_edge(benchmark_id, task_id, "HAS_TASK", provenance=[source])
        dataset_id = builder.add_node(
            f"dataset:{task['task_id']}:{task['canonical_dataset']}", "Dataset",
            task["canonical_dataset"], maturity="verified", provenance=[source],
            attributes={"task_id": task["task_id"]},
        )
        builder.add_edge(task_id, dataset_id, "DESCRIBED_BY", provenance=[source])


def _add_current_pipeline_strategies(builder: GraphBuilder, repo: Path) -> None:
    planner_path = repo / "ml_scientist" / "planner.py"
    source = _source(planner_path, repo)
    rows = [
        ("strategy:evidence_gated_paper_writing", "Evidence-gated paper writing", ["paper_outline", "paper_methods", "paper_writing"]),
        ("strategy:multi_role_paper_review", "Multi-role paper review", ["paper_integrity", "paper_peer_review"]),
        ("strategy:claim_evidence_audit", "Claim-evidence audit", ["paper_integrity", "paper_peer_review"]),
        ("strategy:deterministic_contract_validation", "Deterministic contract validation", ["governance", "contract"]),
    ]
    for node_id, label, stages in rows:
        builder.add_node(
            node_id, "Strategy", label, maturity="observation", provenance=[source],
            attributes={"learned_from_current_pipeline": True, "eligible_stages": stages},
        )


def _add_benchmark_contract(builder: GraphBuilder, contract: dict[str, Any], source: dict[str, Any]) -> None:
    benchmark_id = f"benchmark:{contract['benchmark_id']}"
    if benchmark_id not in builder.nodes:
        builder.add_node(
            benchmark_id, "Benchmark", contract["benchmark_id"],
            maturity=contract.get("maturity", "observation"), provenance=[source],
            attributes={"version": contract.get("benchmark_version"), "activation_eligible": contract.get("activation_eligible", False)},
        )
    for name, requirement in sorted(contract.get("evaluation_boundary", {}).items()):
        constraint_id = f"constraint:{contract['benchmark_id']}:{name}"
        builder.add_node(
            constraint_id, "Constraint", requirement, maturity="verified", provenance=[source],
            attributes={"constraint_kind": name, "blocking": True},
        )
        builder.add_edge(benchmark_id, constraint_id, "DESCRIBED_BY", provenance=[source])
    for metric in contract.get("metrics", []):
        metric_id = f"metric:{metric['metric_id']}"
        builder.add_node(
            metric_id, "Metric", metric.get("name", metric["metric_id"]),
            maturity=metric.get("maturity", contract.get("maturity", "observation")), provenance=[source],
            attributes={key: value for key, value in metric.items() if key not in {"metric_id", "name", "maturity"}},
        )
        task = metric.get("task_id")
        if task and f"task:{task}" in builder.nodes:
            builder.add_edge(f"task:{task}", metric_id, "EVALUATED_BY", provenance=[source])
        else:
            builder.add_edge(benchmark_id, metric_id, "EVALUATED_BY", provenance=[source])
        for stage in metric.get("stages", []):
            stage_id = f"stage:{stage}"
            if stage_id not in builder.nodes:
                builder.add_node(stage_id, "PipelineStage", stage, maturity="verified", provenance=[source])
            builder.add_edge(metric_id, stage_id, "AVAILABLE_AT", provenance=[source])


def _add_memory_registry(builder: GraphBuilder, path: Path, root: Path) -> None:
    payload = _safe_json(path)
    if not isinstance(payload, dict):
        return
    source = _source(path, root)
    for entry in payload.get("entries", []) + payload.get("pending_entries", []):
        memory_id = f"memory:{entry.get('memory_id', _sha_payload(entry)[:16])}"
        builder.add_node(
            memory_id, "Memory", entry.get("claim", entry.get("memory_id", "memory")),
            maturity=entry.get("maturity", "observation"), provenance=[source], attributes=entry,
        )
        builder.add_edge(memory_id, "benchmark:fml-bench", "MIGRATED_FROM", provenance=[source])


def _add_skill_registry(builder: GraphBuilder, path: Path, root: Path) -> None:
    payload = _safe_json(path)
    if not isinstance(payload, dict):
        return
    source = _source(path, root)
    for channel, skills in payload.get("channels", {}).items():
        for skill in skills:
            for version in skill.get("versions", []):
                skill_id = f"skill:{skill['skill_id']}:v{version['version']}"
                builder.add_node(
                    skill_id, "Skill", skill["skill_id"], maturity=version.get("maturity", "observation"),
                    provenance=[source],
                    attributes={
                        "channel": channel, "version": version["version"], "rule": version.get("rule"),
                        "status": version.get("status"), "active": skill.get("active_version") == version["version"],
                        "acceptance_gates": version.get("acceptance_gates", []),
                        "rollback_condition": version.get("rollback_condition"),
                        "policy_role": version.get("policy_role"),
                        "instruction": version.get("rule"),
                        "eligible_stages": version.get("eligible_stages", []),
                        "requires_capabilities": version.get("requires_capabilities", []),
                        "provides_capabilities": version.get("provides_capabilities", []),
                        "expected_cost": version.get("expected_cost", 0.1),
                        "credit_assignment": version.get("credit_assignment"),
                        "skill_branch": version.get("skill_branch", "task_specific"),
                        "research_contexts": version.get("research_contexts", []),
                        "structured_skill": version.get("structured_skill", {}),
                        "source_episode_ids": version.get("source_episode_ids", []),
                        "distillation": version.get("distillation", {}),
                        "fingerprint": version.get("fingerprint"),
                        "utility": version.get("utility", {"global": 0.5, "by_context": {}}),
                    },
                )
                for outcome in version.get("outcomes", []):
                    episode_id = f"episode:skill:{skill['skill_id']}:{_sha_payload(outcome)[:16]}"
                    builder.add_node(
                        episode_id, "Episode", f"{skill['skill_id']} outcome", maturity="verified",
                        provenance=[source], attributes=outcome,
                    )
                    builder.add_edge(skill_id, episode_id, "VALIDATED_ON", provenance=[source])
                for context_key, utility in version.get("utility", {}).get("by_context", {}).items():
                    context_id = f"research-context:utility:{_sha_payload(context_key)[:16]}"
                    if context_id not in builder.nodes:
                        builder.add_node(
                            context_id, "ResearchContext", context_key, maturity="verified",
                            provenance=[source], attributes={"context_key": context_key},
                        )
                    builder.add_edge(
                        skill_id, context_id, "ASSESSED_IN", provenance=[source],
                        attributes={"utility": utility},
                    )


def _add_dossiers(builder: GraphBuilder, knowledge_dir: Path, root: Path) -> None:
    for kind, entity_prefix in (("agents", "agent"), ("tasks", "task")):
        directory = knowledge_dir / kind
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name == "index.json":
                continue
            payload = _safe_json(path)
            if not isinstance(payload, dict):
                continue
            source = _source(path, root)
            entity = payload.get("agent_id") or payload.get("task_id")
            if not entity or f"{entity_prefix}:{entity}" not in builder.nodes:
                continue
            artifact_id = f"artifact:dossier:{entity_prefix}:{entity}"
            builder.add_node(
                artifact_id, "EvidenceArtifact", f"{entity} source-grounded dossier",
                maturity="verified", provenance=[source],
                attributes={"knowledge_status": payload.get("knowledge_status"), "kind": f"{entity_prefix}_dossier"},
            )
            builder.add_edge(f"{entity_prefix}:{entity}", artifact_id, "DESCRIBED_BY", provenance=[source])


def _add_paper_protocol(builder: GraphBuilder, path: Path, root: Path) -> None:
    payload = _safe_json(path)
    if not isinstance(payload, dict):
        return
    source = _source(path, root)
    for row in payload.get("hard_gates", []):
        node_id = f"paper-criterion:{row['gate_id']}"
        builder.add_node(node_id, "PaperCriterion", row["requirement"], maturity="verified", provenance=[source], attributes={"blocking": True, "observability": "PAPER_REVIEW_ONLY"})
    for row in payload.get("rubric", []):
        node_id = f"paper-criterion:{row['criterion']}"
        builder.add_node(node_id, "PaperCriterion", row["criterion"], maturity="verified", provenance=[source], attributes={**row, "blocking": False, "observability": "PAPER_REVIEW_ONLY"})


def _add_published_prior(builder: GraphBuilder, directory: Path, root: Path) -> None:
    if not directory.is_dir():
        return
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        source = _source(path, root)
        artifact_id = f"artifact:published-prior:{path.name}"
        builder.add_node(
            artifact_id, "EvidenceArtifact", f"Published prior: {path.name}", maturity="verified",
            provenance=[source], attributes={"evidence_class": "published_prior", "local_campaign_claim_eligible": False},
        )
        builder.add_edge("benchmark:fml-bench", artifact_id, "DESCRIBED_BY", provenance=[source])
    agent_summary = directory / "published_agent_summary.csv"
    if agent_summary.is_file():
        source = _source(agent_summary, root)
        with agent_summary.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                claim_id = f"claim:published:agent:{row['agent_id']}"
                builder.add_node(
                    claim_id, "Claim", f"Published aggregate for {row['agent_id']}", maturity="observation",
                    provenance=[source], attributes={**row, "local_campaign_claim_eligible": False},
                )
                builder.add_edge("artifact:published-prior:published_agent_summary.csv", claim_id, "SUPPORTS", provenance=[source])
                if f"agent:{row['agent_id']}" in builder.nodes:
                    builder.add_edge(claim_id, f"agent:{row['agent_id']}", "OBSERVED", provenance=[source])
    agent_task = directory / "published_agent_task_mean_sd.csv"
    if agent_task.is_file():
        source = _source(agent_task, root)
        with agent_task.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                claim_id = f"claim:published:agent-task:{index:03d}"
                builder.add_node(
                    claim_id, "Claim", f"Published {row['agent_id']} on {row['task_id']}", maturity="observation",
                    provenance=[source], attributes={**row, "local_campaign_claim_eligible": False},
                )
                builder.add_edge("artifact:published-prior:published_agent_task_mean_sd.csv", claim_id, "SUPPORTS", provenance=[source])
                for entity_id in (f"agent:{row['agent_id']}", f"task:{row['task_id']}"):
                    if entity_id in builder.nodes:
                        builder.add_edge(claim_id, entity_id, "OBSERVED", provenance=[source])
    process = directory / "published_process_metrics.csv"
    if process.is_file():
        source = _source(process, root)
        with process.open(encoding="utf-8", newline="") as handle:
            for index, row in enumerate(csv.DictReader(handle), start=1):
                claim_id = f"claim:published:process:{index:02d}"
                builder.add_node(
                    claim_id, "Claim", f"Published process metric: {row['metric']}", maturity="observation",
                    provenance=[source], attributes={**row, "local_campaign_claim_eligible": False, "multiplicity_corrected": False},
                )
                builder.add_edge("artifact:published-prior:published_process_metrics.csv", claim_id, "SUPPORTS", provenance=[source])
                metric_id = f"metric:process::{row['metric']}"
                if metric_id in builder.nodes:
                    builder.add_edge(claim_id, metric_id, "OBSERVED", provenance=[source])


def _add_operational_memory(builder: GraphBuilder, project_artifacts: Path, root: Path) -> list[dict[str, Any]]:
    migrated: list[dict[str, Any]] = []
    checkpoints = sorted((project_artifacts / "codex_ssh_protocol").glob("pause_checkpoint_*/README.md"))
    for path in checkpoints:
        source = _source(path, root)
        text = path.read_text(encoding="utf-8")
        memory_id = f"memory:pause-checkpoint:{path.parent.name}"
        builder.add_node(
            memory_id, "Memory", path.parent.name, maturity="verified", provenance=[source],
            attributes={
                "kind": "operational_pause_checkpoint",
                "experiment_status": "PAUSED_BY_USER",
                "resume_requires_explicit_user_authorization": True,
                "content_excerpt": text[:1200],
            },
        )
        builder.add_edge(memory_id, "benchmark:fml-bench", "APPLICABLE_TO", provenance=[source])
        if "Implementation-reachability finding" in text:
            failure_id = f"failure:{path.parent.name}:unreachable-proposal"
            builder.add_node(
                failure_id, "Failure", "Generated proposal was not reachable from the executed call path",
                maturity="verified", provenance=[source],
                attributes={
                    "failure_class": "implementation_reachability",
                    "scientific_result_eligible": False,
                    "required_future_gate": "proposal -> diff -> reachable call path -> runtime marker",
                    "preserve_interrupted_attempt": True,
                },
            )
            builder.add_edge(memory_id, failure_id, "OBSERVED", provenance=[source])
        for evidence_path in sorted((path.parent / "evidence").glob("*")):
            if not evidence_path.is_file():
                continue
            evidence_source = _source(evidence_path, root)
            artifact_id = f"artifact:pause:{path.parent.name}:{evidence_path.name}"
            builder.add_node(
                artifact_id, "EvidenceArtifact", evidence_path.name, maturity="verified",
                provenance=[evidence_source], attributes={"kind": "pause_checkpoint_evidence", "scientific_result_eligible": False},
            )
            builder.add_edge(memory_id, artifact_id, "PRODUCED", provenance=[source])
            migrated.append(evidence_source)
        migrated.append(source)
    live_status = project_artifacts / "pilot_live_report" / "experiment_dataset_status.json"
    if live_status.is_file():
        source = _source(live_status, root)
        payload = _safe_json(live_status)
        builder.add_node(
            "artifact:pilot-live-report-status", "EvidenceArtifact", "Pilot live report status",
            maturity="verified", provenance=[source], attributes={"kind": "experiment_dataset_status", "status": payload},
        )
        builder.add_edge("benchmark:fml-bench", "artifact:pilot-live-report-status", "DESCRIBED_BY", provenance=[source])
        migrated.append(source)
    return migrated


def _add_plans(builder: GraphBuilder, plans_dir: Path, root: Path) -> None:
    if not plans_dir.is_dir():
        return
    for path in sorted(plans_dir.glob("*.json")):
        plan = _safe_json(path)
        if not isinstance(plan, dict) or not plan.get("task_id"):
            continue
        source = _source(path, root)
        recipe_id = f"pipeline:{plan['task_id']}:{plan.get('schema_version', 'unknown')}"
        builder.add_node(
            recipe_id, "PipelineRecipe", f"{plan['task_id']} conditional research pipeline",
            maturity="observation", provenance=[source],
            attributes={"task_id": plan["task_id"], "plan_mode": plan.get("plan_mode", "fixed_template"), "schema_version": plan.get("schema_version")},
        )
        if f"task:{plan['task_id']}" in builder.nodes:
            builder.add_edge(recipe_id, f"task:{plan['task_id']}", "APPLICABLE_TO", provenance=[source])
        for node in plan.get("nodes", []):
            stage_id = f"stage:{node['node_id']}"
            if stage_id not in builder.nodes:
                builder.add_node(stage_id, "PipelineStage", node.get("title", node["node_id"]), maturity="observation", provenance=[source], attributes={"node_type": node.get("node_type")})
            builder.add_edge(recipe_id, stage_id, "HAS_STAGE", provenance=[source])
            for strategy in node.get("metadata", {}).get("strategy_candidates", []):
                strategy_id = strategy if str(strategy).startswith("strategy:") else f"strategy:{strategy}"
                if strategy_id not in builder.nodes:
                    builder.add_node(
                        strategy_id, "Strategy", strategy_id.removeprefix("strategy:").replace("_", " "),
                        maturity="observation", provenance=[source],
                        attributes={"learned_from_current_pipeline_plan": True, "eligible_stages": [node.get("node_type", "governance")]},
                    )
                builder.add_edge(stage_id, strategy_id, "APPLICABLE_TO", provenance=[source])
            for role, policies in node.get("metadata", {}).get("policy_candidates_by_role", {}).items():
                for policy_id in policies:
                    if policy_id in builder.nodes:
                        builder.add_edge(
                            stage_id,
                            policy_id,
                            "APPLICABLE_TO",
                            provenance=[source],
                            attributes={"policy_role": role},
                        )


def _add_episodes(builder: GraphBuilder, paths: Iterable[Path], root: Path) -> None:
    for path in paths:
        if not path.is_file():
            continue
        source = _source(path, root)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                episode = json.loads(line)
            except json.JSONDecodeError:
                continue
            episode_id = f"episode:{episode.get('episode_id') or _sha_payload(episode)[:20]}"
            builder.add_node(
                episode_id, "Episode", episode.get("label", episode_id),
                maturity=episode.get("maturity", "observation"), provenance=[{**source, "line": line_number}], attributes=episode,
            )
            decision_payload = {
                "node_id": episode.get("node_id"),
                "selected_strategy_id": episode.get("selected_strategy_id"),
                "selected_policy_ids": episode.get("selected_policy_ids", []),
                "selected_policy_by_role": episode.get("selected_policy_by_role", {}),
                "visible_evidence": episode.get("visible_evidence", []),
                "snapshot_sha256": episode.get("snapshot_sha256"),
                "budget_before": episode.get("budget_before", episode.get("budget", {})),
                "budget_after": episode.get("budget_after", {}),
                "research_context": episode.get("research_context", {}),
                "state_before": episode.get("state_before", {}),
                "next_state": episode.get("next_state", {}),
                "resource_telemetry": episode.get("resource_telemetry", {}),
            }
            decision_id = f"decision:{episode.get('decision_id') or _sha_payload(decision_payload)[:20]}"
            builder.add_node(
                decision_id, "Decision", f"Decision for {episode.get('node_id', 'unknown node')}",
                maturity=episode.get("maturity", "observation"), provenance=[{**source, "line": line_number}],
                attributes=decision_payload,
            )
            builder.add_edge(episode_id, decision_id, "OBSERVED", provenance=[source])
            research_context = episode.get("research_context")
            if isinstance(research_context, dict) and research_context:
                context_payload = {
                    "task_id": research_context.get("task_id"),
                    "task_domain": research_context.get("task_domain"),
                    "research_request": research_context.get("research_request"),
                }
                context_id = f"research-context:{_sha_payload(context_payload)[:20]}"
                if context_id not in builder.nodes:
                    builder.add_node(
                        context_id, "ResearchContext",
                        research_context.get("research_request") or research_context.get("task_id") or "research context",
                        maturity="observation", provenance=[{**source, "line": line_number}],
                        attributes=context_payload,
                    )
                builder.add_edge(episode_id, context_id, "OCCURRED_IN", provenance=[source])
            strategy = episode.get("selected_strategy_id")
            if strategy:
                strategy_id = strategy if str(strategy).startswith("strategy:") else f"strategy:{strategy}"
                if strategy_id in builder.nodes:
                    builder.add_edge(decision_id, strategy_id, "SELECTED", provenance=[source])
            selected_policies = episode.get("selected_policy_ids", [])
            for policy in selected_policies:
                policy_id = policy if str(policy).startswith("policy:") else f"policy:{policy}"
                if policy_id in builder.nodes:
                    builder.add_edge(decision_id, policy_id, "SELECTED", provenance=[source])
            for evaluation in episode.get("policy_evaluations", []):
                policy = evaluation.get("policy_id")
                policy_id = policy if str(policy).startswith("policy:") else f"policy:{policy}"
                if policy_id not in builder.nodes:
                    continue
                builder.add_edge(episode_id, policy_id, "CREDITED_TO", provenance=[source], attributes={
                    "policy_role": evaluation.get("role"),
                    "status": evaluation.get("status"),
                    "applied": bool(evaluation.get("applied")),
                    "credit_assignment": "explicit_policy_trace",
                })
                attributes = builder.nodes[policy_id]["attributes"]
                context = f"{episode.get('task_id', 'unknown')}::{episode.get('node_id', 'unknown')}"
                stage_key = episode.get("node_type") or episode.get("node_id", "unknown")
                status = evaluation.get("status")
                applied = bool(evaluation.get("applied"))
                if applied and status == "PASSED":
                    contexts = set(attributes.get("successful_context_ids", []))
                    contexts.add(context)
                    attributes["successful_context_ids"] = sorted(contexts)
                    attributes["successful_contexts"] = len(contexts)
                    by_stage = dict(attributes.get("successful_context_ids_by_stage", {}))
                    stage_contexts = set(by_stage.get(stage_key, []))
                    stage_contexts.add(context)
                    by_stage[stage_key] = sorted(stage_contexts)
                    attributes["successful_context_ids_by_stage"] = by_stage
                    attributes["successful_contexts_by_stage"] = {
                        stage: len(values) for stage, values in sorted(by_stage.items())
                    }
                elif applied and status == "FAILED":
                    attributes["failure_count"] = int(attributes.get("failure_count", 0)) + 1
                    failures_by_stage = dict(attributes.get("failure_count_by_stage", {}))
                    failures_by_stage[stage_key] = int(failures_by_stage.get(stage_key, 0)) + 1
                    attributes["failure_count_by_stage"] = failures_by_stage
                attributes["credit_assignment"] = "explicit_policy_trace_only"
            for evidence in episode.get("visible_evidence", []):
                metric = evidence.get("metric_id")
                if metric:
                    metric_id = metric if str(metric).startswith("metric:") else f"metric:{metric}"
                    if metric_id in builder.nodes:
                        builder.add_edge(decision_id, metric_id, "USES_METRIC", provenance=[source])
            task_id = episode.get("task_id")
            if task_id and f"task:{task_id}" in builder.nodes:
                builder.add_edge(episode_id, f"task:{task_id}", "VALIDATED_ON", provenance=[source])
            if episode.get("outcome") in {"INVALID_EXECUTION", "FAILED_GATE", "VALID_REGRESSION"}:
                failure_id = f"failure:episode:{episode.get('episode_id') or _sha_payload(episode)[:20]}"
                builder.add_node(
                    failure_id, "Failure", f"{episode.get('outcome')} at {episode.get('node_id')}",
                    maturity="observation", provenance=[{**source, "line": line_number}],
                    attributes={"outcome": episode.get("outcome"), "retained_for_future_routing": True},
                )
                builder.add_edge(episode_id, failure_id, "OBSERVED", provenance=[source])


def build_project_knowledge_graph(
    repo: Path,
    catalog: dict[str, Any],
    artifact_root: Path,
    *,
    episode_paths: Iterable[Path] = (),
    include_execution_recipes: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Migrate current learned artifacts into the canonical typed graph."""
    repo = repo.resolve()
    artifact_root = artifact_root.resolve()
    builder = GraphBuilder(catalog["repository_commit"])
    catalog_path = artifact_root / "catalog" / "fml_catalog.json"
    catalog_source = _source(catalog_path, repo) if catalog_path.is_file() else {"path": "in_memory_catalog", "sha256": _sha_payload(catalog)}
    policy_source = _source(repo / "ml_scientist" / "policy_model.py", repo)
    _add_catalog(builder, catalog, catalog_source, policy_source)
    _add_current_pipeline_strategies(builder, repo)

    migrated: list[dict[str, Any]] = [catalog_source, policy_source]
    benchmark_dir = artifact_root / "benchmark_contracts"
    for path in sorted(benchmark_dir.glob("*.json")) if benchmark_dir.is_dir() else []:
        if path.name == "index.json":
            continue
        contract = _safe_json(path)
        if isinstance(contract, dict) and contract.get("benchmark_id"):
            source = _source(path, repo)
            _add_benchmark_contract(builder, contract, source)
            migrated.append(source)

    memory_path = artifact_root / "governance" / "memory_registry.json"
    skill_path = artifact_root / "governance" / "skill_registry.json"
    if memory_path.is_file():
        _add_memory_registry(builder, memory_path, repo)
        migrated.append(_source(memory_path, repo))
    if skill_path.is_file():
        _add_skill_registry(builder, skill_path, repo)
        migrated.append(_source(skill_path, repo))

    _add_dossiers(builder, artifact_root / "knowledge_base", repo)
    for path in sorted((artifact_root / "knowledge_base").rglob("*.json")) if (artifact_root / "knowledge_base").is_dir() else []:
        migrated.append(_source(path, repo))
    paper_protocol = artifact_root / "paper_evaluation" / "paper_evaluation_protocol.json"
    if paper_protocol.is_file():
        _add_paper_protocol(builder, paper_protocol, repo)
        migrated.append(_source(paper_protocol, repo))
    _add_published_prior(builder, artifact_root / "published_prior", repo)
    for path in sorted((artifact_root / "published_prior").iterdir()) if (artifact_root / "published_prior").is_dir() else []:
        if path.is_file():
            migrated.append(_source(path, repo))
    migrated.extend(_add_operational_memory(builder, artifact_root.parent, repo))
    if include_execution_recipes:
        _add_plans(builder, artifact_root / "plans", repo)
        for path in sorted((artifact_root / "plans").glob("*.json")) if (artifact_root / "plans").is_dir() else []:
            migrated.append(_source(path, repo))
    _add_episodes(builder, episode_paths, repo)
    for path in episode_paths:
        if path.is_file():
            migrated.append(_source(path, repo))

    graph = builder.to_dict()
    manifest = {
        "schema_version": "ml-scientist-memory-migration-v1",
        "canonical_target": "knowledge_graph.json",
        "source_file_count": len({row["path"] for row in migrated}),
        "sources": sorted({row["path"]: row for row in migrated}.values(), key=lambda row: row["path"]),
        "legacy_registry_retained_as_compatibility_view": True,
        "execution_graphs_kept_separate": not include_execution_recipes,
        "migration_does_not_promote_maturity": True,
    }
    return graph, manifest


def validate_knowledge_graph(graph: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_ids = [node.get("node_id") for node in nodes]
    edge_ids = [edge.get("edge_id") for edge in edges]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node IDs")
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge IDs")
    known = set(node_ids)
    for node in nodes:
        if node.get("node_type") not in NODE_TYPES:
            errors.append(f"unknown node type: {node.get('node_type')}")
        if not node.get("provenance"):
            errors.append(f"missing provenance: {node.get('node_id')}")
        if node.get("node_type") == "Policy":
            attributes = node.get("attributes", {})
            if not attributes.get("policy_role"):
                errors.append(f"policy missing role: {node.get('node_id')}")
            if not attributes.get("eligible_stages"):
                errors.append(f"policy missing stage eligibility: {node.get('node_id')}")
            if attributes.get("credit_assignment") != "explicit_policy_trace_only":
                errors.append(f"policy has unsafe credit assignment: {node.get('node_id')}")
    for edge in edges:
        if edge.get("relation") not in RELATION_TYPES:
            errors.append(f"unknown relation: {edge.get('relation')}")
        if edge.get("source") not in known or edge.get("target") not in known:
            errors.append(f"dangling edge: {edge.get('edge_id')}")
    protected = [node for node in nodes if node.get("node_type") == "Metric" and node.get("attributes", {}).get("protected")]
    for node in protected:
        if node.get("attributes", {}).get("observability") != "PROTECTED_FINAL_ONLY":
            errors.append(f"protected metric visibility violation: {node['node_id']}")
    return {
        "schema_version": "ml-scientist-knowledge-graph-validation-v1",
        "passed": not errors,
        "errors": errors,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_type_counts": {node_type: sum(node.get("node_type") == node_type for node in nodes) for node_type in sorted(NODE_TYPES)},
        "protected_metric_count": len(protected),
    }


def query_knowledge_graph(
    graph: dict[str, Any],
    query: str,
    *,
    node_types: set[str] | None = None,
    stage: str | None = None,
    limit: int = 12,
) -> list[dict[str, Any]]:
    query_tokens = _tokens(query)
    nodes = {node["node_id"]: node for node in graph.get("nodes", [])}
    scores: dict[str, float] = {}
    for node in nodes.values():
        attributes = node.get("attributes", {})
        text = " ".join([node.get("node_id", ""), node.get("label", ""), json.dumps(attributes, ensure_ascii=False, sort_keys=True)])
        overlap = len(query_tokens & _tokens(text))
        maturity_bonus = {"repeated_success": 3.0, "provisional_success": 2.0, "verified": 1.5, "observation": 0.0, "contradiction": -4.0}.get(node.get("maturity"), 0.0)
        if overlap or not query_tokens:
            scores[node["node_id"]] = overlap * 2.0 + maturity_bonus
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    for edge in graph.get("edges", []):
        if edge.get("source") in adjacency and edge.get("target") in adjacency:
            adjacency[edge["source"]].add(edge["target"])
            adjacency[edge["target"]].add(edge["source"])
    frontier = sorted(scores.items(), key=lambda row: (-row[1], row[0]))[:24]
    seen_depth = {node_id: 0 for node_id, _ in frontier}
    for depth in (1, 2):
        next_frontier: list[tuple[str, float]] = []
        for source_id, source_score in frontier:
            for target_id in sorted(adjacency.get(source_id, ())):
                if seen_depth.get(target_id, 99) <= depth:
                    continue
                propagated = source_score * (0.45 if depth == 1 else 0.20)
                scores[target_id] = max(scores.get(target_id, float("-inf")), propagated)
                seen_depth[target_id] = depth
                next_frontier.append((target_id, propagated))
        frontier = next_frontier
    scored: list[tuple[float, str, dict[str, Any]]] = []
    for node_id, score in scores.items():
        node = nodes[node_id]
        if node_types and node.get("node_type") not in node_types:
            continue
        attributes = node.get("attributes", {})
        if node.get("node_type") == "Skill" and not attributes.get("active"):
            continue
        eligible = attributes.get("eligible_stages") or attributes.get("stages") or []
        if stage and eligible and stage not in eligible and "all_runtime_nodes" not in eligible:
            continue
        if stage and attributes.get("protected") and stage not in {"protected_final_test", "confirmatory_analysis", "paper_writing", "paper_review"}:
            continue
        scored.append((score, node_id, node))
    scored.sort(key=lambda row: (-row[0], row[1]))
    return [{"score": score, **node} for score, _, node in scored[:limit]]


def write_knowledge_graph(
    graph: dict[str, Any],
    migration_manifest: dict[str, Any],
    out_dir: Path,
    *,
    governance_dir: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = out_dir / "knowledge_graph.json"
    graph_path.write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    validation = validate_knowledge_graph(graph)
    (out_dir / "validation_report.json").write_text(json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "migration_manifest.json").write_text(json.dumps(migration_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    ontology = {
        "schema_version": "ml-scientist-knowledge-graph-ontology-v1",
        "node_types": sorted(NODE_TYPES),
        "relation_types": sorted(RELATION_TYPES),
        "views": {
            "benchmark_metric": ["Benchmark", "Task", "Metric", "Constraint", "Dataset"],
            "strategy_skill": ["Agent", "Strategy", "Policy", "Skill"],
            "episode_evidence": ["Episode", "Decision", "ResearchContext", "EvidenceArtifact", "Failure"],
            "claim_paper": ["Claim", "PaperCriterion"],
        },
        "execution_graph_is_separate": True,
    }
    (out_dir / "ontology.json").write_text(json.dumps(ontology, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    snapshot = {
        "schema_version": "ml-scientist-graph-snapshot-v1",
        "created_at": _now(),
        "graph_path": graph_path.name,
        "graph_sha256": _sha_file(graph_path),
        "graph_content_sha256": graph["content_sha256"],
        "pending_writes_visible": False,
        "mid_arm_mutation_allowed": False,
        "protected_final_metrics_retrievable_by_search": False,
    }
    (out_dir / "retrieval_snapshot.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if governance_dir is not None:
        memory_path = governance_dir / "memory_registry.json"
        memory = _safe_json(memory_path) if memory_path.is_file() else {}
        if not isinstance(memory, dict):
            memory = {}
        memory.update(
            {
                "schema_version": "fml-scientist-memory-registry-v2-graph-view",
                "canonical_store": str(graph_path.resolve()),
                "canonical_store_status": "ACTIVE_VALIDATED" if validation["passed"] else "INVALID",
                "legacy_compatibility_view": True,
                "legacy_entries_migrated": True,
                "mid_arm_mutation_allowed": False,
            }
        )
        memory.pop("canonical_store_sha256", None)
        memory_path.write_text(json.dumps(memory, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {**validation, "graph_sha256": snapshot["graph_sha256"], "snapshot": snapshot}


def append_episode(path: Path, episode: dict[str, Any]) -> dict[str, Any]:
    """Append post-run experience; callers must never use the new row in the current arm."""
    if episode.get("protected_metric_used_for_routing"):
        raise ValueError("Protected final metrics cannot be used for routing")
    if episode.get("protected_metric_used_for_skill_evolution"):
        raise ValueError("Protected final metrics cannot be used for search-skill evolution")
    required = {"episode_id", "run_id", "node_id", "visible_evidence", "outcome", "snapshot_sha256"}
    missing = sorted(required - set(episode))
    if missing:
        raise ValueError(f"Episode missing required fields: {missing}")
    if not episode.get("selected_strategy_id") and not episode.get("selected_policy_ids"):
        raise ValueError("Episode requires a selected strategy or at least one selected policy")
    if episode.get("schema_version") == "ml-scientist-adaptive-episode-v2":
        transition_fields = {"research_context", "state_before", "action", "next_state", "resource_telemetry"}
        transition_missing = sorted(transition_fields - set(episode))
        if transition_missing:
            raise ValueError(f"Enriched episode missing transition fields: {transition_missing}")
    row = dict(episode)
    row.setdefault("maturity", "observation")
    row.setdefault("recorded_at", _now())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return row
