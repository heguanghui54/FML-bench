"""Command-line entrypoint for catalog, planning, and paper-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .architecture_report import write_architecture_checkpoint_report
from .adaptive_runtime import AdaptiveResearchRuntime
from .benchmark_learning import build_fml_benchmark_contract, learn_manifest, write_benchmark_contract
from .catalog import build_catalog
from .campaign import run_campaign
from .experiment_design import (
    preflight_environment,
    write_experiment_protocol,
    write_resource_matched_sensitivity_protocol,
    write_heldout_transfer_protocol,
)
from .fml_episode_bridge import import_completed_summaries
from .governance import EvidenceGatedSkillRegistry, initialize_governance_artifacts
from .handbook import write_handbook, write_provisional_paper
from .knowledge_base import write_knowledge_base
from .knowledge_graph import append_episode, build_project_knowledge_graph, query_knowledge_graph, validate_knowledge_graph, write_knowledge_graph
from .live_pipeline_report import write_live_pipeline_report
from .planner import build_research_paper_plan
from .paper_evaluation import evaluate_paper_evaluation, write_paper_evaluation_protocol
from .paper_package import write_paper_package
from .published_prior import write_published_prior
from .reporting import write_catalog_artifacts, write_experiment_artifacts
from .skill_evolution import SkillEvolutionEngine


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Self-evolving FML research-to-paper tooling")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    learn = sub.add_parser("learn", help="Catalog every registered agent, task, and metric")
    learn.add_argument("--out", type=Path, required=True)
    plan = sub.add_parser("plan", help="Build Stage-3 unified research-and-paper graphs")
    plan.add_argument("--task", default="all")
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--knowledge-graph", type=Path)
    plan.add_argument("--request", default="")
    plan.add_argument("--candidate-steps", type=int, default=8)
    plan.add_argument("--token-budget", type=int, default=0)
    plan.add_argument("--wall-clock-hours", type=float, default=0)
    report = sub.add_parser("report", help="Ingest real run summaries into paper tables")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--metric-reports", type=Path)
    report.add_argument("--out", type=Path, required=True)
    protocol = sub.add_parser("protocol", help="Freeze a controlled baseline experiment matrix")
    protocol.add_argument("--out", type=Path, required=True)
    protocol.add_argument("--model", default="SET_MODEL")
    protocol.add_argument("--provider", default="OpenAI")
    protocol.add_argument("--results", default="benchmark_results/controlled")
    protocol.add_argument("--pilot-steps", type=int, default=1)
    protocol.add_argument("--confirmatory-steps", type=int, default=3)
    protocol.add_argument("--full-extension", action="store_true")
    protocol.add_argument("--eval-backend", choices=["local", "ssh"], default="local")
    protocol.add_argument("--ssh-host", default="ubuntu-heshi")
    protocol.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
    sensitivity = sub.add_parser(
        "sensitivity-protocol",
        help="Write the separately labelled matched-budget baseline-plus-adaptive matrix",
    )
    sensitivity.add_argument("--out", type=Path, required=True)
    sensitivity.add_argument("--model", default="gpt-5.6-sol")
    sensitivity.add_argument("--provider", default="CodexCLI")
    sensitivity.add_argument("--results", default="benchmark_results/resource_matched_sensitivity")
    sensitivity.add_argument("--eval-backend", choices=["local", "ssh"], default="ssh")
    sensitivity.add_argument("--ssh-host", default="ubuntu-heshi")
    sensitivity.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
    heldout = sub.add_parser(
        "heldout-transfer-protocol",
        help="Freeze the paper-eligible held-out paired-task comparison",
    )
    heldout.add_argument("--out", type=Path, required=True)
    heldout.add_argument("--model", default="gpt-5.6-sol")
    heldout.add_argument("--provider", default="CodexCLI")
    heldout.add_argument("--results", default="benchmark_results/heldout_transfer_confirmatory")
    heldout.add_argument("--eval-backend", choices=["local", "ssh"], default="ssh")
    heldout.add_argument("--ssh-host", default="ubuntu-heshi")
    heldout.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
    preflight = sub.add_parser("preflight", help="Check live-run prerequisites without exposing secrets")
    preflight.add_argument("--out", type=Path, required=True)
    preflight.add_argument("--model", default="SET_MODEL")
    preflight.add_argument("--provider", default="OpenAI")
    preflight.add_argument("--eval-backend", choices=["local", "ssh"], default="local")
    preflight.add_argument("--ssh-host", default="ubuntu-heshi")
    preflight.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
    governance = sub.add_parser("governance-init", help="Initialize frozen memory and evidence-gated skill registries")
    governance.add_argument("--out", type=Path, required=True)
    published = sub.add_parser("published-prior", help="Write source-labeled published FML tables and charts")
    published.add_argument("--out", type=Path, required=True)
    paper_eval = sub.add_parser("paper-evaluation", help="Write the manuscript-quality evaluation protocol")
    paper_eval.add_argument("--out", type=Path, required=True)
    paper_eval_report = sub.add_parser("paper-evaluation-report", help="Reduce completed blinded manuscript-review records")
    paper_eval_report.add_argument("--data", type=Path, required=True)
    paper_eval_report.add_argument("--out", type=Path, required=True)
    paper_package = sub.add_parser("paper-package", help="Build an evidence-locked manuscript handoff from generated artifacts")
    paper_package.add_argument("--artifact-root", type=Path, required=True)
    paper_package.add_argument("--out", type=Path, required=True)
    knowledge = sub.add_parser("knowledge-base", help="Write source-grounded agent and task dossiers")
    knowledge.add_argument("--out", type=Path, required=True)
    benchmark_learn = sub.add_parser("benchmark-learn", help="Learn and validate a benchmark/metric contract")
    benchmark_learn.add_argument("--source", type=Path)
    benchmark_learn.add_argument("--activate", action="store_true")
    benchmark_learn.add_argument("--out", type=Path, required=True)
    graph_build = sub.add_parser("memory-build-graph", help="Migrate current learned memory into the canonical knowledge graph")
    graph_build.add_argument("--artifact-root", type=Path, required=True)
    graph_build.add_argument("--out", type=Path, required=True)
    graph_build.add_argument("--episodes", type=Path, action="append", default=[])
    graph_validate = sub.add_parser("memory-validate-graph", help="Validate a knowledge graph")
    graph_validate.add_argument("--graph", type=Path, required=True)
    graph_query = sub.add_parser("memory-query-graph", help="Query typed project memory")
    graph_query.add_argument("--graph", type=Path, required=True)
    graph_query.add_argument("--query", required=True)
    graph_query.add_argument("--type", action="append", default=[])
    graph_query.add_argument("--stage")
    graph_query.add_argument("--limit", type=int, default=12)
    episode_append = sub.add_parser("memory-append-episode", help="Append a completed post-run episode for the next graph snapshot")
    episode_append.add_argument("--episodes", type=Path, required=True)
    episode_append.add_argument("--input", type=Path, required=True)
    skill_evolve = sub.add_parser("skill-evolve", help="Run post-arm Read-Write-Assess-Govern skill evolution")
    skill_evolve.add_argument("--episodes", type=Path, required=True)
    skill_evolve.add_argument("--governance-root", type=Path, required=True)
    skill_evolve.add_argument("--out", type=Path, required=True)
    architecture_report = sub.add_parser("architecture-report", help="Write a self-contained HTML architecture checkpoint")
    architecture_report.add_argument("--artifact-root", type=Path, required=True)
    architecture_report.add_argument("--out", type=Path, required=True)
    manifest_create = sub.add_parser(
        "adaptive-runtime-manifest",
        help="Create a hash-bound execution manifest; --authorize enables execution",
    )
    manifest_create.add_argument("--plan", type=Path, required=True)
    manifest_create.add_argument("--graph", type=Path, required=True)
    manifest_create.add_argument("--run-id", required=True)
    manifest_create.add_argument("--result-root", type=Path, required=True)
    manifest_create.add_argument("--workspace", type=Path, required=True)
    manifest_create.add_argument("--out", type=Path, required=True)
    manifest_create.add_argument("--backend", choices=["local", "ssh"], default="local")
    manifest_create.add_argument("--model", default="gpt-5")
    manifest_create.add_argument("--provider", default="CodexCLI")
    manifest_create.add_argument("--node-commands", type=Path)
    manifest_create.add_argument("--target-file", action="append", default=[])
    manifest_create.add_argument("--authorize", action="store_true")
    for name, help_text in (
        ("adaptive-runtime-init", "Initialize a graph-bound Stage 4-6 runtime without launching experiments"),
        ("adaptive-runtime-status", "Show ready nodes in an adaptive runtime"),
        ("adaptive-runtime-decide", "Select and record a node strategy from visible evidence"),
        ("adaptive-runtime-record", "Record an executor outcome and update node readiness"),
        ("adaptive-runtime-amend", "Route a paper-review issue to an upstream amendment target"),
        ("adaptive-runtime-export", "Export one enriched completed transition episode"),
        ("adaptive-runtime-evolve", "Execute the dependency-ready Stage-7 skill lifecycle"),
        ("adaptive-runtime-run", "Execute dependency-ready DAG nodes from an authorized frozen manifest"),
    ):
        runtime = sub.add_parser(name, help=help_text)
        runtime.add_argument("--plan", type=Path, required=True)
        runtime.add_argument("--graph", type=Path, required=True)
        runtime.add_argument("--run-id", required=True)
        runtime.add_argument("--state", type=Path, required=True)
        if name == "adaptive-runtime-decide":
            runtime.add_argument("--node", required=True)
            runtime.add_argument("--evidence", type=Path)
            runtime.add_argument("--remaining-budget-fraction", type=float, default=1.0)
        elif name == "adaptive-runtime-record":
            runtime.add_argument("--node", required=True)
            runtime.add_argument("--outcome", required=True)
            runtime.add_argument("--evidence-artifacts", type=Path)
            runtime.add_argument("--policy-evaluations", type=Path)
            runtime.add_argument("--budget-remaining", action=argparse.BooleanOptionalAction, default=True)
            runtime.add_argument("--budget-after", type=Path)
            runtime.add_argument("--telemetry", type=Path)
            runtime.add_argument("--state-observations", type=Path)
            runtime.add_argument("--skill-validation-evaluations", type=Path)
        elif name == "adaptive-runtime-amend":
            runtime.add_argument("--issue-id", required=True)
            runtime.add_argument("--issue-type", required=True)
            runtime.add_argument("--source-paper-node", default="paper-peer-review")
            runtime.add_argument("--protected-test-exposed", action="store_true")
        elif name == "adaptive-runtime-export":
            runtime.add_argument("--node", required=True)
            runtime.add_argument("--episodes", type=Path)
        elif name == "adaptive-runtime-evolve":
            runtime.add_argument("--episodes", type=Path, required=True)
            runtime.add_argument("--governance-root", type=Path, required=True)
            runtime.add_argument("--out", type=Path, required=True)
        elif name == "adaptive-runtime-run":
            runtime.add_argument("--execution-manifest", type=Path, required=True)
            runtime.add_argument("--max-nodes", type=int)
    campaign = sub.add_parser("campaign", help="Execute or dry-run a frozen, resumable run matrix")
    campaign.add_argument("--matrix", type=Path, required=True)
    campaign.add_argument("--logs", type=Path, required=True)
    campaign.add_argument("--phase", action="append", default=[])
    campaign.add_argument("--max-runs", type=int)
    campaign.add_argument("--dry-run", action="store_true")
    live_report = sub.add_parser("live-pipeline-report", help="Refresh the live baseline-versus-adaptive HTML dashboard")
    live_report.add_argument("--results", type=Path, required=True)
    live_report.add_argument("--campaign-dir", type=Path, required=True)
    live_report.add_argument("--matrix", type=Path, required=True)
    live_report.add_argument("--graph", type=Path, required=True)
    live_report.add_argument("--issues", type=Path, required=True)
    live_report.add_argument("--new-pipeline-results", type=Path)
    live_report.add_argument("--episodes", type=Path)
    live_report.add_argument("--out", type=Path, required=True)
    import_episodes = sub.add_parser("fml-import-episodes", help="Import completed FML summaries as non-crediting observations")
    import_episodes.add_argument("--results", type=Path, required=True)
    import_episodes.add_argument("--episodes", type=Path, required=True)
    import_episodes.add_argument("--graph", type=Path, required=True)
    bootstrap = sub.add_parser("bootstrap", help="Learn catalog, plan all tasks, and initialize paper data")
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument("--results", type=Path, default=Path("benchmark_results"))
    bootstrap.add_argument("--metric-reports", type=Path, default=Path("metric_reports"))
    bootstrap.add_argument("--pilot-steps", type=int, default=1)
    bootstrap.add_argument("--confirmatory-steps", type=int, default=3)
    bootstrap.add_argument("--model", default="SET_MODEL")
    bootstrap.add_argument("--provider", default="OpenAI")
    bootstrap.add_argument("--eval-backend", choices=["local", "ssh"], default="local")
    bootstrap.add_argument("--ssh-host", default="ubuntu-heshi")
    bootstrap.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo = args.repo.resolve()
    catalog = build_catalog(repo)
    if args.command == "learn":
        write_catalog_artifacts(catalog, args.out)
    elif args.command == "plan":
        graph = json.loads(args.knowledge_graph.read_text(encoding="utf-8")) if args.knowledge_graph else None
        selected = catalog["tasks"] if args.task == "all" else [task for task in catalog["tasks"] if task["task_id"] == args.task]
        if not selected:
            raise SystemExit(f"Unknown task: {args.task}")
        for task in selected:
            _write_json(
                args.out / f"{task['task_id']}.json",
                build_research_paper_plan(
                    task,
                    catalog["agents"],
                    knowledge_graph=graph,
                    research_request=args.request,
                    budget={
                        "candidate_steps": args.candidate_steps,
                        "token_budget": args.token_budget,
                        "wall_clock_hours": args.wall_clock_hours,
                    },
                ),
            )
    elif args.command == "report":
        write_experiment_artifacts(args.results, args.out, catalog, args.metric_reports)
    elif args.command == "protocol":
        write_experiment_protocol(
            catalog,
            args.out,
            model=args.model,
            provider=args.provider,
            output_dir=args.results,
            pilot_steps=args.pilot_steps,
            confirmatory_steps=args.confirmatory_steps,
            include_full_extension=args.full_extension,
            eval_backend=args.eval_backend,
            ssh_host=args.ssh_host,
            remote_project_root=args.remote_project_root,
        )
    elif args.command == "sensitivity-protocol":
        print(json.dumps(write_resource_matched_sensitivity_protocol(
            catalog,
            args.out,
            model=args.model,
            provider=args.provider,
            output_dir=args.results,
            eval_backend=args.eval_backend,
            ssh_host=args.ssh_host,
            remote_project_root=args.remote_project_root,
        ), indent=2, ensure_ascii=False))
    elif args.command == "heldout-transfer-protocol":
        print(json.dumps(write_heldout_transfer_protocol(
            catalog,
            args.out,
            model=args.model,
            provider=args.provider,
            output_dir=args.results,
            eval_backend=args.eval_backend,
            ssh_host=args.ssh_host,
            remote_project_root=args.remote_project_root,
        ), indent=2, ensure_ascii=False))
    elif args.command == "preflight":
        _write_json(
            args.out / "preflight.json",
            preflight_environment(
                catalog, provider=args.provider, model=args.model, repo=repo,
                eval_backend=args.eval_backend, ssh_host=args.ssh_host,
                remote_project_root=args.remote_project_root,
            ),
        )
    elif args.command == "governance-init":
        initialize_governance_artifacts(catalog, args.out)
    elif args.command == "published-prior":
        write_published_prior(args.out)
    elif args.command == "paper-evaluation":
        write_paper_evaluation_protocol(args.out)
    elif args.command == "paper-evaluation-report":
        evaluate_paper_evaluation(args.data, args.out)
    elif args.command == "paper-package":
        artifact_root = args.artifact_root
        write_paper_package(
            catalog,
            args.out,
            catalog_dir=artifact_root / "catalog",
            knowledge_base_dir=artifact_root / "knowledge_base",
            plans_dir=artifact_root / "plans",
            published_prior_dir=artifact_root / "published_prior",
            protocol_dir=artifact_root / "protocol",
            experiments_dir=artifact_root / "experiments",
            paper_evaluation_dir=artifact_root / "paper_evaluation",
        )
    elif args.command == "knowledge-base":
        write_knowledge_base(repo, catalog, args.out)
    elif args.command == "benchmark-learn":
        contract = learn_manifest(args.source, activate=args.activate) if args.source else build_fml_benchmark_contract(catalog)
        print(json.dumps(write_benchmark_contract(contract, args.out), indent=2, ensure_ascii=False))
    elif args.command == "memory-build-graph":
        graph, migration = build_project_knowledge_graph(repo, catalog, args.artifact_root, episode_paths=args.episodes)
        print(json.dumps(write_knowledge_graph(graph, migration, args.out, governance_dir=args.artifact_root / "governance"), indent=2, ensure_ascii=False))
    elif args.command == "memory-validate-graph":
        graph = json.loads(args.graph.read_text(encoding="utf-8"))
        report = validate_knowledge_graph(graph)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        if not report["passed"]:
            raise SystemExit(2)
    elif args.command == "memory-query-graph":
        graph = json.loads(args.graph.read_text(encoding="utf-8"))
        print(json.dumps(query_knowledge_graph(graph, args.query, node_types=set(args.type) or None, stage=args.stage, limit=args.limit), indent=2, ensure_ascii=False))
    elif args.command == "memory-append-episode":
        episode = json.loads(args.input.read_text(encoding="utf-8"))
        print(json.dumps(append_episode(args.episodes, episode), indent=2, ensure_ascii=False))
    elif args.command == "skill-evolve":
        registry = EvidenceGatedSkillRegistry(args.governance_root, catalog["repository_commit"])
        report = SkillEvolutionEngine(registry=registry, out_dir=args.out).evolve(args.episodes)
        print(json.dumps(report, indent=2, ensure_ascii=False))
    elif args.command == "architecture-report":
        print(json.dumps(write_architecture_checkpoint_report(
            graph_path=args.artifact_root / "knowledge_graph" / "knowledge_graph.json",
            benchmark_contract_path=args.artifact_root / "benchmark_contracts" / "fml-bench.json",
            plans_dir=args.artifact_root / "plans",
            out_path=args.out,
        ), indent=2, ensure_ascii=False))
    elif args.command == "adaptive-runtime-manifest":
        from .node_execution import ExecutionManifest

        plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
        graph_payload = json.loads(args.graph.read_text(encoding="utf-8"))
        node_commands = (
            json.loads(args.node_commands.read_text(encoding="utf-8"))
            if args.node_commands else {}
        )
        manifest = ExecutionManifest.create(
            run_id=args.run_id,
            plan=plan_payload,
            graph=graph_payload,
            result_root=args.result_root,
            workspace=args.workspace,
            authorized=args.authorize,
            backend=args.backend,
            model=args.model,
            provider=args.provider,
            node_commands=node_commands,
            target_files=tuple(args.target_file),
            task_description=plan_payload.get("research_request", ""),
        )
        _write_json(args.out, manifest.to_dict())
        print(json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False))
    elif args.command.startswith("adaptive-runtime-"):
        plan_payload = json.loads(args.plan.read_text(encoding="utf-8"))
        graph_payload = json.loads(args.graph.read_text(encoding="utf-8"))
        runtime = AdaptiveResearchRuntime(
            plan=plan_payload, graph=graph_payload, run_id=args.run_id, state_path=args.state,
        )
        if args.command == "adaptive-runtime-decide":
            evidence = json.loads(args.evidence.read_text(encoding="utf-8")) if args.evidence else []
            result = runtime.choose_operator(
                args.node, evidence=evidence,
                budget_before={"remaining_fraction": args.remaining_budget_fraction},
            )
        elif args.command == "adaptive-runtime-record":
            artifacts = json.loads(args.evidence_artifacts.read_text(encoding="utf-8")) if args.evidence_artifacts else []
            policy_evaluations = json.loads(args.policy_evaluations.read_text(encoding="utf-8")) if args.policy_evaluations else None
            budget_after = json.loads(args.budget_after.read_text(encoding="utf-8")) if args.budget_after else None
            telemetry = json.loads(args.telemetry.read_text(encoding="utf-8")) if args.telemetry else None
            state_observations = json.loads(args.state_observations.read_text(encoding="utf-8")) if args.state_observations else None
            skill_validations = json.loads(args.skill_validation_evaluations.read_text(encoding="utf-8")) if args.skill_validation_evaluations else None
            result = runtime.record_outcome(
                args.node, outcome=args.outcome, evidence_artifacts=artifacts,
                budget_remaining=args.budget_remaining,
                policy_evaluations=policy_evaluations,
                budget_after=budget_after,
                telemetry=telemetry,
                state_observations=state_observations,
                skill_validation_evaluations=skill_validations,
            )
        elif args.command == "adaptive-runtime-amend":
            result = runtime.apply_review_issue(
                issue_id=args.issue_id, issue_type=args.issue_type,
                source_paper_node=args.source_paper_node,
                protected_test_exposed=args.protected_test_exposed,
            )
        elif args.command == "adaptive-runtime-export":
            result = runtime.export_episode(args.node)
            if args.episodes:
                result = append_episode(args.episodes, result)
        elif args.command == "adaptive-runtime-evolve":
            result = runtime.run_skill_evolution(
                episodes_path=args.episodes,
                governance_root=args.governance_root,
                out_dir=args.out,
            )
        elif args.command == "adaptive-runtime-run":
            from .node_execution import ExecutionManifest, FullDagRunner

            manifest = ExecutionManifest.load(
                args.execution_manifest,
                plan=plan_payload,
                graph=graph_payload,
            )
            if manifest.run_id != args.run_id:
                raise ValueError("Execution manifest run ID does not match --run-id")
            result = FullDagRunner(runtime=runtime, manifest=manifest).run(max_nodes=args.max_nodes)
        else:
            result = {"state": runtime.state, "ready_nodes": runtime.ready_nodes()}
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.command == "campaign":
        state = run_campaign(
            matrix_path=args.matrix,
            repo=repo,
            log_dir=args.logs,
            phases=set(args.phase) or None,
            max_runs=args.max_runs,
            dry_run=args.dry_run,
        )
        print(json.dumps(state, indent=2, ensure_ascii=False))
    elif args.command == "live-pipeline-report":
        print(json.dumps(write_live_pipeline_report(
            results_root=args.results,
            campaign_dir=args.campaign_dir,
            matrix_path=args.matrix,
            graph_path=args.graph,
            issue_registry_path=args.issues,
            out_dir=args.out,
            catalog=catalog,
            new_pipeline_results_root=args.new_pipeline_results,
            episode_buffer_path=args.episodes,
        ), indent=2, ensure_ascii=False))
    elif args.command == "fml-import-episodes":
        print(json.dumps(import_completed_summaries(
            results_root=args.results,
            episodes_path=args.episodes,
            graph_path=args.graph,
            catalog=catalog,
        ), indent=2, ensure_ascii=False))
    else:
        write_catalog_artifacts(catalog, args.out / "catalog")
        write_handbook(catalog, args.out / "knowledge")
        write_knowledge_base(repo, catalog, args.out / "knowledge_base")
        write_published_prior(args.out / "published_prior")
        write_paper_evaluation_protocol(args.out / "paper_evaluation")
        write_provisional_paper(catalog, args.out / "paper")
        initialize_governance_artifacts(catalog, args.out / "governance")
        write_benchmark_contract(build_fml_benchmark_contract(catalog), args.out / "benchmark_contracts")
        write_experiment_protocol(
            catalog,
            args.out / "protocol",
            model=args.model,
            provider=args.provider,
            pilot_steps=args.pilot_steps,
            confirmatory_steps=args.confirmatory_steps,
            eval_backend=args.eval_backend,
            ssh_host=args.ssh_host,
            remote_project_root=args.remote_project_root,
        )
        _write_json(
            args.out / "protocol" / "preflight.json",
            preflight_environment(
                catalog, provider=args.provider, model=args.model, repo=repo,
                eval_backend=args.eval_backend, ssh_host=args.ssh_host,
                remote_project_root=args.remote_project_root,
            ),
        )
        graph, migration = build_project_knowledge_graph(repo, catalog, args.out)
        write_knowledge_graph(graph, migration, args.out / "knowledge_graph", governance_dir=args.out / "governance")
        for task in catalog["tasks"]:
            _write_json(
                args.out / "plans" / f"{task['task_id']}.json",
                build_research_paper_plan(task, catalog["agents"], knowledge_graph=graph),
            )
        write_architecture_checkpoint_report(
            graph_path=args.out / "knowledge_graph" / "knowledge_graph.json",
            benchmark_contract_path=args.out / "benchmark_contracts" / "fml-bench.json",
            plans_dir=args.out / "plans",
            out_path=args.out / "reports" / "adaptive_pipeline_checkpoint.html",
        )
        write_experiment_artifacts(args.results, args.out / "experiments", catalog, args.metric_reports)
        write_paper_package(
            catalog,
            args.out / "paper",
            catalog_dir=args.out / "catalog",
            knowledge_base_dir=args.out / "knowledge_base",
            plans_dir=args.out / "plans",
            published_prior_dir=args.out / "published_prior",
            protocol_dir=args.out / "protocol",
            experiments_dir=args.out / "experiments",
            paper_evaluation_dir=args.out / "paper_evaluation",
        )


if __name__ == "__main__":
    main()
