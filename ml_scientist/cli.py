"""Command-line entrypoint for catalog, planning, and paper-data artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .catalog import build_catalog
from .campaign import run_campaign
from .experiment_design import preflight_environment, write_experiment_protocol
from .governance import initialize_governance_artifacts
from .handbook import write_handbook, write_provisional_paper
from .knowledge_base import write_knowledge_base
from .planner import build_research_paper_plan
from .paper_evaluation import evaluate_paper_evaluation, write_paper_evaluation_protocol
from .paper_package import write_paper_package
from .published_prior import write_published_prior
from .reporting import write_catalog_artifacts, write_experiment_artifacts


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
    report = sub.add_parser("report", help="Ingest real run summaries into paper tables")
    report.add_argument("--results", type=Path, required=True)
    report.add_argument("--metric-reports", type=Path)
    report.add_argument("--out", type=Path, required=True)
    protocol = sub.add_parser("protocol", help="Freeze a controlled baseline experiment matrix")
    protocol.add_argument("--out", type=Path, required=True)
    protocol.add_argument("--model", default="SET_MODEL")
    protocol.add_argument("--provider", default="OpenAI")
    protocol.add_argument("--results", default="benchmark_results/controlled")
    protocol.add_argument("--full-extension", action="store_true")
    protocol.add_argument("--eval-backend", choices=["local", "ssh"], default="local")
    protocol.add_argument("--ssh-host", default="ubuntu-heshi")
    protocol.add_argument("--remote-project-root", default="/media/heshi/game/fml-scientist/repo")
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
    campaign = sub.add_parser("campaign", help="Execute or dry-run a frozen, resumable run matrix")
    campaign.add_argument("--matrix", type=Path, required=True)
    campaign.add_argument("--logs", type=Path, required=True)
    campaign.add_argument("--phase", action="append", default=[])
    campaign.add_argument("--max-runs", type=int)
    campaign.add_argument("--dry-run", action="store_true")
    bootstrap = sub.add_parser("bootstrap", help="Learn catalog, plan all tasks, and initialize paper data")
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument("--results", type=Path, default=Path("benchmark_results"))
    bootstrap.add_argument("--metric-reports", type=Path, default=Path("metric_reports"))
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
        selected = catalog["tasks"] if args.task == "all" else [task for task in catalog["tasks"] if task["task_id"] == args.task]
        if not selected:
            raise SystemExit(f"Unknown task: {args.task}")
        for task in selected:
            _write_json(args.out / f"{task['task_id']}.json", build_research_paper_plan(task, catalog["agents"]))
    elif args.command == "report":
        write_experiment_artifacts(args.results, args.out, catalog, args.metric_reports)
    elif args.command == "protocol":
        write_experiment_protocol(
            catalog,
            args.out,
            model=args.model,
            provider=args.provider,
            output_dir=args.results,
            include_full_extension=args.full_extension,
            eval_backend=args.eval_backend,
            ssh_host=args.ssh_host,
            remote_project_root=args.remote_project_root,
        )
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
    else:
        write_catalog_artifacts(catalog, args.out / "catalog")
        write_handbook(catalog, args.out / "knowledge")
        write_knowledge_base(repo, catalog, args.out / "knowledge_base")
        write_published_prior(args.out / "published_prior")
        write_paper_evaluation_protocol(args.out / "paper_evaluation")
        write_provisional_paper(catalog, args.out / "paper")
        initialize_governance_artifacts(catalog, args.out / "governance")
        write_experiment_protocol(
            catalog,
            args.out / "protocol",
            model=args.model,
            provider=args.provider,
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
        for task in catalog["tasks"]:
            _write_json(args.out / "plans" / f"{task['task_id']}.json", build_research_paper_plan(task, catalog["agents"]))
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
