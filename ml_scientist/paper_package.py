"""Evidence-locked handoff from FML experiments to an academic manuscript.

The package deliberately writes substantive background and methods before the
campaign runs, while keeping numerical Results, conclusion claims, review, and
finalization locked until their exact provenance artifacts exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .handbook import TRANSFER
from .published_prior import PAPER


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _bundle_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    found = False
    for path in sorted(paths):
        if not path.is_file():
            continue
        found = True
        digest.update(path.name.encode("utf-8"))
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest() if found else ""


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _configuration(catalog: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "fml-scientist-paper-configuration-v1",
        "confirmation_status": "AWAITING_HUMAN_CONFIRMATION",
        "working_title": "From Search Dynamics to Evidence-Gated Papers: A Self-Evolving ML Research Pipeline",
        "paper_type": "empirical ML systems and research-agent study",
        "discipline": "machine learning systems, AI for scientific discovery, and agent evaluation",
        "target_venue": "UNSET_REQUIRES_HUMAN_DECISION",
        "citation_style": "UNSET_REQUIRES_VENUE_DECISION",
        "main_language": "English proposed; requires confirmation",
        "abstracts": "English plus independently written Chinese abstract proposed; locked until Results",
        "output_formats": ["Markdown evidence draft", "LaTeX and PDF only after final integrity pass"],
        "provisional_word_budget": 8000,
        "study_inventory": catalog["counts"],
        "required_human_decisions": [
            "confirm or revise working title",
            "choose target venue and citation style",
            "confirm main language and word budget",
            "confirm authors, CRediT roles, funding, and conflicts",
        ],
        "full_drafting_allowed": False,
    }


def _research_brief(catalog: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Research-question brief",
            "",
            "Status: **FROZEN QUESTIONS; EMPIRICAL ANSWERS LOCKED**",
            "",
            "## Study objective",
            "",
            "Determine how the search topology of an ML research agent affects protected-test improvement, research-process behavior, and the reliability of an evidence-gated transition from experiments to a reviewed paper.",
            "",
            "## Primary questions",
            "",
            "1. Under matched model, provider, task, seed, and step budgets, how do seven search strategies differ in mean FML normalized improvement on the fixed eight-task Lite suite?",
            "2. Which task properties and search-process trajectories explain heterogeneous agent effects rather than a single universal ranking?",
            "3. Does a regime-adaptive controller behave differently from fixed greedy, multi-idea, tree, MCTS, and evolutionary controllers across dense- and sparse-opportunity tasks?",
            "4. As a separate local extension, does evidence gating plus multi-role review reduce unsupported claims and integrity failures relative to one-shot paper writing?",
            "",
            "## Preregistered hypotheses and boundaries",
            "",
            "- **H1, task-dependent strategy effect:** agent contrasts vary materially by task; the main estimand is therefore a fixed-suite average accompanied by per-task heterogeneity.",
            "- **H2, process-performance association:** validation efficiency, reliability, exploration, and generalization diagnostics explain distinct aspects of final performance. Associations are reported with their definitional overlap and are not treated as causal effects.",
            "- **H3, adaptive-search interaction:** the published opportunity-density split motivates a new confirmatory interaction hypothesis, but the published split remains post-hoc context and cannot count as local confirmation.",
            "- **H4, paper-control benefit:** the full writing/review/amendment pipeline is compared with W0 and W1 under matched evidence and writing budgets. Its scores remain separate from all FML metrics.",
            "",
            "## Evidence design",
            "",
            f"The repository currently specifies {catalog['counts']['agents']} agents, {catalog['counts']['tasks']} tasks, {catalog['counts']['lite_tasks']} Lite tasks, and {catalog['counts']['process_metrics']} process metrics. The diagnostic pilot cannot change the confirmatory matrix. Protected-test outcomes cannot return to method search. Published aggregates guide hypotheses only and are never merged with new trial rows.",
            "",
            "## Falsification conditions",
            "",
            "H1 is weakened if paired task effects are homogeneous and intervals exclude practically relevant heterogeneity. H2 is weakened when process associations are unstable across tasks or disappear after transparent sensitivity checks. H3 is weakened if the preregistered interaction is absent or reverses. H4 is weakened if W2 does not improve hard-gate pass rates or creates greater review cost without a defensible quality gain.",
            "",
        ]
    )


def _claim_rows(paths: dict[str, Path], catalog: dict[str, Any], experiment_status: dict[str, Any]) -> list[dict[str, Any]]:
    catalog_path = paths["catalog"]
    prior_manifest = paths["published_prior"] / "provenance_manifest.json"
    plan_files = sorted(paths["plans"].glob("*.json")) if paths["plans"].is_dir() else []
    pair_path = paths["experiments"] / "paired_agent_comparisons.csv"
    task_path = paths["experiments"] / "paired_agent_task_effects.csv"
    process_path = paths["experiments"] / "process_metric_group_statistics.csv"
    sensitivity_path = paths["experiments"] / "scorer_semantics_sensitivity.csv"
    interaction_path = paths["experiments"] / "adaptive_opportunity_interactions.csv"
    paper_eval_path = paths["paper_evaluation"] / "paper_evaluation_summary.json"
    paper_eval_summary = _read_json(paper_eval_path)
    real_records = int(experiment_status.get("record_count") or 0)
    pair_rows = _csv_rows(pair_path)
    complete_confirmatory_pairs = sum(
        row.get("analysis_class") == "confirmatory" and row.get("complete_block_gate_passed", "").lower() == "true"
        for row in pair_rows
    )
    interaction_rows = _csv_rows(interaction_path)
    complete_confirmatory_interactions = sum(
        row.get("analysis_class") == "confirmatory" and row.get("complete_block_gate_passed", "").lower() == "true"
        for row in interaction_rows
    )

    def row(
        claim_id: str,
        claim_type: str,
        section: str,
        wording: str,
        boundary: str,
        evidence: Path,
        unlock_rule: str,
        status: str,
        artifact_hash: str | None = None,
    ) -> dict[str, Any]:
        return {
            "claim_id": claim_id,
            "claim_type": claim_type,
            "paper_section": section,
            "permissible_wording": wording,
            "prohibited_overclaim": boundary,
            "required_evidence": str(evidence.resolve()),
            "artifact_sha256": _sha256(evidence) if artifact_hash is None else artifact_hash,
            "unlock_rule": unlock_rule,
            "status": status,
        }

    static_ready = catalog_path.is_file()
    prior_ready = prior_manifest.is_file()
    return [
        row("K1", "repository_fact", "Method", f"The audited harness contains {catalog['counts']['agents']} registered baseline-agent strategies.", "Do not call the source taxonomy an empirical ranking.", catalog_path, "catalog exists and source/config hashes resolve", "READY_STATIC" if static_ready else "LOCKED_MISSING_CATALOG"),
        row("K2", "repository_fact", "Experimental protocol", f"The configured suite contains {catalog['counts']['tasks']} tasks, including {catalog['counts']['lite_tasks']} Lite tasks and {catalog['counts']['process_metrics']} process metrics.", "Do not imply coverage of all ML research domains.", catalog_path, "catalog exists and task contracts resolve", "READY_STATIC" if static_ready else "LOCKED_MISSING_CATALOG"),
        row("K3", "system_design", "Method", "Every research, analysis, writing, and review node is planned at Stage 3 and executes a typed inner loop.", "Do not describe the architecture as empirically superior before ablation.", paths["plans"], "all generated task plans validate", "READY_STATIC" if len(plan_files) == catalog['counts']['tasks'] else "LOCKED_INCOMPLETE_PLANS", artifact_hash=_bundle_sha256(plan_files)),
        row("P1", "published_context", "Related work", "The current FML study reports aggregate performance and process diagnostics for controlled research-agent strategies.", "Do not present published aggregate rows as observations from this campaign.", prior_manifest, "published-prior manifest is complete and explicitly separated", "READY_PUBLISHED_CONTEXT" if prior_ready else "LOCKED_MISSING_PRIOR"),
        row("E1", "new_empirical", "Results", "Agent strategies differ in mean normalized improvement on the fixed confirmatory suite.", "No superiority, significance, or equivalence claim without complete adjusted pairwise evidence.", pair_path, "21 confirmatory agent pairs pass complete seed-block gates", "READY_NEW_EVIDENCE" if complete_confirmatory_pairs == 21 else "LOCKED_NO_COMPLETE_CONFIRMATORY_COMPARISONS"),
        row("E2", "new_empirical", "Results", "Agent effects are heterogeneous across the preregistered tasks.", "Do not treat task-seed cells as independent replications or generalize to all ML problems.", task_path, "all 168 pair-task rows have three matched seeds", "READY_NEW_EVIDENCE" if len(_csv_rows(task_path)) == 168 else "LOCKED_NO_COMPLETE_TASK_EFFECTS"),
        row("E3", "new_empirical", "Process analysis", "Search strategies exhibit distinct exploration, reliability, efficiency, generalization, and cost profiles.", "Do not combine incompatible process units or imply causal mediation from correlation.", process_path, "replicated process rows cover every ingested agent-task-seed run", "READY_NEW_EVIDENCE" if real_records > 0 and int(experiment_status.get('process_metric_run_coverage_count') or 0) == real_records else "LOCKED_INCOMPLETE_PROCESS_COVERAGE"),
        row("E4", "new_empirical", "Robustness", "The main conclusions are robust to separately named scorer-semantics sensitivity analyses.", "Do not overwrite the official scorer or claim robustness when membership differences remain unrecomputed.", sensitivity_path, "sensitivity audit covers every real run and required recomputations are complete", "READY_NEW_EVIDENCE" if real_records > 0 and int(experiment_status.get('scorer_sensitivity_run_count') or 0) == real_records else "LOCKED_NO_SENSITIVITY_EVIDENCE"),
        row("E5", "new_empirical", "Ablations", "The regime-adaptive controller has a different relative advantage across the preregistered opportunity strata.", "Do not reuse the published post-hoc split as confirmatory proof or claim a directional benefit when the interaction does not support it.", interaction_path, "six AdaptiveSearch-versus-baseline interactions have complete trials and Holm-aware reporting", "READY_NEW_EVIDENCE" if complete_confirmatory_interactions == 6 else "LOCKED_NO_COMPLETE_INTERACTION_ANALYSIS"),
        row("W1", "paper_extension", "Paper-pipeline evaluation", "The full evidence-gated writing pipeline differs in manuscript integrity or quality relative to W0 and W1.", "Never average paper-quality scores with FML outcome or process metrics or assert improvement when the paired contrast does not support it.", paper_eval_path, "all writing arms, blinded reviews, hard gates, issue dispositions, and paired contrasts are complete", "READY_LOCAL_EXTENSION" if paper_eval_summary.get("status") == "COMPLETE_EVALUATION" else "LOCKED_NO_COMPLETE_MANUSCRIPT_EVALUATION"),
    ]


def _argument_rows() -> list[dict[str, Any]]:
    return [
        {"argument_id": "A1", "section": "Introduction", "claim": "Research-agent evaluation should distinguish search outcomes, search dynamics, and paper integrity.", "evidence_needed": "FML metric taxonomy plus separate paper-evaluation contract", "counterargument": "A single aggregate score is easier to compare.", "response": "Aggregation across incompatible constructs conceals failure modes and creates invalid tradeoffs.", "status": "READY_CONCEPTUAL"},
        {"argument_id": "A2", "section": "Method", "claim": "Search strategies can be represented as reusable operators inside one governed DAG without erasing their differences.", "evidence_needed": "seven source-grounded agent dossiers and Stage-3 plans", "counterargument": "A unified controller may reduce every method to the same generic loop.", "response": "Each operator retains its state, proposal, selection, memory, diversity, and debugging contract.", "status": "READY_STATIC"},
        {"argument_id": "A3", "section": "Experimental protocol", "claim": "Matched task-seed budgets and a one-way protected test isolate strategy differences more credibly than uncontrolled demonstrations.", "evidence_needed": "frozen run matrix, task commits, seeds, budgets, and test-boundary logs", "counterargument": "Residual stochastic and model-provider effects remain.", "response": "The estimand is conditional on the frozen model/provider and uncertainty is computed across matched seed blocks.", "status": "READY_METHODS_ONLY"},
        {"argument_id": "A4", "section": "Results", "claim": "Search strategy affects fixed-suite protected-test performance and task-specific behavior.", "evidence_needed": "complete paired confirmatory and task-effect tables", "counterargument": "Three seeds provide weak inferential resolution.", "response": "Lead with effects and intervals, disclose low resolution, and never equate non-significance with equivalence.", "status": "LOCKED_NO_DATA"},
        {"argument_id": "A5", "section": "Paper evaluation", "claim": "Review and upstream amendment loops improve the fidelity of the final paper.", "evidence_needed": "matched W0/W1/W2 drafts, blinded review, hard-gate outcomes, and cost", "counterargument": "Review may merely increase verbosity and expense.", "response": "Report quality, integrity, cost, and time jointly, and accept the null or negative result.", "status": "LOCKED_NO_PAPER_ABLATION"},
    ]


def _figure_rows(paths: dict[str, Path], experiment_status: dict[str, Any]) -> list[dict[str, Any]]:
    real = int(experiment_status.get("record_count") or 0) > 0
    paper_evaluation_complete = _read_json(paths["paper_evaluation"] / "paper_evaluation_summary.json").get("status") == "COMPLETE_EVALUATION"
    return [
        {"item_id": "T1", "kind": "table", "paper_section": "Method", "title": "Seven baseline-agent search contracts", "source": str(paths["catalog"].resolve()), "status": "READY_STATIC", "claim_boundary": "taxonomy only"},
        {"item_id": "T2", "kind": "table", "paper_section": "Experimental protocol", "title": "Eighteen task and native-metric contracts", "source": str(paths["catalog"].resolve()), "status": "READY_STATIC", "claim_boundary": "suite composition only"},
        {"item_id": "T3", "kind": "table", "paper_section": "Experimental protocol", "title": "Twelve process metrics and implementation semantics", "source": str((paths["knowledge_base"] / "metric_implementation_audit.json").resolve()), "status": "READY_STATIC", "claim_boundary": "metric definitions, not observed values"},
        {"item_id": "F1", "kind": "figure", "paper_section": "Method", "title": "Evidence-gated research-to-paper DAG", "source": str((paths["catalog_dir"] / "figures" / "unified_research_paper_pipeline.svg").resolve()), "status": "READY_STATIC", "claim_boundary": "architecture only"},
        {"item_id": "F2", "kind": "figure", "paper_section": "Related work", "title": "Published FML agent-task performance heatmap", "source": str((paths["published_prior"] / "figures" / "published_agent_task_heatmap.svg").resolve()), "status": "READY_PUBLISHED_CONTEXT", "claim_boundary": "published prior; not local evidence"},
        {"item_id": "T4", "kind": "table", "paper_section": "Results", "title": "Matched agent comparisons on complete seed blocks", "source": str((paths["experiments"] / "paired_agent_comparisons.csv").resolve()), "status": "READY_NEW_DATA" if real else "LOCKED_NO_DATA", "claim_boundary": "fixed suite and frozen model/provider only"},
        {"item_id": "T5", "kind": "table", "paper_section": "Results", "title": "Per-task paired effects and heterogeneity", "source": str((paths["experiments"] / "paired_agent_task_effects.csv").resolve()), "status": "READY_NEW_DATA" if real else "LOCKED_NO_DATA", "claim_boundary": "descriptive fixed-task heterogeneity"},
        {"item_id": "F3", "kind": "figure", "paper_section": "Results", "title": "Overall normalized improvement with seed-block uncertainty", "source": str((paths["experiments"] / "figures" / "agent_overall_normalized_improvement.svg").resolve()), "status": "READY_NEW_DATA" if (paths["experiments"] / "figures" / "agent_overall_normalized_improvement.svg").is_file() else "LOCKED_NO_REPLICATED_DATA", "claim_boundary": "no ranking beyond displayed uncertainty"},
        {"item_id": "F4", "kind": "figure", "paper_section": "Process analysis", "title": "Separate process-metric uncertainty panels", "source": str((paths["experiments"] / "figures" / "process").resolve()), "status": "READY_NEW_DATA" if int(experiment_status.get('process_metric_figure_count') or 0) > 0 else "LOCKED_NO_PROCESS_DATA", "claim_boundary": "incompatible units never share an axis"},
        {"item_id": "T6", "kind": "table", "paper_section": "Robustness", "title": "Official-versus-sensitivity scorer semantics", "source": str((paths["experiments"] / "scorer_semantics_sensitivity.csv").resolve()), "status": "READY_NEW_DATA" if real else "LOCKED_NO_DATA", "claim_boundary": "sensitivity columns never replace official metrics"},
        {"item_id": "T7", "kind": "table", "paper_section": "Paper evaluation", "title": "Matched W0/W1/W2 manuscript evaluation", "source": str((paths["paper_evaluation"] / "paper_evaluation_summary.json").resolve()), "status": "READY_LOCAL_EXTENSION" if paper_evaluation_complete else "LOCKED_NO_COMPLETE_REVIEW_DATA", "claim_boundary": "paper quality remains separate from FML"},
        {"item_id": "T8", "kind": "table", "paper_section": "Ablations", "title": "AdaptiveSearch opportunity-regime interaction", "source": str((paths["experiments"] / "adaptive_opportunity_interactions.csv").resolve()), "status": "READY_NEW_DATA" if int(experiment_status.get('complete_confirmatory_adaptive_interaction_count') or 0) == 6 else "LOCKED_NO_COMPLETE_INTERACTIONS", "claim_boundary": "published post-hoc labels define strata; only new outcomes enter the estimator"},
    ]


def _methods(catalog: dict[str, Any]) -> str:
    agents = catalog["agents"]
    tasks = catalog["tasks"]
    process_metrics = [metric for metric in catalog["metrics"] if metric["category"] != "task_performance"]
    agent_table = _table(
        ["Agent", "Search topology", "Selection", "Reusable operator"],
        [[agent["display_name"], agent["strategy_family"], agent["selection_policy"], TRANSFER[agent["agent_id"]]] for agent in agents],
    )
    task_table = _table(
        ["Task", "Domain", "Lite", "Native metric", "Direction"],
        [[task["task_id"], task["domain"], "yes" if task["lite"] else "no", task["canonical_metric"], task["canonical_direction"]] for task in tasks],
    )
    metric_table = _table(
        ["Family", "Metric", "Checked-in definition"],
        [[metric["category"], metric["name"], metric["definition"]] for metric in process_metrics],
    )
    return "\n".join(
        [
            "# Evidence-locked manuscript",
            "",
            "Status: **METHODS DRAFT ONLY. ABSTRACT RESULTS, EMPIRICAL RESULTS, DISCUSSION CLAIMS, AND CONCLUSION ARE LOCKED.**",
            "",
            "Working title: *From Search Dynamics to Evidence-Gated Papers: A Self-Evolving ML Research Pipeline*",
            "",
            "## Abstract",
            "",
            "[LOCKED until all confirmatory comparisons, sensitivity analyses, and claim-evidence hashes pass.]",
            "",
            "## 1. Introduction",
            "",
            "Automatic ML research agents are often compared by the final score of the code they produce. That endpoint is necessary but incomplete. Two agents can reach similar protected-test performance through different search trajectories, failure rates, validation efficiency, and computational costs. A research-to-paper system introduces another distinction: experimental evidence and manuscript quality are related, but they are not the same construct and should not be collapsed into one score.",
            "",
            "This study treats seven FML research-agent implementations as explicit search contracts. Their proposal, selection, memory, debugging, and diversity operators are placed inside a shared evidence-gated DAG. The graph also plans paper nodes at Stage 3, but final writing remains locked behind experiments, confirmatory statistics, and a frozen claim-evidence bundle. When writing or review exposes an empirical defect, the issue traces back to a versioned analysis, experiment, or code amendment rather than being repaired only in prose.",
            "",
            "The empirical study asks how search topology affects a fixed suite of ML research tasks under matched budgets. A separate manuscript ablation asks whether evidence gates and multi-role review improve paper integrity. Published FML aggregates are used to form hypotheses and diagnostic expectations, never as substitutes for new trials.",
            "",
            "## 2. Related-work boundary",
            "",
            f"The current FML study, *{PAPER['title']}* ({PAPER['arxiv_id']}{PAPER['version']}), supplies the benchmark protocol and published aggregate context. Repository source dossiers characterize the seven local implementations. A venue-specific literature review remains required before submission; no additional citation is asserted in this provisional draft.",
            "",
            "## 3. System and agent methods",
            "",
            "### 3.1 Evidence-gated controller",
            "",
            "The controller has eight outer stages: memory snapshot, evidence-contract freeze, method/skill selection, unified graph planning, dependency-ready execution, type-specific review, repair or backtracking, and final governance. Every node receives a typed inner loop with allowed operators, reviewers, hard gates, outcomes, and budgets. Research execution, research review, paper writing, and paper review use separate evidence-gated skill registries; no skill can mutate a running experimental arm.",
            "",
            "### 3.2 Baseline-derived search operators",
            "",
            agent_table,
            "",
            "The common graph does not claim these agents are equivalent. It preserves their primary state representation and search decisions while standardizing the task interface, model/provider, step budget, validation boundary, protected test, and provenance requirements.",
            "",
            "### 3.3 Paper-node locking and amendment semantics",
            "",
            "The outline and methods skeleton can be drafted before experiments. Numerical Results, conclusion-bearing abstract text, and final paper writing cannot unlock until the protected test is followed by matched confirmatory analysis and a hashed evidence bundle. Review issues are triaged to writing, analysis, experiments, or code. Any empirical change after protected-test exposure requires a new hidden evaluation; the exposed test result never becomes search feedback.",
            "",
            "## 4. Experimental protocol",
            "",
            "### 4.1 Tasks and outcomes",
            "",
            task_table,
            "",
            "Each task retains its native validation and protected-test metric. Cross-task summaries use the checked-in FML normalized-improvement transform and official baseline-fallback policy. Failed tests and runs with no valid validation candidate remain in the analysis with baseline credit; they are not deleted as outliers.",
            "",
            "### 4.2 Controlled comparison",
            "",
            "The diagnostic pilot contains 14 runs (seven agents on two contrasting tasks) and is ineligible for primary claims. The confirmatory Lite design contains 168 runs: seven agents by eight tasks by three frozen seeds. Every comparison uses the same model, provider, task commit, target-file boundary, step budget, validation feedback, and single protected-test policy. The full planned matrix therefore contains 182 runs. Pilot outcomes cannot add, remove, or replace confirmatory arms, tasks, metrics, or trials.",
            "",
            "### 4.3 Process evaluation",
            "",
            metric_table,
            "",
            "The official scorer is frozen for comparability. Two implementation-sensitive variants are reported only as named sensitivities: the first rather than last successful match to the best validation metric, and exploration over validation-success snapshots rather than every persisted snapshot.",
            "",
            "### 4.4 Statistical analysis",
            "",
            "The primary estimand is the paired difference in FML normalized improvement on the fixed Lite suite. For each seed, outcomes are averaged equally over all eight preregistered tasks; uncertainty is computed across complete paired seed-block means. The paper reports the mean and median paired effect, sample SD, two-sided 95% Student-t interval, Cohen dz when defined, seed-block win/tie/loss counts, and an exact two-sided sign test. Holm correction controls the family of all unordered agent comparisons within a phase, model, and provider. Per-task effects and task/task-seed win rates describe heterogeneity but are not treated as independent primary replications. Three seeds provide weak inferential resolution, so effect estimates and intervals lead; a non-significant result is not interpreted as equivalence.",
            "",
            "A preregistered secondary interaction tests the published opportunity-regime hypothesis using new outcomes only. Within each seed and fixed baseline, we average the AdaptiveSearch-minus-baseline contrast separately over dense and sparse tasks and subtract the sparse average from the dense average. Six AdaptiveSearch-versus-baseline interactions receive their own Holm correction. The dense/sparse labels originated from a published post-hoc analysis, so they define frozen strata but contribute no local outcome evidence.",
            "",
            "### 4.5 Separate paper-quality evaluation",
            "",
            "Paper quality is not an FML metric. Three writing arms receive the same frozen evidence bundle, writing model/provider, and token budget: W0 one-shot writing, W1 evidence gating without peer revision, and W2 the full integrity, review, amendment, re-review, and final-integrity pipeline. Blinded reviewers score scientific validity, methodology, statistics, reproducibility, positioning, clarity, and limitations only after six hard integrity gates pass. These scores are never averaged with agent performance or process metrics.",
            "",
            "## 5. Results",
            "",
            "[LOCKED. Populate only from `paired_agent_comparisons.csv`, `paired_agent_task_effects.csv`, failure counts, and hashed figures after complete confirmatory blocks pass.]",
            "",
            "## 6. Process analysis, error analysis, and ablations",
            "",
            "[LOCKED. Report invalid executions, baseline-fallback credit, task heterogeneity, process metrics, mechanism ablations, and scorer sensitivities without changing official columns.]",
            "",
            "## 7. Discussion",
            "",
            "[LOCKED. Alternative explanations must include model/provider dependence, three-seed precision, fixed-suite scope, task-cost heterogeneity, benchmark contamination, and the post-hoc origin of the opportunity split.]",
            "",
            "## 8. Limitations and threats to validity",
            "",
            "The design is conditional on a fixed model and provider, covers a finite benchmark suite, and uses three confirmatory seeds. The protected test reduces adaptive overfitting but cannot eliminate benchmark familiarity or contamination. Process metrics depend on the checked-in snapshot and best-step semantics. The manuscript-quality experiment evaluates one evidence context and cannot establish universal writing superiority. These limitations must be updated after observing run failures and uncertainty.",
            "",
            "## 9. Conclusion",
            "",
            "[LOCKED. No agent-superiority or pipeline-benefit claim before confirmatory evidence and manuscript review.]",
            "",
            "## Declarations",
            "",
            "- **Data and code availability:** repository commit and artifact archive to be inserted after evidence freeze.",
            "- **Ethics:** no human-subject claim is made; task-specific data licenses and risks require final verification.",
            "- **Author contributions (CRediT):** awaiting human author assignment.",
            "- **Funding:** awaiting human declaration.",
            "- **Competing interests:** awaiting human declaration.",
            "- **AI-tool disclosure:** awaiting target-venue policy and complete process record.",
            "",
            "## Verified reference currently admissible",
            "",
            f"[1] {PAPER['title']}. arXiv:{PAPER['arxiv_id']}{PAPER['version']}. {PAPER['abstract_url']}",
            "",
        ]
    )


def _readiness(
    configuration: dict[str, Any],
    catalog: dict[str, Any],
    paths: dict[str, Path],
    experiment_status: dict[str, Any],
    claims: list[dict[str, Any]],
) -> dict[str, Any]:
    protocol = _read_json(paths["protocol"] / "experiment_protocol.json")
    preflight = _read_json(paths["protocol"] / "preflight.json")
    paper_eval_summary = _read_json(paths["paper_evaluation"] / "paper_evaluation_summary.json")
    pair_rows = _csv_rows(paths["experiments"] / "paired_agent_comparisons.csv")
    complete_pairs = sum(row.get("complete_block_gate_passed", "").lower() == "true" and row.get("analysis_class") == "confirmatory" for row in pair_rows)
    gates = [
        {"gate": "paper_configuration_confirmed", "pass": configuration["confirmation_status"] == "CONFIRMED", "evidence": "paper_configuration.json"},
        {"gate": "complete_repository_knowledge", "pass": catalog["counts"]["agents"] == 7 and catalog["counts"]["tasks"] == 18 and catalog["counts"]["process_metrics"] == 12, "evidence": str(paths["catalog"].resolve())},
        {"gate": "experiment_protocol_frozen", "pass": protocol.get("status") == "FROZEN_READY_FOR_PREFLIGHT", "evidence": str((paths["protocol"] / "experiment_protocol.json").resolve())},
        {"gate": "execution_preflight_ready", "pass": preflight.get("ready") is True, "evidence": str((paths["protocol"] / "preflight.json").resolve())},
        {"gate": "all_planned_runs_ingested", "pass": int(experiment_status.get("record_count") or 0) == int(protocol.get("total_planned_runs") or -1), "evidence": str((paths["experiments"] / "experiment_dataset_status.json").resolve())},
        {"gate": "all_21_confirmatory_pairs_complete", "pass": complete_pairs == 21, "evidence": str((paths["experiments"] / "paired_agent_comparisons.csv").resolve())},
        {"gate": "process_metrics_ingested", "pass": int(experiment_status.get("record_count") or 0) > 0 and int(experiment_status.get("process_metric_run_coverage_count") or 0) == int(experiment_status.get("record_count") or 0), "evidence": str((paths["experiments"] / "process_metric_records.csv").resolve())},
        {"gate": "scorer_sensitivity_audited", "pass": int(experiment_status.get("record_count") or 0) > 0 and int(experiment_status.get("scorer_sensitivity_run_count") or 0) == int(experiment_status.get("record_count") or 0), "evidence": str((paths["experiments"] / "scorer_semantics_sensitivity_status.json").resolve())},
        {"gate": "empirical_claims_unlocked", "pass": all(not claim["status"].startswith("LOCKED") for claim in claims if claim["claim_type"] == "new_empirical"), "evidence": "claim_evidence_registry.csv"},
        {"gate": "paper_ablation_review_and_human_release_complete", "pass": paper_eval_summary.get("status") == "COMPLETE_EVALUATION" and paper_eval_summary.get("human_release_approved") is True, "evidence": str((paths["paper_evaluation"] / "paper_evaluation_summary.json").resolve())},
    ]
    empirical_ready = all(gate["pass"] for gate in gates[4:9])
    review_ready = gates[-1]["pass"]
    if empirical_ready and review_ready and gates[0]["pass"]:
        stage = "READY_FOR_FINAL_INTEGRITY"
    elif empirical_ready and gates[0]["pass"]:
        stage = "READY_FOR_RESULTS_DRAFT_AND_REVIEW"
    else:
        stage = "METHODS_ONLY_EMPIRICAL_SECTIONS_LOCKED"
    return {
        "schema_version": "fml-scientist-paper-readiness-v1",
        "stage": stage,
        "ready_for_numerical_results": empirical_ready,
        "ready_for_finalization": empirical_ready and review_ready and gates[0]["pass"],
        "gates": gates,
        "passed_gate_n": sum(gate["pass"] for gate in gates),
        "total_gate_n": len(gates),
        "locked_claim_n": sum(claim["status"].startswith("LOCKED") for claim in claims),
        "next_actions": [
            gate["gate"] for gate in gates if not gate["pass"]
        ],
        "integrity_rule": "No Results prose, conclusion-bearing abstract, full peer review, or final PDF before its prerequisite gates pass.",
    }


def _readiness_markdown(readiness: dict[str, Any]) -> str:
    rows = [
        [gate["gate"], "PASS" if gate["pass"] else "LOCKED", gate["evidence"]]
        for gate in readiness["gates"]
    ]
    return "\n".join(
        [
            "# Paper readiness",
            "",
            f"Current stage: **{readiness['stage']}**",
            "",
            _table(["Gate", "Status", "Evidence"], rows),
            "",
            f"Passed {readiness['passed_gate_n']} of {readiness['total_gate_n']} gates; {readiness['locked_claim_n']} registered claims remain locked.",
            "",
            "Numerical Results and finalization are deliberately unavailable until the missing gates are satisfied. Empty CSVs and published-prior tables cannot unlock new empirical prose.",
            "",
        ]
    )


def write_paper_package(
    catalog: dict[str, Any],
    out_dir: Path,
    *,
    catalog_dir: Path,
    knowledge_base_dir: Path,
    plans_dir: Path,
    published_prior_dir: Path,
    protocol_dir: Path,
    experiments_dir: Path,
    paper_evaluation_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "catalog_dir": catalog_dir,
        "catalog": catalog_dir / "fml_catalog.json",
        "knowledge_base": knowledge_base_dir,
        "plans": plans_dir,
        "published_prior": published_prior_dir,
        "protocol": protocol_dir,
        "experiments": experiments_dir,
        "paper_evaluation": paper_evaluation_dir,
    }
    experiment_status = _read_json(experiments_dir / "experiment_dataset_status.json")
    configuration = _configuration(catalog)
    claims = _claim_rows(paths, catalog, experiment_status)
    figures = _figure_rows(paths, experiment_status)
    readiness = _readiness(configuration, catalog, paths, experiment_status, claims)
    _write_json(out_dir / "paper_configuration.json", configuration)
    (out_dir / "research_question_brief.md").write_text(_research_brief(catalog), encoding="utf-8")
    _write_csv(
        out_dir / "argument_blueprint.csv",
        _argument_rows(),
        ["argument_id", "section", "claim", "evidence_needed", "counterargument", "response", "status"],
    )
    _write_csv(
        out_dir / "claim_evidence_registry.csv",
        claims,
        ["claim_id", "claim_type", "paper_section", "permissible_wording", "prohibited_overclaim", "required_evidence", "artifact_sha256", "unlock_rule", "status"],
    )
    _write_csv(
        out_dir / "table_figure_plan.csv",
        figures,
        ["item_id", "kind", "paper_section", "title", "source", "status", "claim_boundary"],
    )
    (out_dir / "evidence_locked_manuscript.md").write_text(_methods(catalog), encoding="utf-8")
    _write_json(out_dir / "paper_readiness.json", readiness)
    (out_dir / "paper_readiness.md").write_text(_readiness_markdown(readiness), encoding="utf-8")
    manifest_files = sorted(path for path in out_dir.iterdir() if path.is_file() and path.name != "paper_package_manifest.json")
    manifest = {
        "schema_version": "fml-scientist-paper-package-v1",
        "repository_commit": catalog["repository_commit"],
        "paper_stage": readiness["stage"],
        "artifact_count": len(manifest_files),
        "artifacts": [
            {"path": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
            for path in manifest_files
        ],
        "new_empirical_results_present": int(experiment_status.get("record_count") or 0) > 0,
        "published_prior_merged_into_new_results": False,
    }
    _write_json(out_dir / "paper_package_manifest.json", manifest)
    return readiness
