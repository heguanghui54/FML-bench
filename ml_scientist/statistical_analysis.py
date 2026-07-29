"""Preregistered analysis contract for controlled FML agent comparisons.

The contract separates the fixed-suite estimand from task-level diagnostics and
from scorer-semantics sensitivity analyses.  It is written before results are
seen so reporting decisions cannot drift with the observed rankings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_statistical_analysis_protocol(
    *,
    primary_phase: str = "confirmatory_lite",
    suite_label: str = "the fixed FML-Lite confirmatory task suite",
    seed_role: str = "research-agent seed",
) -> dict[str, Any]:
    return {
        "schema_version": "fml-scientist-statistical-analysis-v1",
        "status": "PREREGISTRATION_TEMPLATE_MUST_BE_FROZEN_BEFORE_PROTECTED_TESTS",
        "scope": "new controlled campaign records only; published FML aggregates are excluded",
        "estimand": {
            "population": f"{suite_label} under the frozen model, provider, budgets, and task commits",
            "outcome": "FML normalized protected-test improvement with the checked-in fallback policy",
            "contrast": "agent-left minus agent-right, averaged equally over every task in a complete trial block",
            "experimental_run_unit": f"one agent-task-{seed_role.replace(' ', '_')} run",
            "primary_uncertainty_unit": f"one matched seed trial ({seed_role}) after averaging the complete fixed task suite",
            "task_seed_boundary": (
                "Task split/training seeds are benchmark-owned and fixed across agents and trials; "
                "the trial seed measures research-agent stochasticity, not independent model-training replication."
            ),
            "why": "tasks are repeated benchmark components inside an agent-seed block and are not treated as independent stochastic replications",
        },
        "primary_analysis": {
            "phase": primary_phase,
            "complete_case_rule": "include a paired seed block only when both agents have one credited outcome for every preregistered task",
            "reported_statistics": [
                "paired seed-block mean and median difference",
                "sample SD across paired seed-block means",
                "two-sided 95% Student-t confidence interval",
                "paired standardized effect Cohen dz when the paired SD is nonzero",
                "exact two-sided sign-test p-value over non-tied seed-block means",
            ],
            "multiplicity": "Holm family-wise adjustment across all unordered agent pairs within each phase, model, and provider",
            "decision_boundary": "effect sizes and intervals are primary; adjusted p-values are supporting evidence, not a ranking substitute",
            "small_n_warning": "three seed blocks give weak inferential resolution; absence of significance is not evidence of equivalence",
        },
        "secondary_descriptive_analyses": [
            {
                "name": "task_heterogeneity",
                "unit": "matched seed difference within each task",
                "outputs": "per-task mean, SD, interval, Cohen dz, and seed-level win/tie/loss counts",
            },
            {
                "name": "suite_wide_robustness",
                "unit": "task mean after averaging matched seeds",
                "outputs": "task win/tie/loss rates and an exact sign test with a separate Holm adjustment",
                "claim_boundary": "descriptive robustness across the fixed task suite, not population inference over all ML problems",
            },
            {
                "name": "run_level_behavior",
                "unit": "matched task-seed cell",
                "outputs": "cell win/tie/loss counts and rates",
                "claim_boundary": "descriptive only because cells share tasks and seeds",
            },
            {
                "name": "adaptive_opportunity_interaction",
                "unit": "within each seed, AdaptiveSearch-minus-baseline mean on dense tasks minus the same mean on sparse tasks",
                "outputs": "six AdaptiveSearch-versus-baseline interaction estimates, intervals, sign tests, and Holm adjustment",
                "claim_boundary": "the published post-hoc partition is frozen as a new hypothesis stratum; only new campaign outcomes test it",
            },
        ],
        "failure_and_missingness": {
            "failed_protected_test": "retain the run and apply the official task-baseline fallback credit",
            "all_validation_steps_failed": "retain the run and apply the official task-baseline fallback credit",
            "missing_summary_or_duplicate_cell": "do not impute; fail the complete-block gate and disclose the affected run IDs",
            "attrition": "report attempted runs, successful protected tests, fallback credits, and excluded incomplete paired blocks",
        },
        "sensitivity_analyses": [
            {
                "name": "first_achieved_best_step",
                "official": "last successful exact match to summary.best_val_metric",
                "sensitivity": "first successful exact match to summary.best_val_metric",
                "replacement_allowed": False,
            },
            {
                "name": "valid_only_exploration_membership",
                "official": "every persisted step snapshot",
                "sensitivity": "persisted snapshots whose validation step succeeded",
                "replacement_allowed": False,
                "recomputation": "GraphCodeBERT exploration metrics are recomputed only after real snapshots exist; the reporter first emits membership diagnostics",
            },
        ],
        "claim_eligibility": {
            "pilot": "diagnostic only",
            "confirmatory_lite": "eligible only after protocol freeze, complete provenance, and multiplicity-aware reporting",
            "confirmatory_full": "eligible as a separately labeled extension if frozen before execution",
            "confirmatory_heldout_transfer": "eligible only after exposure audit, protocol freeze, complete provenance, and multiplicity-aware reporting",
            "published_prior": "context and hypothesis formation only",
        },
        "precision_escalation": {
            "rule": "any increase beyond the three planned seeds must be specified and frozen before viewing protected-test outcomes",
            "prohibited": "adding seeds, tasks, or arms because an observed p-value or ranking is unfavorable",
        },
        "required_tables": [
            "experiment_records.csv",
            "experiment_cell_integrity_issues.csv",
            "agent_trial_aggregates.csv",
            "agent_overall_statistics.csv",
            "paired_agent_comparisons.csv",
            "paired_agent_task_effects.csv",
            "adaptive_opportunity_interactions.csv",
            "scorer_semantics_sensitivity.csv",
        ],
    }


def _markdown(protocol: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Statistical analysis and paper-result contract",
            "",
            f"Status: **{protocol['status']}**",
            "",
            "## Primary estimand",
            "",
            "Compare two agents using FML normalized protected-test improvement. Within each matched research-agent seed, average all preregistered tasks equally; uncertainty is then computed across complete agent-seed block means. Task training/split seeds remain benchmark-owned and fixed, so these trials measure research-agent stochasticity and must not be described as independent model-training seeds.",
            "",
            "## Required primary result table",
            "",
            "| Left agent | Right agent | Complete seed blocks | Mean paired difference | 95% CI | Cohen dz | Seed wins/ties/losses | Exact sign p | Holm p |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
            "| populate only from `paired_agent_comparisons.csv` | | | | | | | | |",
            "",
            "Positive differences favor the left agent. Three seed blocks have weak inferential resolution, so the paper must lead with the estimate and interval and must not treat a non-significant result as equivalence.",
            "",
            "## Required task-heterogeneity table",
            "",
            "| Left agent | Right agent | Task | Matched seeds | Mean paired difference | 95% CI | Cohen dz | Seed wins/ties/losses |",
            "|---|---|---|---:|---:|---:|---:|---:|",
            "| populate only from `paired_agent_task_effects.csv` | | | | | | | |",
            "",
            "Task and task-seed win rates are descriptive views of heterogeneity. Only complete, preregistered confirmatory blocks can support the main comparison.",
            "",
            "## AdaptiveSearch opportunity-regime interaction",
            "",
            "For each seed and baseline, compute AdaptiveSearch minus baseline on each task, average that contrast within the published dense and sparse strata, then subtract sparse from dense. Report all six baseline interactions with intervals and a separate Holm correction. The published partition is a frozen hypothesis label, not local outcome evidence.",
            "",
            "## Failure, multiplicity, and sensitivity rules",
            "",
            "- Keep failures in the analysis with the official FML baseline-fallback credit and report failure counts separately.",
            "- Apply Holm adjustment within each phase/model/provider family to every unordered agent comparison.",
            "- Keep pilot results diagnostic and published aggregates contextual.",
            "- Report first-achieved-best and valid-only snapshot analyses under explicit sensitivity labels; never overwrite official scorer columns.",
            "- Freeze any additional seeds before viewing protected-test outcomes.",
            "",
        ]
    )


def write_statistical_analysis_protocol(
    out_dir: Path,
    *,
    primary_phase: str = "confirmatory_lite",
    suite_label: str = "the fixed FML-Lite confirmatory task suite",
    seed_role: str = "research-agent seed",
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_statistical_analysis_protocol(
        primary_phase=primary_phase,
        suite_label=suite_label,
        seed_role=seed_role,
    )
    (out_dir / "statistical_analysis_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "statistical_analysis_protocol.md").write_text(
        _markdown(protocol), encoding="utf-8"
    )
    return protocol
