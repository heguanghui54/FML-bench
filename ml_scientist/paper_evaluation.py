"""Paper-quality evaluation that complements, but is not part of, FML-bench."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


RUBRIC = [
    ("scientific_validity", 0.25, "Claims follow from the frozen evidence and conclusions respect uncertainty."),
    ("methodology_and_controls", 0.20, "Methods, controls, baselines, data boundaries, and ablations answer the research question."),
    ("statistical_analysis", 0.15, "Estimands, units, uncertainty, multiplicity, missingness, and limitations are handled correctly."),
    ("reproducibility", 0.15, "A reader can resolve versions, configurations, seeds, artifacts, and commands needed to reproduce results."),
    ("novelty_and_positioning", 0.10, "Contributions are distinguished from prior work without overstating novelty."),
    ("clarity_and_structure", 0.10, "The paper is coherent, precise, and uses tables and figures that expose the evidence."),
    ("limitations_and_ethics", 0.05, "Threats to validity, negative results, failure cases, and broader impacts are explicit."),
]

REVIEWERS = [
    ("editor_in_chief", "scope, contribution, coherence, and decision synthesis"),
    ("methodology", "experimental design, controls, leakage, and causal strength of claims"),
    ("statistics", "estimands, uncertainty, multiplicity, dependence, and visualization"),
    ("reproducibility", "artifact completeness, version traceability, seeds, and executable instructions"),
    ("devils_advocate", "counterexamples, alternative explanations, hidden assumptions, and overclaims"),
]

HARD_GATES = [
    ("HG1", "Every numerical manuscript claim resolves to a current evidence row and SHA-256 artifact."),
    ("HG2", "No empirical result is copied from published-prior evidence and presented as a local campaign result."),
    ("HG3", "Every citation resolves and supports the adjacent claim; no fabricated or retracted source is treated as valid."),
    ("HG4", "Methods, model/provider, task commits, seeds, budgets, and platform match the executed provenance."),
    ("HG5", "Protected-test results never appear in search or method-selection history."),
    ("HG6", "All review issues and upstream amendments have an owner, disposition, and current evidence version."),
]

WRITING_ARMS = [
    ("W0_one_shot", "One evidence-conditioned draft with no structured review loop."),
    ("W1_gated_no_peer_revision", "Traceability gates and structured sections, but no multi-role revision cycle."),
    ("W2_full_pipeline", "Evidence gates, multi-role review, upstream amendment triage, revision, re-review, and final integrity."),
]

_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def _t95(df: int) -> float:
    return _T95.get(df, 1.96 if df > 30 else 2.262)


def _float_score(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if 1.0 <= parsed <= 5.0 else None


def _truthy(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "major", "critical", "pass", "passed"}


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


def _paired_statistics(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "sd": None, "ci95_halfwidth": None, "ci95_low": None, "ci95_high": None, "cohen_dz": None, "wins": 0, "ties": 0, "losses": 0, "exact_sign_p": None}
    n = len(values)
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if n >= 2 else None
    ci = _t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
    wins = sum(value > 1e-12 for value in values)
    losses = sum(value < -1e-12 for value in values)
    ties = n - wins - losses
    non_ties = wins + losses
    sign_p = None
    if non_ties:
        tail = min(wins, losses)
        sign_p = min(1.0, 2.0 * sum(math.comb(non_ties, k) for k in range(tail + 1)) / (2 ** non_ties))
    return {
        "n": n, "mean": mean, "sd": sd, "ci95_halfwidth": ci,
        "ci95_low": mean - ci if ci is not None else None,
        "ci95_high": mean + ci if ci is not None else None,
        "cohen_dz": mean / sd if sd is not None and sd > 0 else None,
        "wins": wins, "ties": ties, "losses": losses, "exact_sign_p": sign_p,
    }


def _icc2k(score_matrix: dict[str, dict[str, float]]) -> float | None:
    """Two-way random-effects, absolute-agreement ICC(2,k)."""
    criteria = [criterion for criterion, scores in score_matrix.items() if len(scores) == len(REVIEWERS)]
    reviewers = [role for role, _ in REVIEWERS]
    if len(criteria) < 2 or any(any(role not in score_matrix[criterion] for role in reviewers) for criterion in criteria):
        return None
    n, k = len(criteria), len(reviewers)
    matrix = [[score_matrix[criterion][role] for role in reviewers] for criterion in criteria]
    grand = statistics.fmean(value for row in matrix for value in row)
    row_means = [statistics.fmean(row) for row in matrix]
    column_means = [statistics.fmean(matrix[i][j] for i in range(n)) for j in range(k)]
    ms_rows = k * sum((value - grand) ** 2 for value in row_means) / (n - 1)
    ms_columns = n * sum((value - grand) ** 2 for value in column_means) / (k - 1)
    residual_ss = sum(
        (matrix[i][j] - row_means[i] - column_means[j] + grand) ** 2
        for i in range(n) for j in range(k)
    )
    ms_error = residual_ss / ((n - 1) * (k - 1))
    denominator = ms_rows + (ms_columns - ms_error) / n
    return (ms_rows - ms_error) / denominator if abs(denominator) > 1e-15 else None


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_paper_evaluation_protocol() -> dict[str, Any]:
    return {
        "schema_version": "fml-scientist-paper-evaluation-v1",
        "benchmark_status": "local_extension_not_part_of_fml_bench",
        "purpose": "Evaluate whether the pipeline converts one frozen empirical evidence bundle into a valid, reproducible, well-reviewed ML paper.",
        "separation_from_fml": {
            "fml_evaluates": "ML research search outcomes and process dynamics",
            "this_extension_evaluates": "manuscript integrity, scientific reasoning, reporting, and reproducibility",
            "scores_must_not_be_averaged_together": True,
        },
        "hard_gates": [{"gate_id": gate_id, "requirement": requirement} for gate_id, requirement in HARD_GATES],
        "rubric": [
            {"criterion": criterion, "weight": weight, "anchor_1": "serious deficiencies", "anchor_3": "adequate with material limitations", "anchor_5": "publication-ready", "definition": definition}
            for criterion, weight, definition in RUBRIC
        ],
        "reviewer_roles": [{"role": role, "focus": focus} for role, focus in REVIEWERS],
        "decision_rule": {
            "hard_gate_rule": "all hard gates pass; any failure is blocking",
            "criterion_scale": "integer 1-5 from each reviewer",
            "aggregate": "weighted mean of per-criterion reviewer medians",
            "minimum_weighted_score": 4.0,
            "minimum_each_criterion_median": 3.0,
            "major_issue_rule": "zero unresolved major issues",
            "human_checkpoint": "a human editor verifies the final integrity package and makes the release decision",
        },
        "matched_writing_ablation": {
            "arms": [{"arm": arm, "description": description} for arm, description in WRITING_ARMS],
            "drafts_per_arm": 3,
            "reviewers_per_draft": len(REVIEWERS),
            "total_planned_reviews": len(WRITING_ARMS) * 3 * len(REVIEWERS),
            "controls": [
                "same frozen evidence bundle and claim contract",
                "same writing model/provider and matched token budget",
                "independent frozen draft seeds",
                "blinded randomized manuscript identifiers and review order",
                "reviewers cannot see arm labels or other reviewer scores",
                "each manuscript-reviewer report has a distinct run ID, seed, randomized order, file path, and SHA-256 hash",
            ],
            "primary_estimand": "paired difference in weighted paper-quality score between W2 and W0 under the same evidence bundle",
            "secondary_outcomes": ["hard-gate pass rate", "major issue count", "unsupported claim count", "revision count", "review cost", "time to acceptance"],
            "analysis": [
                "report every draft and reviewer score",
                "aggregate reviewer medians by criterion before applying weights",
                "report paired arm differences with uncertainty; do not infer population superiority from one evidence bundle",
                "report inter-rater agreement and adjudication changes",
                "keep reviewer text and issue dispositions as provenance artifacts",
            ],
        },
        "required_outputs": [
            "hard_gate_results.csv",
            "review_scores.csv",
            "review_reports.json",
            "issue_dispositions.csv",
            "claim_evidence_map.json",
            "reproducibility_manifest.json",
            "paper_evaluation_summary.json",
            "paper_release_decision.json",
        ],
        "skill_promotion_rule": "paper-writing or paper-review skills may be promoted only after a complete held-out paper evaluation with no hard-gate regression in a materially distinct context",
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def evaluate_paper_evaluation(data_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Reduce completed blinded review records without editing any manuscript."""
    out_dir.mkdir(parents=True, exist_ok=True)
    review_rows = _read_csv(data_dir / "review_scores.csv")
    hard_rows = _read_csv(data_dir / "hard_gate_results.csv")
    disposition_rows = _read_csv(data_dir / "issue_dispositions.csv")
    release_decision = _read_json(data_dir / "paper_release_decision.json")
    weights = {criterion: weight for criterion, weight, _ in RUBRIC}
    criteria = list(weights)
    reviewer_roles = [role for role, _ in REVIEWERS]
    arms = [arm for arm, _ in WRITING_ARMS]

    score_cells: dict[tuple[str, int, str, str], list[dict[str, str]]] = defaultdict(list)
    hard_cells: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    score_ids: dict[tuple[str, int], set[str]] = defaultdict(set)
    hard_ids: dict[tuple[str, int], set[str]] = defaultdict(set)
    for row in review_rows:
        try:
            draft = int(row.get("draft") or 0)
        except ValueError:
            draft = 0
        arm = row.get("arm_private") or ""
        score_cells[(arm, draft, row.get("criterion") or "", row.get("reviewer_role") or "")].append(row)
        if row.get("manuscript_blind_id"):
            score_ids[(arm, draft)].add(row["manuscript_blind_id"])
    for row in hard_rows:
        try:
            draft = int(row.get("draft") or 0)
        except ValueError:
            draft = 0
        arm = row.get("arm_private") or ""
        hard_cells[(arm, draft, row.get("gate_id") or "")].append(row)
        if row.get("manuscript_blind_id"):
            hard_ids[(arm, draft)].add(row["manuscript_blind_id"])
    resolved_statuses = {"RESOLVED", "REVIEWER_DISAGREE_UPHELD"}
    dispositions = {
        (row.get("manuscript_blind_id") or "", row.get("issue_id") or ""): (row.get("status") or "").strip().upper()
        for row in disposition_rows
        if row.get("manuscript_blind_id") and row.get("issue_id")
    }
    review_provenance_issues: list[str] = []
    seen_run_ids: set[str] = set()
    seen_seeds: set[str] = set()
    seen_orders: set[int] = set()
    for arm in arms:
        for draft in range(1, 4):
            for role in reviewer_roles:
                cells = [
                    row
                    for criterion in criteria
                    for row in score_cells[(arm, draft, criterion, role)]
                ]
                if len(cells) != len(criteria):
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:missing_or_duplicate_criterion_rows")
                    continue
                metadata_fields = ("reviewer_run_id", "reviewer_seed", "review_order", "reviewer_model", "report_path", "report_sha256")
                metadata = {field: {str(row.get(field) or "").strip() for row in cells} for field in metadata_fields}
                if any(len(values) != 1 or not next(iter(values)) for values in metadata.values()):
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:inconsistent_or_missing_review_provenance")
                    continue
                run_id = next(iter(metadata["reviewer_run_id"]))
                seed = next(iter(metadata["reviewer_seed"]))
                if run_id in seen_run_ids:
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:reused_reviewer_run_id")
                seen_run_ids.add(run_id)
                if seed in seen_seeds:
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:reused_reviewer_seed")
                seen_seeds.add(seed)
                try:
                    order = int(next(iter(metadata["review_order"])))
                except ValueError:
                    order = 0
                if order < 1 or order > len(WRITING_ARMS) * 3 * len(REVIEWERS) or order in seen_orders:
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:invalid_or_reused_blind_review_order")
                seen_orders.add(order)
                report_path = Path(next(iter(metadata["report_path"])))
                expected_hash = next(iter(metadata["report_sha256"]))
                if not report_path.is_file() or _file_sha256(report_path) != expected_hash:
                    review_provenance_issues.append(f"{arm}:{draft}:{role}:report_hash_mismatch")
    reviewer_independence_audit_pass = not review_provenance_issues
    manuscript_rows: list[dict[str, Any]] = []
    criterion_rows: list[dict[str, Any]] = []
    for arm in arms:
        for draft in range(1, 4):
            blind_ids = score_ids[(arm, draft)] | hard_ids[(arm, draft)]
            blind_id = next(iter(blind_ids)) if len(blind_ids) == 1 else ""
            matrix: dict[str, dict[str, float]] = defaultdict(dict)
            criterion_complete = True
            major_issue_ids: set[str] = set()
            for criterion in criteria:
                values: list[float] = []
                for role in reviewer_roles:
                    cells = score_cells[(arm, draft, criterion, role)]
                    if len(cells) != 1 or cells[0].get("manuscript_blind_id") != blind_id:
                        criterion_complete = False
                        continue
                    value = _float_score(cells[0].get("score_1_to_5"))
                    if value is None or not _truthy(cells[0].get("review_complete")):
                        criterion_complete = False
                        continue
                    values.append(value)
                    matrix[criterion][role] = value
                    if _truthy(cells[0].get("major_issue")):
                        issue_id = (cells[0].get("issue_id") or "").strip()
                        major_issue_ids.add(issue_id or f"MISSING_ID:{criterion}:{role}")
                criterion_rows.append(
                    {
                        "manuscript_blind_id": blind_id,
                        "arm_private": arm,
                        "draft": draft,
                        "criterion": criterion,
                        "weight": weights[criterion],
                        "reviewer_n": len(values),
                        "median_score": statistics.median(values) if values else None,
                        "mean_score": statistics.fmean(values) if values else None,
                        "sd_score": statistics.stdev(values) if len(values) >= 2 else None,
                        "min_score": min(values) if values else None,
                        "max_score": max(values) if values else None,
                    }
                )
            hard_gate_complete = True
            hard_gate_pass = True
            for gate_id, _ in HARD_GATES:
                cells = hard_cells[(arm, draft, gate_id)]
                if len(cells) != 1 or cells[0].get("manuscript_blind_id") != blind_id or not str(cells[0].get("pass") or "").strip():
                    hard_gate_complete = False
                    hard_gate_pass = False
                elif not _truthy(cells[0].get("pass")):
                    hard_gate_pass = False
            medians = {
                row["criterion"]: row["median_score"]
                for row in criterion_rows
                if row["arm_private"] == arm and row["draft"] == draft and row["median_score"] is not None
            }
            weighted = sum(weights[criterion] * medians[criterion] for criterion in criteria) if len(medians) == len(criteria) else None
            unresolved = sum(
                issue_id.startswith("MISSING_ID:") or dispositions.get((blind_id, issue_id)) not in resolved_statuses
                for issue_id in major_issue_ids
            )
            complete = (
                len(blind_ids) == 1 and bool(blind_id) and criterion_complete
                and hard_gate_complete and len(medians) == len(criteria)
            )
            if not complete:
                decision = "INCOMPLETE_REVIEW_DATA"
            elif not hard_gate_pass:
                decision = "BLOCKED_HARD_GATE"
            elif unresolved:
                decision = "BLOCKED_UNRESOLVED_MAJOR_ISSUE"
            elif weighted is not None and weighted >= 4.0 and min(medians.values()) >= 3.0:
                decision = "PASS_MANUSCRIPT_THRESHOLD"
            else:
                decision = "REVISE_BELOW_SCORE_THRESHOLD"
            manuscript_rows.append(
                {
                    "manuscript_blind_id": blind_id,
                    "arm_private": arm,
                    "draft": draft,
                    "review_data_complete": complete,
                    "hard_gate_pass": hard_gate_pass if hard_gate_complete else False,
                    "weighted_paper_quality_score": weighted,
                    "minimum_criterion_median": min(medians.values()) if medians else None,
                    "major_issue_n": len(major_issue_ids),
                    "unresolved_major_issue_n": unresolved,
                    "icc2k_absolute_agreement": _icc2k(matrix),
                    "decision": decision,
                }
            )
    _write_csv(
        out_dir / "manuscript_criterion_statistics.csv",
        criterion_rows,
        ["manuscript_blind_id", "arm_private", "draft", "criterion", "weight", "reviewer_n", "median_score", "mean_score", "sd_score", "min_score", "max_score"],
    )
    _write_csv(
        out_dir / "manuscript_scores.csv",
        manuscript_rows,
        ["manuscript_blind_id", "arm_private", "draft", "review_data_complete", "hard_gate_pass", "weighted_paper_quality_score", "minimum_criterion_median", "major_issue_n", "unresolved_major_issue_n", "icc2k_absolute_agreement", "decision"],
    )
    arm_rows: list[dict[str, Any]] = []
    for arm in arms:
        members = [row for row in manuscript_rows if row["arm_private"] == arm]
        scores = [float(row["weighted_paper_quality_score"]) for row in members if row["review_data_complete"] and row["weighted_paper_quality_score"] is not None]
        stats = _paired_statistics(scores)
        arm_rows.append(
            {
                "arm_private": arm,
                "complete_draft_n": len(scores),
                "mean_weighted_score": stats["mean"],
                "sd_across_drafts": stats["sd"],
                "ci95_halfwidth_student_t": stats["ci95_halfwidth"],
                "hard_gate_pass_n": sum(row["hard_gate_pass"] for row in members),
                "paper_threshold_pass_n": sum(row["decision"] == "PASS_MANUSCRIPT_THRESHOLD" for row in members),
                "unresolved_major_issue_n": sum(int(row["unresolved_major_issue_n"]) for row in members),
            }
        )
    _write_csv(
        out_dir / "writing_arm_statistics.csv",
        arm_rows,
        ["arm_private", "complete_draft_n", "mean_weighted_score", "sd_across_drafts", "ci95_halfwidth_student_t", "hard_gate_pass_n", "paper_threshold_pass_n", "unresolved_major_issue_n"],
    )
    score_index = {
        (row["arm_private"], int(row["draft"])): float(row["weighted_paper_quality_score"])
        for row in manuscript_rows
        if row["review_data_complete"] and row["weighted_paper_quality_score"] is not None
    }
    planned_contrasts = [("W2_full_pipeline", "W0_one_shot"), ("W2_full_pipeline", "W1_gated_no_peer_revision"), ("W1_gated_no_peer_revision", "W0_one_shot")]
    comparison_rows: list[dict[str, Any]] = []
    for left, right in planned_contrasts:
        differences = [score_index[(left, draft)] - score_index[(right, draft)] for draft in range(1, 4) if (left, draft) in score_index and (right, draft) in score_index]
        stats = _paired_statistics(differences)
        comparison_rows.append(
            {
                "arm_left": left, "arm_right": right, "matched_draft_n": stats["n"],
                "mean_difference_left_minus_right": stats["mean"], "sd_paired_difference": stats["sd"],
                "ci95_halfwidth_student_t": stats["ci95_halfwidth"], "ci95_low": stats["ci95_low"], "ci95_high": stats["ci95_high"],
                "cohen_dz": stats["cohen_dz"], "wins_left": stats["wins"], "ties": stats["ties"], "losses_left": stats["losses"],
                "exact_sign_p": stats["exact_sign_p"],
            }
        )
    ordered = sorted(
        ((index, row["exact_sign_p"]) for index, row in enumerate(comparison_rows) if row["exact_sign_p"] is not None),
        key=lambda item: float(item[1]),
    )
    running = 0.0
    for rank, (index, p_value) in enumerate(ordered):
        running = max(running, min(1.0, (len(comparison_rows) - rank) * float(p_value)))
        comparison_rows[index]["holm_p"] = running
    for row in comparison_rows:
        row.setdefault("holm_p", None)
    _write_csv(
        out_dir / "writing_arm_paired_comparisons.csv",
        comparison_rows,
        ["arm_left", "arm_right", "matched_draft_n", "mean_difference_left_minus_right", "sd_paired_difference", "ci95_halfwidth_student_t", "ci95_low", "ci95_high", "cohen_dz", "wins_left", "ties", "losses_left", "exact_sign_p", "holm_p"],
    )
    expected_review_rows = len(WRITING_ARMS) * 3 * len(REVIEWERS) * len(RUBRIC)
    expected_hard_rows = len(WRITING_ARMS) * 3 * len(HARD_GATES)
    support_artifacts = {
        "review_reports.json": bool(_read_json(data_dir / "review_reports.json")),
        "claim_evidence_map.json": bool(_read_json(data_dir / "claim_evidence_map.json")),
        "reproducibility_manifest.json": bool(_read_json(data_dir / "reproducibility_manifest.json")),
        "issue_dispositions.csv": (data_dir / "issue_dispositions.csv").is_file(),
    }
    complete = (
        len(review_rows) == expected_review_rows
        and len(hard_rows) == expected_hard_rows
        and all(row["review_data_complete"] for row in manuscript_rows)
        and all(support_artifacts.values())
        and reviewer_independence_audit_pass
    )
    selected_blind_id = str(release_decision.get("selected_manuscript_blind_id") or "")
    selected_manuscript = next(
        (row for row in manuscript_rows if row["manuscript_blind_id"] == selected_blind_id),
        None,
    )
    human_release_approved = (
        str(release_decision.get("approval_status") or "").upper() == "APPROVED"
        and bool(str(release_decision.get("human_editor") or "").strip())
        and bool(str(release_decision.get("rationale") or "").strip())
        and selected_manuscript is not None
        and selected_manuscript["arm_private"] == "W2_full_pipeline"
        and selected_manuscript["decision"] == "PASS_MANUSCRIPT_THRESHOLD"
        and Path(str(release_decision.get("verified_final_integrity_report") or "")).is_file()
    )
    summary = {
        "schema_version": "fml-scientist-paper-evaluation-results-v1",
        "status": "COMPLETE_EVALUATION" if complete else "INCOMPLETE_EVALUATION_DATA",
        "reviewer_read_only_contract": True,
        "fml_metric_merge_allowed": False,
        "expected_manuscript_n": len(WRITING_ARMS) * 3,
        "evaluated_manuscript_n": sum(row["review_data_complete"] for row in manuscript_rows),
        "review_score_row_n": len(review_rows),
        "expected_review_score_row_n": expected_review_rows,
        "hard_gate_row_n": len(hard_rows),
        "expected_hard_gate_row_n": expected_hard_rows,
        "support_artifacts": support_artifacts,
        "reviewer_independence_audit_pass": reviewer_independence_audit_pass,
        "review_provenance_issue_n": len(review_provenance_issues),
        "review_provenance_issues": review_provenance_issues,
        "paper_threshold_pass_n": sum(row["decision"] == "PASS_MANUSCRIPT_THRESHOLD" for row in manuscript_rows),
        "hard_gate_blocked_n": sum(row["decision"] == "BLOCKED_HARD_GATE" for row in manuscript_rows),
        "unresolved_major_issue_blocked_n": sum(row["decision"] == "BLOCKED_UNRESOLVED_MAJOR_ISSUE" for row in manuscript_rows),
        "human_release_approved": human_release_approved,
        "selected_manuscript_blind_id": selected_blind_id or None,
        "human_release_rule": "A named human editor must approve one passing W2 manuscript with a recorded rationale before finalization.",
        "primary_contrast": next((row for row in comparison_rows if row["arm_left"] == "W2_full_pipeline" and row["arm_right"] == "W0_one_shot"), None),
        "claim_boundary": "A complete evaluation describes these nine drafts under one frozen evidence bundle; it does not prove universal writing-pipeline superiority.",
    }
    (out_dir / "paper_evaluation_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return summary


def write_paper_evaluation_protocol(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    protocol = build_paper_evaluation_protocol()
    (out_dir / "paper_evaluation_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    rubric_rows = [
        {
            "criterion": criterion,
            "weight": weight,
            "definition": definition,
            "score_1": "serious deficiencies",
            "score_3": "adequate with material limitations",
            "score_5": "publication-ready",
        }
        for criterion, weight, definition in RUBRIC
    ]
    _write_csv(out_dir / "paper_review_rubric.csv", rubric_rows, list(rubric_rows[0]))
    review_rows = []
    for arm, _ in WRITING_ARMS:
        for draft in range(1, 4):
            for role, _ in REVIEWERS:
                for criterion, weight, _ in RUBRIC:
                    review_rows.append(
                        {
                            "manuscript_blind_id": "ASSIGN_BEFORE_REVIEW",
                            "arm_private": arm,
                            "draft": draft,
                            "reviewer_role": role,
                            "reviewer_run_id": "",
                            "reviewer_seed": "",
                            "review_order": "",
                            "reviewer_model": "",
                            "report_path": "",
                            "report_sha256": "",
                            "criterion": criterion,
                            "weight": weight,
                            "score_1_to_5": "",
                            "major_issue": "",
                            "issue_id": "",
                            "evidence_path": "",
                            "review_complete": False,
                        }
                    )
    _write_csv(out_dir / "paper_review_score_template.csv", review_rows, list(review_rows[0]))
    hard_gate_rows = []
    for arm, _ in WRITING_ARMS:
        for draft in range(1, 4):
            for gate_id, requirement in HARD_GATES:
                hard_gate_rows.append(
                    {
                        "manuscript_blind_id": "",
                        "arm_private": arm,
                        "draft": draft,
                        "gate_id": gate_id,
                        "requirement": requirement,
                        "pass": "",
                        "evidence_path": "",
                        "reviewer": "",
                    }
                )
    _write_csv(out_dir / "hard_gate_template.csv", hard_gate_rows, list(hard_gate_rows[0]))
    _write_csv(
        out_dir / "issue_disposition_template.csv",
        [],
        ["manuscript_blind_id", "issue_id", "severity", "status", "owner", "evidence_path", "rationale"],
    )
    (out_dir / "paper_release_decision_template.json").write_text(
        json.dumps(
            {
                "approval_status": "AWAITING_HUMAN_DECISION",
                "selected_manuscript_blind_id": "",
                "human_editor": "",
                "rationale": "",
                "verified_final_integrity_report": "",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    status = {
        "status": "TEMPLATE_NO_PAPER_SCORES",
        "rubric_criterion_n": len(RUBRIC),
        "reviewer_role_n": len(REVIEWERS),
        "hard_gate_n": len(HARD_GATES),
        "writing_arm_n": len(WRITING_ARMS),
        "planned_review_row_n": len(review_rows),
        "planned_hard_gate_row_n": len(hard_gate_rows),
        "fml_metric_merge_allowed": False,
    }
    (out_dir / "paper_evaluation_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status
