"""Paper-quality evaluation that complements, but is not part of, FML-bench."""

from __future__ import annotations

import csv
import json
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
        ],
        "skill_promotion_rule": "paper-writing or paper-review skills may be promoted only after a complete held-out paper evaluation with no hard-gate regression in a materially distinct context",
    }


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
                            "criterion": criterion,
                            "weight": weight,
                            "score_1_to_5": "",
                            "major_issue": "",
                            "evidence_path": "",
                            "review_complete": False,
                        }
                    )
    _write_csv(out_dir / "paper_review_score_template.csv", review_rows, list(review_rows[0]))
    hard_gate_rows = [
        {
            "manuscript_blind_id": "",
            "gate_id": gate_id,
            "requirement": requirement,
            "pass": "",
            "evidence_path": "",
            "reviewer": "",
        }
        for gate_id, requirement in HARD_GATES
    ]
    _write_csv(out_dir / "hard_gate_template.csv", hard_gate_rows, list(hard_gate_rows[0]))
    status = {
        "status": "TEMPLATE_NO_PAPER_SCORES",
        "rubric_criterion_n": len(RUBRIC),
        "reviewer_role_n": len(REVIEWERS),
        "hard_gate_n": len(HARD_GATES),
        "writing_arm_n": len(WRITING_ARMS),
        "planned_review_row_n": len(review_rows),
        "fml_metric_merge_allowed": False,
    }
    (out_dir / "paper_evaluation_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return status
