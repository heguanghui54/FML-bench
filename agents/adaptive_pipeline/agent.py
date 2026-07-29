"""FML execution adapter for the graph-bound adaptive research pipeline."""

from __future__ import annotations

import json
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from ml_scientist.adaptive_fml_executor import (
    audit_modified_code_reachability,
    build_adaptive_fml_operator,
    enforce_resource_budget,
    validate_post_execution_policy_evaluations,
)
from ml_scientist.adaptive_controller import assess_negative_memory_candidate
from ml_scientist.strategy_operator import compile_policy_assessment_operator
from ml_scientist.artifact_contract import materialize_planner_artifacts

from ..base import AgentResult, StepResult
from ..llm import get_response_from_llm
from ..autoresearch.agent import AutoresearchAgent, ExperimentRecord


logger = logging.getLogger(__name__)


class AdaptivePipelineAgent(AutoresearchAgent):
    """One executor arm whose decisions come from frozen atomic-policy memory.

    Registration is a post-baseline-gate extension: the frozen seven-baseline
    campaign completed before this agent was added to the harness registry.
    """

    def __init__(self, config):
        super().__init__(config)
        self.graph: dict[str, Any] = {}
        self.plan: dict[str, Any] = {}
        self.adaptive_contracts: list[dict[str, Any]] = []
        self.adaptive_assessments: list[dict[str, Any]] = []
        self.reachability_audits: list[dict[str, Any]] = []
        self._latest_planned_policy_use: list[dict[str, Any]] = []
        self._adaptive_started_at = 0.0

    def initialize(self) -> None:
        super().initialize()

    def _load_adaptive_context(self) -> None:
        params = self.config.agent_params
        graph_path = Path(params.get(
            "graph_path", "artifacts/ml_scientist/bootstrap/knowledge_graph/knowledge_graph.json",
        ))
        plan_root = Path(params.get("plan_root", "artifacts/ml_scientist/bootstrap/plans"))
        task_id = self.config.runtime_params.get("benchmark_name")
        if not task_id:
            raise ValueError("adaptive_pipeline requires runtime_params.benchmark_name before plan loading")
        plan_path = plan_root / f"{task_id}.json"
        self.graph = json.loads(graph_path.read_text(encoding="utf-8"))
        self.plan = json.loads(plan_path.read_text(encoding="utf-8"))

    def run(self, task_description=None, target_files=None, baseline_results=None) -> AgentResult:
        # BenchmarkRunner injects benchmark_name and the remaining task runtime
        # parameters immediately before run(), not before initialize().
        self._load_adaptive_context()
        self.adaptive_contracts = []
        self.adaptive_assessments = []
        self.reachability_audits = []
        self._adaptive_started_at = time.monotonic()
        result = super().run(task_description, target_files, baseline_results)
        result.metadata["adaptive_pipeline"] = {
            "schema_version": "adaptive-fml-agent-trace-v1",
            "graph_content_sha256": self.graph.get("content_sha256"),
            "plan_task_id": self.plan.get("task_id"),
            "decision_contracts": self.adaptive_contracts,
            "reachability_audits": self.reachability_audits,
            "post_execution_assessments": self.adaptive_assessments,
            "pre_execution_credit_assignment_allowed": False,
            "protected_metric_used_for_routing": False,
            "runtime_marker_status": "PILOT_STATIC_GATE_ONLY",
        }
        return result

    def _resource_budget(self) -> dict[str, int | float]:
        params = self.config.agent_params
        return {
            "candidate_steps": self.step_budget,
            "token_budget": int(params.get("token_budget", 120000)),
            "wall_clock_seconds": float(params.get("wall_clock_seconds", 7200)),
        }

    def _budget_gate(self) -> dict[str, Any]:
        return enforce_resource_budget(
            tokens_consumed=int(self.get_token_usage_summary().get("total_tokens", 0)),
            wall_clock_seconds=max(time.monotonic() - self._adaptive_started_at, 0.0),
            candidate_steps_consumed=self.step_count,
            budget=self._resource_budget(),
        )

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
        try:
            value = json.loads(stripped)
            return value if isinstance(value, dict) else {"action": text}
        except json.JSONDecodeError:
            return {"action": text, "planned_policy_use": []}

    @staticmethod
    def _artifact_descriptor(
        path: str | Path,
        artifact_type: str,
        content_summary: dict[str, Any],
    ) -> dict[str, Any]:
        artifact_path = Path(path)
        payload = artifact_path.read_bytes()
        return {
            "path": str(artifact_path),
            "artifact_type": artifact_type,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "size_bytes": len(payload),
            "content_summary": content_summary,
        }

    def _latest_validation_artifact(self) -> Path | None:
        paths = list(Path(self._parent_workspace).rglob("val_info.json"))
        return max(paths, key=lambda path: path.stat().st_mtime_ns) if paths else None

    @staticmethod
    def _classify_validation_failure(error: str | None) -> dict[str, Any]:
        text = error or ""
        if "Invalid results:" not in text and "constraint violated" not in text.lower():
            return {"kind": "EXECUTION_FAILED", "error_tail": text[-2000:]}
        violation_candidates = re.findall(r"Invalid results:\s*([^\n]+)", text)
        violation_reason = next(
            (value for value in reversed(violation_candidates) if "{" not in value),
            violation_candidates[-1] if violation_candidates else "constraint violation",
        )
        average_match = re.search(
            r"Average result: AUC ([0-9.]+), TPR@0\.1%FPR of ([0-9.]+), TPR@0\.0%FPR of ([0-9.]+)",
            text,
        )
        accuracy_match = re.search(r"Accuracy constraint violated:\s*([0-9.]+)\s*<\s*([0-9.]+)", text)
        max_tpr_match = re.search(r"TPR@0\.1%=([0-9.]+), TPR@0\.0%=([0-9.]+)", text)
        observed: dict[str, float] = {}
        if average_match:
            auc = float(average_match.group(1))
            observed.update({
                "auc_mean": auc,
                "auc_gap_mean": abs(auc - 0.5),
                "tpr_at_0_1_fpr_mean": float(average_match.group(2)),
                "tpr_at_0_0_fpr_mean": float(average_match.group(3)),
            })
        if accuracy_match:
            observed.update({
                "test_acc_mean": float(accuracy_match.group(1)),
                "test_acc_minimum": float(accuracy_match.group(2)),
            })
        if max_tpr_match:
            observed.update({
                "tpr_at_0_1_fpr_gate_value": float(max_tpr_match.group(1)),
                "tpr_at_0_0_fpr_gate_value": float(max_tpr_match.group(2)),
            })
        return {
            "kind": "CONSTRAINT_FAILED",
            "violation_reason": violation_reason,
            "observed": observed,
        }

    def _record_constraint_failure(self, failure: dict[str, Any], idea: str) -> None:
        self._restore_snapshot(self.current_snapshot)
        self.consecutive_crashes = 0
        self._pending_debug = False
        self.experiment_log.append(ExperimentRecord(
            step_id=self.step_count,
            primary_metric=None,
            status="constraint_failed",
            description=idea[:200],
            error_context=str(failure.get("violation_reason", "constraint violation"))[:300],
        ))
        print(
            f"  CONSTRAINT_FAILED | step={self.step_count} | "
            f"{failure.get('violation_reason', 'constraint violation')}"
        )

    def _generate_idea(self) -> str:
        visible_evidence = [{
            "metric_name": self.metric_name,
            "value": self.current_best_metric,
            "observability": "ONLINE_VISIBLE",
            "split": "validation",
            "status": "baseline_or_current_best",
        }]
        remaining = max(0.0, 1.0 - self.step_count / max(self.step_budget, 1))
        contract = build_adaptive_fml_operator(
            plan=self.plan,
            graph=self.graph,
            run_id=self.config.runtime_params.get("workspace_label", self._parent_workspace),
            node_id="hypothesis-frontier",
            visible_evidence=visible_evidence,
            budget_before={**self._resource_budget(), "remaining_fraction": remaining},
        )
        prompt = contract["operator"]["instruction"] + "\n\n## Current allowed code\n" + self._format_current_code()
        text, _, usage = get_response_from_llm(
            prompt,
            client=self.client,
            model=self.config.model,
                system_message=(
                "You are the planning component of an adaptive ML research pipeline. "
                "Return only the requested JSON. Plan one minimal, testable modification; "
                "do not claim execution, results, policy credit, or learning signals."
            ),
        )
        if usage:
            self.token_usage_log.append(usage)
        parsed = self._json_object(text)
        planned_action = parsed.get("action")
        nested_distinction = (
            planned_action.get("counterfactual_distinction")
            if isinstance(planned_action, dict) else None
        )
        negative_gate = assess_negative_memory_candidate(
            planned_action,
            contract.get("negative_memory", []),
            counterfactual_distinction=(
                parsed.get("counterfactual_distinction") or nested_distinction
            ),
        )
        contract["negative_memory_gate"] = negative_gate
        self._latest_negative_memory_rejection = not negative_gate["passed"]
        planned = parsed.get("planned_policy_use")
        self._latest_planned_policy_use = planned if isinstance(planned, list) else []
        contract["planner_output"] = parsed
        contract_path = Path(self._parent_workspace) / f"adaptive_contract_step_{self.step_count + 1:04d}.json"
        contract_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        contract["artifact_path"] = str(contract_path)
        self.adaptive_contracts.append(contract)
        action = parsed.get("action")
        if isinstance(action, str):
            return action.strip()
        return json.dumps(action, ensure_ascii=False)

    def _post_execution_assessment(
        self,
        *,
        contract: dict[str, Any],
        outcome: str,
        evidence_artifacts: list[dict[str, Any]],
        audit: dict[str, Any],
        val_result: dict[str, Any],
        step_tokens: dict[str, Any],
        step_seconds: float,
    ) -> dict[str, Any]:
        node = next(row for row in self.plan["nodes"] if row["node_id"] == "hypothesis-frontier")
        assessment = compile_policy_assessment_operator(
            graph=self.graph,
            selected_policy_by_role=contract["decision"]["selected_policy_by_role"],
            node=node,
            planned_policy_use=self._latest_planned_policy_use,
            outcome=outcome,
            evidence_artifacts=evidence_artifacts,
            state_before={"validation_metric": self.current_best_metric},
            next_state={
                "validation_success": bool(val_result.get("success")),
                "validation_metric": val_result.get("primary_metric"),
                "reachability_gate_passed": audit.get("passed"),
            },
            resource_telemetry={"tokens_consumed": step_tokens.get("total_tokens"), "wall_clock_seconds": step_seconds},
        )
        fallback = [
            {
                "policy_id": policy_id,
                "role": role,
                "applied": False,
                "status": "INCONCLUSIVE",
                "evidence_artifact_refs": [row["path"] for row in evidence_artifacts],
                "gate_checks": ["post_execution_review_unavailable_or_invalid"],
                "learning_signal": {"intent": "NONE"},
            }
            for role, policy_id in contract["decision"]["selected_policy_by_role"].items()
        ]
        assessment["policy_evaluations"] = fallback
        assessment["review_status"] = "NOT_RUN_BUDGET_OR_VALIDATION_GATE"
        # Persist the exact hashed descriptors, not only their rendering inside
        # the prompt. This makes failed reviewer calls independently repairable
        # without reconstructing or weakening the evidence boundary.
        assessment["evidence_artifacts"] = evidence_artifacts
        gate = self._budget_gate()
        if evidence_artifacts and gate["checks"]["token_budget"]:
            attempts = 0
            try:
                attempts += 1
                text, _, usage = get_response_from_llm(
                    assessment["instruction"],
                    client=self.client,
                    model=self.config.model,
                    system_message=(
                        "You are a post-execution ML research auditor. Return only JSON with "
                        "policy_evaluations. Credit a policy only when its planned use is "
                        "supported by one of the supplied real artifact references. A successful "
                        "node does not imply every policy passed. Use learning intent NONE unless "
                    "the observed transition supports a reusable procedure with a future held-out gate."
                ),
                budget_category="review",
            )
                if usage:
                    self.token_usage_log.append(usage)
                reviewed = self._json_object(text)
                assessment["policy_evaluations"] = validate_post_execution_policy_evaluations(
                    payload=reviewed,
                    selected_policy_by_role=contract["decision"]["selected_policy_by_role"],
                    evidence_artifacts=evidence_artifacts,
                )
                assessment["review_status"] = "VALIDATED_POST_EXECUTION_LLM_REVIEW"
            except Exception as exc:
                assessment["review_error"] = str(exc)
                # One schema-repair attempt is allowed and charged as another
                # review. The repair prompt enumerates the only valid artifact
                # references; it may fix formatting, but may not invent credit.
                if self._budget_gate()["checks"]["token_budget"]:
                    try:
                        attempts += 1
                        allowed_refs = [
                            value
                            for artifact in evidence_artifacts
                            for value in (artifact.get("path"), artifact.get("sha256"))
                            if value
                        ]
                        repair_instruction = (
                            assessment["instruction"]
                            + "\n\nYour previous response failed validation: " + str(exc)
                            + "\nFor every applied policy, evidence_artifact_refs must use only exact values "
                              "from this JSON list (no aliases or role names):\n"
                            + json.dumps(allowed_refs, indent=2, ensure_ascii=False)
                        )
                        text, _, usage = get_response_from_llm(
                            repair_instruction,
                            client=self.client,
                            model=self.config.model,
                            system_message=(
                                "You are repairing an evidence-gated policy assessment. Return only JSON with "
                                "policy_evaluations. Do not change a substantive judgment merely to pass schema. "
                                "Use only exact supplied artifact paths or hashes as evidence references."
                            ),
                            budget_category="review",
                        )
                        if usage:
                            self.token_usage_log.append(usage)
                        reviewed = self._json_object(text)
                        assessment["policy_evaluations"] = validate_post_execution_policy_evaluations(
                            payload=reviewed,
                            selected_policy_by_role=contract["decision"]["selected_policy_by_role"],
                            evidence_artifacts=evidence_artifacts,
                        )
                        assessment["review_status"] = "VALIDATED_POST_EXECUTION_LLM_REVIEW_AFTER_SCHEMA_RETRY"
                    except Exception as repair_exc:
                        assessment["review_status"] = "REJECTED_REVIEW_FALLBACK_INCONCLUSIVE"
                        assessment["review_repair_error"] = str(repair_exc)
                else:
                    assessment["review_status"] = "REJECTED_REVIEW_FALLBACK_INCONCLUSIVE"
            assessment["review_attempt_count"] = attempts
        return assessment

    def _main_loop(self) -> None:
        while self.budget_remaining():
            if not self._budget_gate()["passed"]:
                break
            token_start = len(self.token_usage_log)
            step_started = time.monotonic()
            idea = self._generate_idea()
            contract = self.adaptive_contracts[-1]
            if not self._budget_gate()["checks"]["token_budget"]:
                break
            self._restore_snapshot(self.current_snapshot)
            before = self._snapshot_target_files()
            edit_ok = False if getattr(self, "_latest_negative_memory_rejection", False) else self._edit_code(idea)
            after = self._snapshot_target_files()
            audit = audit_modified_code_reachability(before, after)
            if getattr(self, "_latest_negative_memory_rejection", False):
                audit["passed"] = False
                audit["errors"].append("NEGATIVE_MEMORY_DUPLICATE_WITHOUT_JUSTIFIED_OVERRIDE")
            audit_path = Path(self._parent_workspace) / f"reachability_step_{self.step_count + 1:04d}.json"
            audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            audit["artifact_path"] = str(audit_path)
            self.reachability_audits.append(audit)
            planner_artifacts: list[dict[str, Any]] = []
            if edit_ok and audit["passed"]:
                try:
                    planner_artifacts = materialize_planner_artifacts(
                        run_dir=Path(self._parent_workspace),
                        step_id=self.step_count + 1,
                        planner_output=contract.get("planner_output") or {},
                        before=before,
                        after=after,
                        research_context={
                            "task_id": self.config.runtime_params.get("benchmark_name"),
                            "workspace_label": self.config.runtime_params.get("workspace_label"),
                            "experimental_seed": (self.config.runtime_params.get("experiment") or {}).get("seed"),
                        },
                    )
                except Exception as exc:
                    audit["passed"] = False
                    audit["errors"].append(f"ARTIFACT_CONTRACT_FAILED:{exc}")
                    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            if edit_ok and audit["passed"]:
                val_result = self._execute_val(self.step_count)
            else:
                self.step_count += 1
                val_result = {
                    "success": False,
                    "primary_metric": None,
                    "error": "PRE_VALIDATION_GATE_FAILED:" + ",".join(audit["errors"]),
                }
            step_seconds = time.monotonic() - step_started
            step_tokens = self._collect_step_tokens(token_start)
            if planner_artifacts:
                planner_artifacts = materialize_planner_artifacts(
                    run_dir=Path(self._parent_workspace),
                    step_id=self.step_count,
                    planner_output=contract.get("planner_output") or {},
                    before=before,
                    after=after,
                    research_context={
                        "task_id": self.config.runtime_params.get("benchmark_name"),
                        "workspace_label": self.config.runtime_params.get("workspace_label"),
                        "experimental_seed": (self.config.runtime_params.get("experiment") or {}).get("seed"),
                    },
                    validation={
                        "success": val_result.get("success"),
                        "primary_metric": val_result.get("primary_metric"),
                        "filtered_results": val_result.get("filtered_results"),
                        "failure": self._classify_validation_failure(val_result.get("error"))
                        if not val_result.get("success") else None,
                    },
                    resources={
                        "tokens_consumed": step_tokens.get("total_tokens"),
                        "wall_clock_seconds": step_seconds,
                    },
                )
            snapshot_path = self._save_step_code_snapshot(self.step_count, self._parent_workspace)
            changed_files = sorted(key for key in set(before) | set(after) if before.get(key) != after.get(key))
            artifacts = [
                self._artifact_descriptor(
                    contract["artifact_path"], "decision_contract", {
                        "contract_sha256": contract.get("contract_sha256"),
                        "selected_policy_by_role": contract["decision"]["selected_policy_by_role"],
                        "planner_action": contract.get("planner_output", {}).get("action"),
                        "planner_rationale": contract.get("planner_output", {}).get("rationale"),
                        "planned_policy_use": self._latest_planned_policy_use,
                    },
                ),
                self._artifact_descriptor(
                    audit_path, "reachability_audit", {
                        key: audit.get(key)
                        for key in ("passed", "changed_files", "new_functions", "uncalled_new_functions", "syntax_errors", "errors")
                    },
                ),
                self._artifact_descriptor(
                    snapshot_path, "code_snapshot", {
                        "changed_files": changed_files,
                        "changed_content_sha256": {
                            key: hashlib.sha256(after[key].encode("utf-8")).hexdigest()
                            for key in changed_files if key in after
                        },
                        "parent_content_sha256": {
                            key: hashlib.sha256(before[key].encode("utf-8")).hexdigest()
                            for key in changed_files if key in before
                        },
                    },
                ),
                *planner_artifacts,
            ]
            validation_path = self._latest_validation_artifact()
            if validation_path is not None:
                artifacts.append(self._artifact_descriptor(
                    validation_path, "validation_result", {
                        "success": val_result.get("success"),
                        "primary_metric": val_result.get("primary_metric"),
                        "filtered_results": val_result.get("filtered_results"),
                        "error": val_result.get("error"),
                    },
                ))
            failure: dict[str, Any] | None = None
            if not val_result.get("success") or val_result.get("primary_metric") is None:
                failure = self._classify_validation_failure(val_result.get("error"))
                outcome = failure["kind"]
                failure_path = Path(self._parent_workspace) / f"validation_failure_step_{self.step_count:04d}.json"
                failure_path.write_text(json.dumps(failure, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                artifacts.append(self._artifact_descriptor(
                    failure_path, "validation_failure", failure,
                ))
            elif self._is_strict_improvement(val_result["primary_metric"]):
                outcome = "VALID_CANDIDATE"
            else:
                outcome = "VALID_REGRESSION"
            assessment = self._post_execution_assessment(
                contract=contract, outcome=outcome, evidence_artifacts=artifacts,
                audit=audit, val_result=val_result, step_tokens=step_tokens, step_seconds=step_seconds,
            )
            assessment_path = Path(self._parent_workspace) / f"assessment_step_{self.step_count:04d}.json"
            assessment_path.write_text(json.dumps(assessment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            assessment["artifact_path"] = str(assessment_path)
            self.adaptive_assessments.append(assessment)
            self.all_steps.append(StepResult(
                step_id=self.step_count,
                idea_id=f"adaptive_exp_{self.step_count}",
                idea_description=idea[:200],
                action="adaptive_policy_composition",
                edit_success=edit_ok,
                val_result=val_result,
                primary_metric=val_result.get("primary_metric"),
                token_usage=step_tokens,
                step_duration_seconds=step_seconds,
                metadata={
                    "code_snapshot_path": snapshot_path,
                    "idea": idea,
                    "instruction": getattr(self, "_last_edit_instruction", idea),
                    "editor_log_path": getattr(self._last_edit_result, "log_path", None),
                    "adaptive_contract_path": contract["artifact_path"],
                    "reachability_audit_path": str(audit_path),
                    "assessment_path": str(assessment_path),
                    "selected_policy_by_role": contract["decision"]["selected_policy_by_role"],
                    "policy_evaluations": assessment["policy_evaluations"],
                },
            ))
            if outcome == "CONSTRAINT_FAILED" and failure is not None:
                self._record_constraint_failure(failure, idea)
            elif not val_result.get("success") or val_result.get("primary_metric") is None:
                self._handle_crash(val_result, idea)
            elif self._is_strict_improvement(val_result["primary_metric"]):
                self._keep(val_result["primary_metric"], idea)
            else:
                self._discard(val_result["primary_metric"], idea)
