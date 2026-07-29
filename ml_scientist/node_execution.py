"""Executable node layer for the graph-bound research-to-paper DAG."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from agents.code_editor import CodeEditor
from agents.llm import create_client, get_response_from_llm

from .adaptive_runtime import AdaptiveResearchRuntime
from .execution_contracts import (
    BudgetLedger,
    MATCHED_BUDGET_V1,
    activate_budget_ledger,
    build_candidate_activation_contract,
    current_budget_ledger,
    write_json_atomic,
)
from .governance import EvidenceGatedSkillRegistry
from .literature_provider import OpenAlexArxivLiteratureProvider
from .skill_evolution import SkillEvolutionEngine


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class ExecutionManifest:
    run_id: str
    authorized: bool
    plan_sha256: str
    graph_content_sha256: str
    task_id: str
    result_root: str
    workspace: str
    backend: str = "local"
    model: str = "gpt-5.6-sol"
    provider: str = "CodexCLI"
    budget_profile: str = "matched-v1"
    budget_limits: dict[str, int | float] = field(default_factory=lambda: dict(MATCHED_BUDGET_V1))
    reproducibility: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=lambda: {
        "sample_seconds": 5,
        "warning_temperature_c": 82,
        "abort_temperature_c": 88,
        "abort_consecutive_samples": 3,
        "thermal_pacing_enabled": True,
        "thermal_pacing_start_c": 78,
        "thermal_pacing_resume_c": 72,
    })
    node_commands: dict[str, str] = field(default_factory=dict)
    node_resources: dict[str, dict[str, Any]] = field(default_factory=dict)
    target_files: tuple[str, ...] = ()
    task_description: str = ""
    allow_network: bool = True
    max_parallel_cpu: int = 3
    max_node_attempts: int = 3

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        plan: dict[str, Any],
        graph: dict[str, Any],
        result_root: Path,
        workspace: Path,
        authorized: bool = False,
        **overrides: Any,
    ) -> "ExecutionManifest":
        """Create a hash-bound manifest; execution still requires authorization."""
        return cls(
            run_id=run_id,
            authorized=authorized,
            plan_sha256=_sha_payload(plan),
            graph_content_sha256=str(graph.get("content_sha256", "")),
            task_id=str(plan.get("task_id", "")),
            result_root=str(result_root.expanduser().resolve()),
            workspace=str(workspace.expanduser().resolve()),
            **overrides,
        )

    @classmethod
    def load(cls, path: Path, *, plan: dict[str, Any], graph: dict[str, Any]) -> "ExecutionManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "ml-scientist-execution-manifest-v1":
            raise ValueError("Unsupported execution manifest schema")
        payload = {key: value for key, value in payload.items() if key != "schema_version"}
        if "target_files" in payload:
            payload["target_files"] = tuple(payload["target_files"])
        manifest = cls(**payload)
        if not manifest.authorized:
            raise ValueError("Execution manifest is not authorized")
        if manifest.plan_sha256 != _sha_payload(plan):
            raise ValueError("Execution manifest plan hash mismatch")
        if manifest.graph_content_sha256 != graph.get("content_sha256"):
            raise ValueError("Execution manifest graph hash mismatch")
        if manifest.task_id != plan.get("task_id"):
            raise ValueError("Execution manifest task mismatch")
        if manifest.backend not in {"local", "ssh"}:
            raise ValueError("Execution manifest backend must be local or ssh")
        result_root = Path(manifest.result_root).expanduser().resolve()
        workspace = Path(manifest.workspace).expanduser().resolve()
        if str(result_root) in {"/", str(Path.home())} or str(workspace) == "/":
            raise ValueError("Unsafe execution manifest path")
        if manifest.budget_profile == "matched-v1":
            missing = sorted(set(MATCHED_BUDGET_V1) - set(manifest.budget_limits))
            if missing:
                raise ValueError(f"Matched budget manifest is missing limits: {missing}")
        return manifest

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = "ml-scientist-execution-manifest-v1"
        return payload


@dataclass(frozen=True)
class NodeExecutionContext:
    run_id: str
    node: dict[str, Any]
    plan: dict[str, Any]
    graph: dict[str, Any]
    manifest: ExecutionManifest
    decision: dict[str, Any] | None
    visible_evidence: tuple[dict[str, Any], ...]
    output_dir: Path
    protected_test_exposed: bool


@dataclass
class NodeExecutionResult:
    node_id: str
    outcome: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    telemetry: dict[str, Any] = field(default_factory=dict)
    state_observations: dict[str, Any] = field(default_factory=dict)
    policy_evaluations: list[dict[str, Any]] | None = None
    review_issues: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "ml-scientist-node-execution-result-v1", **asdict(self)}


class NodeExecutor(Protocol):
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult: ...


class ArtifactVerifier:
    @staticmethod
    def descriptors(output_dir: Path, required: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
        artifacts: list[dict[str, Any]] = []
        errors: list[str] = []
        for relative in required:
            path = (output_dir / relative).resolve()
            if output_dir.resolve() not in path.parents:
                errors.append(f"unsafe artifact path: {relative}")
                continue
            if not path.is_file():
                errors.append(f"missing artifact: {relative}")
                continue
            if path.suffix == ".json":
                try:
                    json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError) as exc:
                    errors.append(f"invalid JSON artifact {relative}: {exc}")
                    continue
            artifacts.append({
                "path": str(path),
                "relative_path": relative,
                "sha256": _sha_file(path),
                "size_bytes": path.stat().st_size,
            })
        return artifacts, errors

    @classmethod
    def finalize(cls, context: NodeExecutionContext, result: NodeExecutionResult) -> NodeExecutionResult:
        required = list(context.node.get("output_artifacts", []))
        descriptors, errors = cls.descriptors(context.output_dir, required)
        result.artifacts = descriptors
        if errors:
            result.outcome = "FAILED_GATE"
            result.error = "; ".join(errors)
        manifest = {
            "schema_version": "ml-scientist-node-artifact-manifest-v1",
            "run_id": context.run_id,
            "node_id": context.node["node_id"],
            "plan_sha256": context.manifest.plan_sha256,
            "graph_content_sha256": context.manifest.graph_content_sha256,
            "artifacts": descriptors,
            "verification_errors": errors,
            "passed": not errors,
        }
        write_json_atomic(context.output_dir / "artifact_manifest.json", manifest)
        result.artifacts.append({
            "path": str((context.output_dir / "artifact_manifest.json").resolve()),
            "relative_path": "artifact_manifest.json",
            "sha256": _sha_file(context.output_dir / "artifact_manifest.json"),
            "size_bytes": (context.output_dir / "artifact_manifest.json").stat().st_size,
        })
        return result


class ContractGovernanceExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "ml-scientist-governance-artifact-v1",
            "run_id": context.run_id,
            "node_id": context.node["node_id"],
            "task_id": context.plan["task_id"],
            "plan_sha256": context.manifest.plan_sha256,
            "graph_content_sha256": context.graph.get("content_sha256"),
            "dependency_artifacts": [row.get("path") for row in context.visible_evidence if row.get("path")],
            "protected_test_exposed": context.protected_test_exposed,
            "claim_boundary": "contract and provenance evidence only; no unexecuted empirical claim",
        }
        for relative in context.node.get("output_artifacts", []):
            path = context.output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                write_json_atomic(path, payload)
            else:
                path.write_text(
                    f"# {context.node['title']}\n\n"
                    f"Run: `{context.run_id}`\n\n"
                    "This artifact records a completed governance transition. It does not create empirical evidence.\n",
                    encoding="utf-8",
                )
        return NodeExecutionResult(context.node["node_id"], "PASSED_GATE")


class LiteratureNodeExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        if not context.manifest.allow_network:
            return NodeExecutionResult(
                context.node["node_id"], "FAILED_GATE", error="Literature network access is disabled"
            )
        context.output_dir.mkdir(parents=True, exist_ok=True)
        query = context.plan.get("research_request") or context.plan.get("research_context", {}).get("task_description") or context.plan["task_id"]
        try:
            records = OpenAlexArxivLiteratureProvider().search(query, limit=8)
        except Exception as exc:
            return NodeExecutionResult(context.node["node_id"], "FAILED_GATE", error=str(exc))
        evidence = {
            "schema_version": "ml-scientist-literature-evidence-v1",
            "query": query,
            "records": records,
            "source_verification_required": True,
            "record_count": len(records),
        }
        write_json_atomic(context.output_dir / "literature_evidence.json", evidence)
        write_json_atomic(
            context.output_dir / "literature_sources.json",
            {"schema_version": "ml-scientist-source-manifest-v1", "sources": records},
        )
        notes = ["# Related work notes", ""]
        for row in records:
            notes.extend([
                f"## {row['title']}",
                f"- Source: {row['url']}",
                f"- Year: {row.get('year')}",
                f"- Evidence boundary: metadata/abstract record; full-paper claims require direct paper inspection.",
                "",
            ])
        (context.output_dir / "related_work_notes.md").write_text("\n".join(notes), encoding="utf-8")
        return NodeExecutionResult(context.node["node_id"], "PASSED_GATE", state_observations={"source_count": len(records)})


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        payload = json.loads(match.group(0))
        if isinstance(payload, dict):
            return payload
    raise ValueError("Codex node response was not a JSON object")


class CodexArtifactExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        context.output_dir.mkdir(parents=True, exist_ok=True)
        instruction = ((context.decision or {}).get("operator_contract") or {}).get("instruction", "")
        required = list(context.node.get("output_artifacts", []))
        prompt = (
            instruction
            + "\n\n## Node artifact contract\nReturn JSON only with an `artifacts` object. "
            "Every key must be one of these exact relative paths: "
            + json.dumps(required, ensure_ascii=False)
            + ". Values are complete file contents; JSON artifact values may be JSON objects. "
            "For review nodes, optionally include `review_issues`, each with issue_id and one issue_type from "
            "wording, citation, claim_evidence, statistics, experimental_design, insufficient_replication, implementation, metric_semantics."
        )
        started = time.monotonic()
        try:
            client, model = create_client(context.manifest.model, context.manifest.provider)
            text, _, usage = get_response_from_llm(
                prompt,
                client=client,
                model=model,
                system_message="You execute one evidence-gated research DAG node. Never invent results or citations.",
                budget_category="review" if "review" in context.node["node_type"] or "integrity" in context.node["node_type"] else "proposal",
            )
            payload = _extract_json_object(text)
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, dict):
                raise ValueError("Codex node response omitted artifacts object")
            for relative in required:
                if relative not in artifacts:
                    raise ValueError(f"Codex node response omitted {relative}")
                path = context.output_dir / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                value = artifacts[relative]
                if path.suffix == ".json":
                    if not isinstance(value, (dict, list)):
                        value = json.loads(str(value))
                    write_json_atomic(path, value)
                else:
                    path.write_text(str(value), encoding="utf-8")
            issues = payload.get("review_issues", [])
            if not isinstance(issues, list):
                issues = []
            return NodeExecutionResult(
                context.node["node_id"],
                "PASSED_GATE",
                telemetry={"wall_clock_seconds": time.monotonic() - started, "token_usage": usage},
                review_issues=[row for row in issues if isinstance(row, dict)],
            )
        except Exception as exc:
            return NodeExecutionResult(
                context.node["node_id"], "FAILED_GATE",
                telemetry={"wall_clock_seconds": time.monotonic() - started}, error=str(exc),
            )


class CodeModificationNodeExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        workspace = Path(context.manifest.workspace).resolve()
        targets = [workspace / relative for relative in context.manifest.target_files]
        if not targets or any(not path.is_file() for path in targets):
            return NodeExecutionResult(context.node["node_id"], "FAILED_GATE", error="Code targets are missing")
        before = {str(path.relative_to(workspace)): path.read_text(encoding="utf-8") for path in targets}
        action = ((context.decision or {}).get("operator_contract") or {}).get("instruction", "")
        editor = CodeEditor(
            model=context.manifest.model,
            provider=context.manifest.provider,
            target_files=[str(path) for path in targets],
            task_description=context.manifest.task_description or context.plan.get("research_request", ""),
            log_dir=str(context.output_dir / "editor_logs"),
        )
        result = editor.edit(action)
        after = {str(path.relative_to(workspace)): path.read_text(encoding="utf-8") for path in targets}
        activation = build_candidate_activation_contract(
            before=before,
            after=after,
            run_id=context.run_id,
            configuration={"task_id": context.plan["task_id"], "node_id": context.node["node_id"]},
        )
        context.output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            context.output_dir / "candidate_code_snapshots.json",
            {
                "schema_version": "ml-scientist-candidate-code-snapshots-v1",
                "edit_success": result.success,
                "files_changed": result.files_changed,
                "before_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in before.items()},
                "after_sha256": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in after.items()},
                "activation_contract": activation,
            },
        )
        return NodeExecutionResult(
            context.node["node_id"],
            "PASSED_GATE" if result.success and activation["passed_static_contract"] else "FAILED_GATE",
            telemetry={"token_usage": result.token_usage},
            state_observations={"activation_contract": activation},
            error=result.error if not result.success else None,
        )


class CommandNodeExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        command = context.manifest.node_commands.get(context.node["node_id"])
        if not command:
            return NodeExecutionResult(
                context.node["node_id"], "FAILED_GATE",
                error=f"No frozen node command for {context.node['node_id']}",
            )
        context.output_dir.mkdir(parents=True, exist_ok=True)
        formatted = command.format(
            output_dir=str(context.output_dir.resolve()),
            workspace=str(Path(context.manifest.workspace).resolve()),
            run_id=context.run_id,
        )
        started = time.monotonic()
        wall_limit = context.manifest.budget_limits.get("wall_clock_seconds")
        active_ledger = current_budget_ledger()
        remaining_wall = (
            None if wall_limit is None else float(wall_limit)
            - (active_ledger.elapsed_seconds() if active_ledger is not None else 0.0)
        )
        timeout = None if remaining_wall is None else max(0.001, remaining_wall)
        try:
            completed = subprocess.run(
                ["bash", "-lc", formatted],
                cwd=Path(context.manifest.workspace).resolve(),
                text=True,
                capture_output=True,
                check=False,
                timeout=timeout,
            )
            stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = str(exc.stdout or "")
            stderr = str(exc.stderr or "") + f"\nNODE_WALL_TIME_EXHAUSTED after {timeout} seconds"
            returncode = 124
        (context.output_dir / "command.stdout.log").write_text(stdout, encoding="utf-8")
        (context.output_dir / "command.stderr.log").write_text(stderr, encoding="utf-8")
        return NodeExecutionResult(
            context.node["node_id"],
            "PASSED_GATE" if returncode == 0 else "FAILED_GATE",
            telemetry={"wall_clock_seconds": time.monotonic() - started, "returncode": returncode},
            error=None if returncode == 0 else stderr[-3000:],
        )


class SkillEvolutionNodeExecutor:
    def execute(self, context: NodeExecutionContext) -> NodeExecutionResult:
        episodes = Path(context.manifest.workspace) / "adaptive_episodes.jsonl"
        if not episodes.is_file():
            return NodeExecutionResult(context.node["node_id"], "FAILED_GATE", error=f"Missing episodes: {episodes}")
        registry = EvidenceGatedSkillRegistry(
            Path(context.manifest.result_root) / "governance",
            context.graph.get("repository_commit", "unknown"),
        )
        report = SkillEvolutionEngine(
            registry=registry,
            out_dir=context.output_dir,
            knowledge_graph=context.graph,
        ).evolve(episodes)
        return NodeExecutionResult(
            context.node["node_id"], "PASSED_GATE",
            state_observations={
                "accepted_episode_count": len(report["accepted_episodes"]),
                "write_action_count": len(report["write_actions"]),
            },
        )


class NodeExecutorRegistry:
    def __init__(self) -> None:
        self.governance = ContractGovernanceExecutor()
        self.literature = LiteratureNodeExecutor()
        self.codex = CodexArtifactExecutor()
        self.code = CodeModificationNodeExecutor()
        self.command = CommandNodeExecutor()
        self.skill = SkillEvolutionNodeExecutor()

    def resolve(self, node: dict[str, Any]) -> NodeExecutor:
        if node["node_id"] == "literature-frontier":
            return self.literature
        if node["node_id"] == "skill-evolution":
            return self.skill
        if node["node_id"] == "paper-finalize":
            return self.command
        if node["node_type"] in {"governance", "contract"}:
            return self.governance
        if node["node_type"] == "code_modification":
            return self.code
        if node["node_type"] in {"baseline_reproduction", "experiment", "replication", "ablation", "statistical_analysis"}:
            return self.command
        return self.codex


class FullDagRunner:
    def __init__(
        self,
        *,
        runtime: AdaptiveResearchRuntime,
        manifest: ExecutionManifest,
        registry: NodeExecutorRegistry | None = None,
    ):
        self.runtime = runtime
        self.manifest = manifest
        self.registry = registry or NodeExecutorRegistry()
        self.result_root = Path(manifest.result_root).expanduser().resolve() / manifest.run_id
        self.result_root.mkdir(parents=True, exist_ok=True)
        self.ledger = BudgetLedger(
            profile_name=manifest.budget_profile,
            limits=dict(manifest.budget_limits),
        ) if manifest.budget_profile == "matched-v1" else BudgetLedger.legacy_unbounded()

    def _visible_evidence(self) -> list[dict[str, Any]]:
        rows = []
        for outcome in self.runtime.state.get("outcomes", []):
            for artifact in outcome.get("evidence_artifacts", []):
                rows.append({**artifact, "observability": "POST_STAGE_VISIBLE"})
        return rows

    def _context(self, node_id: str, decision: dict[str, Any] | None) -> NodeExecutionContext:
        return NodeExecutionContext(
            run_id=self.manifest.run_id,
            node=self.runtime.nodes[node_id],
            plan=self.runtime.plan,
            graph=self.runtime.graph,
            manifest=self.manifest,
            decision=decision,
            visible_evidence=tuple(self._visible_evidence()),
            output_dir=self.result_root / node_id / f"attempt-{self.runtime.state['attempts'][node_id] + 1:03d}",
            protected_test_exposed=any(
                row.get("node_id") == "protected-final-test" and row.get("outcome") in {"PASSED_GATE", "COMPLETE"}
                for row in self.runtime.state.get("outcomes", [])
            ),
        )

    def _decision(self, node_id: str) -> dict[str, Any] | None:
        node = self.runtime.nodes[node_id]
        if node.get("metadata", {}).get("controller_routing_allowed") is False:
            return None
        snapshot = self.ledger.snapshot()
        fractions: list[float] = []
        for dimension, limit in snapshot["limits"].items():
            if limit in {None, 0}:
                continue
            used_key = "total_tokens" if dimension == "token_budget" else dimension
            used = float(snapshot["usage"].get(used_key, 0))
            fractions.append(max(0.0, 1.0 - used / float(limit)))
        remaining_fraction = min(fractions) if fractions else 1.0
        return self.runtime.choose_operator(
            node_id,
            evidence=self._visible_evidence(),
            budget_before={**snapshot["usage"], "remaining_fraction": remaining_fraction},
            stage_complete=False,
        )

    def _execute_one(self, node_id: str, decision: dict[str, Any] | None) -> NodeExecutionResult:
        context = self._context(node_id, decision)
        with activate_budget_ledger(self.ledger):
            result = self.registry.resolve(context.node).execute(context)
        return ArtifactVerifier.finalize(context, result)

    @staticmethod
    def _fallback_policy_evaluations(decision: dict[str, Any] | None) -> list[dict[str, Any]] | None:
        if not decision or not decision.get("selected_policy_by_role"):
            return None
        return [
            {
                "policy_id": policy_id,
                "role": role,
                "applied": False,
                "status": "INCONCLUSIVE",
                "evidence_artifact_refs": [],
                "gate_checks": ["full_dag_executor_requires_post_execution_role_audit"],
                "learning_signal": {"intent": "NONE", "reason": "no role-specific reusable evidence"},
            }
            for role, policy_id in decision["selected_policy_by_role"].items()
        ]

    def _record(self, result: NodeExecutionResult, decision: dict[str, Any] | None) -> None:
        policy_evaluations = result.policy_evaluations or self._fallback_policy_evaluations(decision)
        self.runtime.record_outcome(
            result.node_id,
            outcome=result.outcome,
            evidence_artifacts=result.artifacts,
            budget_remaining=self.ledger.passed(),
            policy_evaluations=policy_evaluations,
            budget_after=self.ledger.snapshot()["usage"],
            telemetry=result.telemetry,
            state_observations=result.state_observations,
        )
        for issue in result.review_issues:
            if not issue.get("issue_id") or not issue.get("issue_type"):
                continue
            self.runtime.apply_review_issue(
                issue_id=str(issue["issue_id"]),
                issue_type=str(issue["issue_type"]),
                source_paper_node=result.node_id,
                protected_test_exposed=any(
                    row.get("node_id") == "protected-final-test" and row.get("outcome") in {"PASSED_GATE", "COMPLETE"}
                    for row in self.runtime.state.get("outcomes", [])
                ),
            )

    def _is_gpu(self, node_id: str) -> bool:
        override = self.manifest.node_resources.get(node_id, {}).get("gpu")
        if override is not None:
            return bool(override)
        return self.runtime.nodes[node_id]["node_type"] in {
            "baseline_reproduction", "experiment", "replication", "ablation"
        }

    def run(self, *, max_nodes: int | None = None) -> dict[str, Any]:
        executed = 0
        results: list[dict[str, Any]] = []
        with activate_budget_ledger(self.ledger):
            while self.runtime.ready_nodes() and (max_nodes is None or executed < max_nodes):
                ready = [
                    node_id for node_id in self.runtime.ready_nodes()
                    if self.runtime.state["attempts"].get(node_id, 0) < self.manifest.max_node_attempts
                ]
                if not ready:
                    break
                remaining = None if max_nodes is None else max_nodes - executed
                if remaining is not None:
                    ready = ready[:remaining]
                decisions: dict[str, dict[str, Any] | None] = {}
                decision_failures: dict[str, NodeExecutionResult] = {}
                for node_id in ready:
                    try:
                        decisions[node_id] = self._decision(node_id)
                    except Exception as exc:
                        decisions[node_id] = None
                        context = self._context(node_id, None)
                        decision_failures[node_id] = ArtifactVerifier.finalize(
                            context,
                            NodeExecutionResult(
                                node_id,
                                "FAILED_GATE",
                                error=f"POLICY_SELECTION_FAILED: {exc}",
                            ),
                        )
                cpu_nodes = [node_id for node_id in ready if not self._is_gpu(node_id)]
                gpu_nodes = [node_id for node_id in ready if self._is_gpu(node_id)]

                if cpu_nodes:
                    with ThreadPoolExecutor(max_workers=max(1, self.manifest.max_parallel_cpu)) as pool:
                        futures = {
                            pool.submit(self._execute_one, node_id, decisions[node_id]): node_id
                            for node_id in cpu_nodes if node_id not in decision_failures
                        }
                        completed: dict[str, NodeExecutionResult] = {}
                        for future in as_completed(futures):
                            node_id = futures[future]
                            try:
                                completed[node_id] = future.result()
                            except Exception as exc:
                                context = self._context(node_id, decisions[node_id])
                                completed[node_id] = ArtifactVerifier.finalize(
                                    context,
                                    NodeExecutionResult(node_id, "FAILED_GATE", error=str(exc)),
                                )
                        for node_id in cpu_nodes:
                            result = decision_failures.get(node_id) or completed[node_id]
                            self._record(result, decisions[node_id])
                            results.append(result.to_dict())
                            executed += 1

                for node_id in gpu_nodes:
                    if node_id in decision_failures:
                        result = decision_failures[node_id]
                    else:
                        try:
                            result = self._execute_one(node_id, decisions[node_id])
                        except Exception as exc:
                            context = self._context(node_id, decisions[node_id])
                            result = ArtifactVerifier.finalize(
                                context,
                                NodeExecutionResult(node_id, "FAILED_GATE", error=str(exc)),
                            )
                    self._record(result, decisions[node_id])
                    results.append(result.to_dict())
                    executed += 1

                if not self.ledger.passed():
                    break

        report = {
            "schema_version": "ml-scientist-full-dag-run-v1",
            "run_id": self.manifest.run_id,
            "executed_node_count": executed,
            "results": results,
            "ready_nodes": self.runtime.ready_nodes(),
            "node_status": dict(self.runtime.state["node_status"]),
            "budget_ledger": self.ledger.snapshot(),
            "complete": self.runtime.state["node_status"].get("skill-evolution") == "COMPLETE",
            "protected_metrics_used_for_routing": False,
        }
        write_json_atomic(self.result_root / "full_dag_run.json", report)
        return report
