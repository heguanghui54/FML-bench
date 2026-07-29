"""Shared contracts for the unified research-and-paper task graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class NodeStatus(str, Enum):
    LOCKED = "LOCKED_DEPENDENCIES"
    READY = "READY"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    INVALID_EXECUTION = "INVALID_EXECUTION"
    VALID_REGRESSION = "VALID_REGRESSION"
    STOCHASTIC_UNCERTAIN = "STOCHASTIC_UNCERTAIN"
    VALID_NONPROMOTABLE = "VALID_NONPROMOTABLE"
    VALID_CANDIDATE = "VALID_CANDIDATE"
    PASSED = "PASSED_GATE"
    FAILED = "FAILED_GATE"
    SUPERSEDED = "SUPERSEDED"
    COMPLETE = "COMPLETE"
    BLOCKED_BUDGET = "BLOCKED_BUDGET"
    BLOCKED_HUMAN = "BLOCKED_HUMAN_REVIEW"


class MetricObservability(str, Enum):
    """When a metric may become visible to the adaptive controller."""

    ONLINE_VISIBLE = "ONLINE_VISIBLE"
    POST_STAGE_VISIBLE = "POST_STAGE_VISIBLE"
    PROTECTED_FINAL_ONLY = "PROTECTED_FINAL_ONLY"
    PAPER_REVIEW_ONLY = "PAPER_REVIEW_ONLY"
    OPERATOR_ONLY = "OPERATOR_ONLY"


class MemoryMaturity(str, Enum):
    OBSERVATION = "observation"
    VERIFIED = "verified"
    PROVISIONAL_SUCCESS = "provisional_success"
    REPEATED_SUCCESS = "repeated_success"
    SUPERSEDED = "superseded"
    CONTRADICTION = "contradiction"


@dataclass(frozen=True)
class MetricContract:
    metric_id: str
    name: str
    category: str
    direction: str | None
    observability: MetricObservability
    stages: tuple[str, ...]
    definition: str
    source: str
    maturity: MemoryMaturity = MemoryMaturity.OBSERVATION
    task_id: str | None = None
    split: str | None = None
    protected: bool = False
    claim_eligible: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["observability"] = self.observability.value
        payload["maturity"] = self.maturity.value
        return payload


@dataclass(frozen=True)
class KnowledgeNode:
    node_id: str
    node_type: str
    label: str
    maturity: str
    provenance: tuple[dict[str, Any], ...]
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class KnowledgeEdge:
    edge_id: str
    source: str
    target: str
    relation: str
    provenance: tuple[dict[str, Any], ...]
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ControllerDecision:
    decision_id: str
    run_id: str
    node_id: str
    stage: int
    snapshot_sha256: str
    visible_evidence: tuple[dict[str, Any], ...]
    candidate_strategy_ids: tuple[str, ...]
    selected_strategy_id: str
    rationale: tuple[str, ...]
    budget_before: dict[str, float | int]
    protected_metric_visible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PolicyCompositionDecision:
    decision_id: str
    run_id: str
    node_id: str
    stage: int
    snapshot_sha256: str
    visible_evidence: tuple[dict[str, Any], ...]
    candidate_policy_ids_by_role: dict[str, tuple[str, ...]]
    selected_policy_by_role: dict[str, str]
    selected_policy_ids: tuple[str, ...]
    rationale: tuple[str, ...]
    budget_before: dict[str, float | int]
    protected_metric_visible: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InnerLoopContract:
    operators: tuple[str, ...]
    reviewers: tuple[str, ...]
    hard_gates: tuple[str, ...]
    allowed_outcomes: tuple[str, ...]
    budget: dict[str, float | int]


@dataclass
class ResearchNode:
    node_id: str
    title: str
    node_type: str
    depends_on: list[str]
    output_artifacts: list[str]
    unlock_conditions: list[str]
    inner_loop: InnerLoopContract
    metadata: dict[str, Any] = field(default_factory=dict)
    planned_at_stage: int = 3
    status: NodeStatus = NodeStatus.LOCKED

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


@dataclass
class PipelinePlan:
    task_id: str
    stages: list[dict[str, Any]]
    nodes: list[ResearchNode]
    edges: list[dict[str, str]]
    memory_contract: dict[str, Any]
    protected_test_contract: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "fml-scientist-unified-dag-v1",
            "task_id": self.task_id,
            "stages": self.stages,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": self.edges,
            "memory_contract": self.memory_contract,
            "protected_test_contract": self.protected_test_contract,
        }
