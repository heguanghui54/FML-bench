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
