"""Shared execution, attribution, reproducibility, and budget contracts.

The classes in this module are deliberately independent from any one FML
agent.  A single ledger is activated around an agent run and is consulted by
the LLM and benchmark executor boundaries.  The same evidence shape is used
for local, SSH, legacy-baseline, and adaptive execution paths.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
import platform
import sys
import time
import threading
import importlib.metadata
import yaml
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator


MATCHED_BUDGET_V1: dict[str, int | float] = {
    "token_budget": 120_000,
    "wall_clock_seconds": 7_200.0,
    "proposal_count": 8,
    "review_count": 8,
    "candidate_validation_count": 1,
    "pre_test_validation_count": 1,
    "protected_test_count": 1,
}


class BudgetExceeded(RuntimeError):
    """Raised before starting work that would exceed a frozen resource cap."""


@dataclass
class BudgetLedger:
    profile_name: str = "matched-v1"
    limits: dict[str, int | float | None] = field(
        default_factory=lambda: dict(MATCHED_BUDGET_V1)
    )
    counters: dict[str, int | float] = field(default_factory=lambda: {
        "total_tokens": 0,
        "llm_calls": 0,
        "proposal_count": 0,
        "review_count": 0,
        "edit_count": 0,
        "candidate_validation_count": 0,
        "pre_test_validation_count": 0,
        "protected_test_count": 0,
        "gpu_active_seconds": 0.0,
    })
    started_monotonic: float = field(default_factory=time.monotonic, repr=False)
    exhausted_dimension: str | None = None
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    @classmethod
    def legacy_unbounded(cls) -> "BudgetLedger":
        return cls(
            profile_name="legacy-unbounded",
            limits={key: None for key in MATCHED_BUDGET_V1},
        )

    def elapsed_seconds(self) -> float:
        return max(0.0, time.monotonic() - self.started_monotonic)

    def _limit(self, dimension: str) -> float | None:
        value = self.limits.get(dimension)
        return None if value is None else float(value)

    def _used(self, dimension: str) -> float:
        if dimension == "token_budget":
            return float(self.counters.get("total_tokens", 0))
        if dimension == "wall_clock_seconds":
            return self.elapsed_seconds()
        return float(self.counters.get(dimension, 0))

    def remaining(self, dimension: str) -> float | None:
        """Return the live remaining allowance, or None for an unbounded dimension."""
        with self._lock:
            limit = self._limit(dimension)
            if limit is None:
                return None
            return max(0.0, limit - self._used(dimension))

    def assert_available(self, dimension: str, amount: int | float = 1) -> None:
        with self._lock:
            limit = self._limit(dimension)
            if limit is None:
                return
            if self._used(dimension) + float(amount) > limit:
                self.exhausted_dimension = dimension
                self.events.append({
                    "event": "BUDGET_EXHAUSTED",
                    "dimension": dimension,
                    "used": self._used(dimension),
                    "requested": float(amount),
                    "limit": limit,
                })
                raise BudgetExceeded(
                    f"Budget exhausted for {dimension}: used={self._used(dimension)}, "
                    f"requested={amount}, limit={limit}"
                )

    def record_llm(self, usage: dict[str, Any] | None, category: str) -> None:
        total = 0
        if usage:
            for key in ("total_tokens", "input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
                value = usage.get(key)
                if key == "total_tokens" and isinstance(value, (int, float)):
                    total = int(value)
                    break
            if not total:
                total = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0) + int(
                    usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
                )
        with self._lock:
            self.counters["total_tokens"] += total
            self.events.append({"event": "LLM_CALL_COMPLETED", "category": category, "tokens": total})
            limit = self._limit("token_budget")
            if limit is not None and self._used("token_budget") > limit:
                self.exhausted_dimension = "token_budget"

    def before_llm(self, category: str, count: int = 1) -> None:
        with self._lock:
            self.assert_available("wall_clock_seconds", 0)
            self.assert_available("token_budget", 0)
            counter = {
                "review": "review_count",
                "edit": "edit_count",
            }.get(category, "proposal_count")
            if category != "edit":
                self.assert_available(counter, count)
            self.counters[counter] += count
            self.counters["llm_calls"] += 1
            self.events.append({"event": "LLM_CALL_STARTED", "category": category, "item_count": count})

    def before_execution(self, phase: str) -> None:
        dimension = {
            "val": "candidate_validation_count",
            "pre_test_val": "pre_test_validation_count",
            "test": "protected_test_count",
        }[phase]
        with self._lock:
            self.assert_available("wall_clock_seconds", 0)
            self.assert_available(dimension)
            self.counters[dimension] += 1
            self.events.append({"event": "EXECUTION_STARTED", "phase": phase})

    def add_gpu_active_seconds(self, seconds: float | int | None) -> None:
        if seconds is not None:
            with self._lock:
                self.counters["gpu_active_seconds"] += max(0.0, float(seconds))

    def passed(self) -> bool:
        if self.exhausted_dimension:
            return False
        for dimension in self.limits:
            limit = self._limit(dimension)
            if limit is not None and self._used(dimension) > limit:
                return False
        return True

    def snapshot(self) -> dict[str, Any]:
        usage = dict(self.counters)
        usage["wall_clock_seconds"] = self.elapsed_seconds()
        return {
            "schema_version": "fml-budget-ledger-v1",
            "profile_name": self.profile_name,
            "limits": dict(self.limits),
            "usage": usage,
            "passed": self.passed(),
            "exhausted_dimension": self.exhausted_dimension,
            "events": list(self.events),
        }


_ACTIVE_BUDGET: ContextVar[BudgetLedger | None] = ContextVar(
    "fml_active_budget_ledger", default=None
)


def current_budget_ledger() -> BudgetLedger | None:
    return _ACTIVE_BUDGET.get()


@contextmanager
def activate_budget_ledger(ledger: BudgetLedger) -> Iterator[BudgetLedger]:
    token = _ACTIVE_BUDGET.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE_BUDGET.reset(token)


def _sha_payload(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


class _DocstringStripper(ast.NodeTransformer):
    """Remove docstrings so documentation-only edits do not count as code changes."""

    @staticmethod
    def _without_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            return body[1:]
        return body

    def visit_Module(self, node: ast.Module) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        self.generic_visit(node)
        node.body = self._without_docstring(node.body)
        return node


def _python_semantic_dump(source: str) -> str:
    tree = ast.parse(source)
    tree = _DocstringStripper().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.dump(tree, include_attributes=False)


def _semantically_equivalent(path: str, before: str, after: str) -> bool:
    """Compare source meaning for formats with a deterministic parser."""
    suffix = Path(path).suffix.lower()
    try:
        if suffix == ".py":
            return _python_semantic_dump(before) == _python_semantic_dump(after)
        if suffix == ".json":
            return json.loads(before) == json.loads(after)
        if suffix in {".yaml", ".yml"}:
            return yaml.safe_load(before) == yaml.safe_load(after)
    except (SyntaxError, json.JSONDecodeError, yaml.YAMLError):
        # Invalid candidate content is handled by the format-specific contract;
        # it must never be silently classified as an equivalent no-op.
        return False
    return before == after


def _callables(source: str) -> dict[str, dict[str, Any]]:
    tree = ast.parse(source)
    result: dict[str, dict[str, Any]] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.parents: list[str] = []

        def _visit_callable(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
            qualified = ".".join([*self.parents, node.name])
            normalized_node = _DocstringStripper().visit(copy.deepcopy(node))
            segment = ast.dump(normalized_node, include_attributes=False)
            result[qualified] = {
                "symbol": qualified,
                "lineno": node.lineno,
                "end_lineno": getattr(node, "end_lineno", node.lineno),
                "sha256": hashlib.sha256(segment.encode()).hexdigest(),
            }
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
            self.parents.append(node.name)
            self.generic_visit(node)
            self.parents.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
            self._visit_callable(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
            self._visit_callable(node)

    Visitor().visit(tree)
    return result


def build_candidate_activation_contract(
    *,
    before: dict[str, str],
    after: dict[str, str],
    run_id: str,
    configuration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed activation contract from a candidate code diff."""
    changed_files = sorted(path for path in set(before) | set(after) if before.get(path) != after.get(path))
    semantic_noop_files = [
        path for path in changed_files
        if _semantically_equivalent(path, before.get(path, ""), after.get(path, ""))
    ]
    semantic_changed_files = [path for path in changed_files if path not in semantic_noop_files]
    changed_symbols: list[dict[str, Any]] = []
    syntax_errors: list[str] = []
    module_assertions: list[dict[str, Any]] = []
    for path in semantic_changed_files:
        if Path(path).suffix.lower() != ".py":
            module_assertions.append({
                "path": path,
                "candidate_file_sha256": hashlib.sha256(after.get(path, "").encode()).hexdigest(),
                "mandatory": True,
                "kind": "MODULE_OR_CONFIGURATION_VALUE_ASSERTION",
            })
            continue
        try:
            prior = _callables(before.get(path, "")) if before.get(path, "").strip() else {}
            current = _callables(after.get(path, "")) if after.get(path, "").strip() else {}
        except SyntaxError as exc:
            syntax_errors.append(f"{path}:{exc.lineno}:{exc.msg}")
            continue
        for symbol, row in current.items():
            if symbol not in prior or prior[symbol]["sha256"] != row["sha256"]:
                changed_symbols.append({"path": path, **row, "new": symbol not in prior, "mandatory": True})
        if not any(row["path"] == path for row in changed_symbols):
            module_assertions.append({
                "path": path,
                "candidate_file_sha256": hashlib.sha256(after.get(path, "").encode()).hexdigest(),
                "mandatory": True,
                "kind": "MODULE_OR_CONFIGURATION_VALUE_ASSERTION",
            })
    called_names: set[str] = set()
    for source in after.values():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    called_names.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    called_names.add(node.func.attr)
    unreachable_new_symbols = [
        row for row in changed_symbols
        if row["new"] and row["symbol"].split(".")[-1] not in called_names
    ]
    candidate_hash = _sha_payload({path: after.get(path) for path in changed_files})
    config_hash = _sha_payload(configuration or {})
    passed_static_contract = (
        bool(semantic_changed_files)
        and not syntax_errors
        and not unreachable_new_symbols
        and bool(changed_symbols or module_assertions)
    )
    static_rejection_reason = None
    if not changed_files:
        static_rejection_reason = "INVALID_CANDIDATE_NO_CHANGES"
    elif not semantic_changed_files:
        static_rejection_reason = "INVALID_CANDIDATE_NO_SEMANTIC_CHANGE"
    elif syntax_errors:
        static_rejection_reason = "INVALID_CANDIDATE_SYNTAX_ERROR"
    elif unreachable_new_symbols:
        static_rejection_reason = "INVALID_EXECUTION_UNREACHABLE_CHANGE"
    elif not (changed_symbols or module_assertions):
        static_rejection_reason = "INVALID_EXECUTION_NO_ACTIVATION_TARGET"
    return {
        "schema_version": "fml-candidate-activation-contract-v2",
        "run_id": str(run_id),
        "candidate_sha256": candidate_hash,
        "configuration_sha256": config_hash,
        "changed_files": changed_files,
        "semantic_changed_files": semantic_changed_files,
        "semantic_noop_files": semantic_noop_files,
        "changed_symbols": changed_symbols,
        "mandatory_activation_targets": [
            {key: row[key] for key in ("path", "symbol", "lineno", "new")}
            for row in changed_symbols if row["mandatory"]
        ],
        "configuration_assertions": module_assertions,
        "called_names": sorted(called_names),
        "unreachable_new_symbols": unreachable_new_symbols,
        "syntax_errors": syntax_errors,
        "passed_static_contract": passed_static_contract,
        "static_rejection_reason": static_rejection_reason,
    }


def validate_runtime_markers(
    contract: dict[str, Any], markers: list[dict[str, Any]]
) -> dict[str, Any]:
    valid_markers = [
        row for row in markers
        if row.get("run_id") == contract.get("run_id")
        and row.get("candidate_sha256") == contract.get("candidate_sha256")
        and row.get("configuration_sha256") == contract.get("configuration_sha256")
    ]
    observed = {(row.get("path"), row.get("symbol")) for row in valid_markers}
    missing = [
        row for row in contract.get("mandatory_activation_targets", [])
        if (row.get("path"), row.get("symbol")) not in observed
    ]
    assertion_observed = {row.get("path") for row in valid_markers if row.get("kind") == "CONFIGURATION_ASSERTION"}
    missing_assertions = [
        row for row in contract.get("configuration_assertions", [])
        if row.get("path") not in assertion_observed
    ]
    return {
        "schema_version": "fml-runtime-activation-validation-v1",
        "passed": bool(valid_markers) and not missing and not missing_assertions,
        "valid_marker_count": len(valid_markers),
        "missing_activation_targets": missing,
        "missing_configuration_assertions": missing_assertions,
        "candidate_sha256": contract.get("candidate_sha256"),
        "configuration_sha256": contract.get("configuration_sha256"),
    }


@dataclass(frozen=True)
class ReproducibilityContract:
    mode: str = "repeat_guarded"
    metric_absolute_tolerance: float = 0.01
    require_same_constraint_classification: bool = True
    independent_replication_seeds: int = 3
    no_favourable_tie_breaker: bool = True
    deterministic_environment: dict[str, str] = field(default_factory=lambda: {
        "PYTHONHASHSEED": "frozen_experimental_seed",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "CUDNN_DETERMINISTIC": "1",
        "CUDNN_BENCHMARK": "0",
    })

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": "fml-reproducibility-contract-v1", **asdict(self)}


def classify_validation(result: dict[str, Any] | None) -> str:
    if not result:
        return "MISSING"
    if result.get("success") and result.get("primary_metric") is not None:
        return "PASSED"
    error = str(result.get("error") or "").lower()
    if "constraint" in error or "invalid results" in error:
        return "CONSTRAINT_FAILED"
    return "EXECUTION_FAILED"


def compare_repeat_validations(
    selected: dict[str, Any] | None,
    repeated: dict[str, Any] | None,
    contract: ReproducibilityContract,
) -> dict[str, Any]:
    selected_class = classify_validation(selected)
    repeated_class = classify_validation(repeated)
    selected_metric = (selected or {}).get("primary_metric")
    repeated_metric = (repeated or {}).get("primary_metric")
    delta: float | None = None
    if selected_metric is not None and repeated_metric is not None:
        delta = abs(float(selected_metric) - float(repeated_metric))
    classification_pass = (
        not contract.require_same_constraint_classification
        or selected_class == repeated_class
    )
    metric_pass = delta is not None and delta <= contract.metric_absolute_tolerance
    passed = selected_class == "PASSED" and repeated_class == "PASSED" and classification_pass and metric_pass
    return {
        "schema_version": "fml-repeat-validation-comparison-v1",
        "status": "CONSISTENT" if passed else "STOCHASTIC_UNCERTAIN",
        "passed": passed,
        "selected_classification": selected_class,
        "repeat_classification": repeated_class,
        "selected_metric": selected_metric,
        "repeat_metric": repeated_metric,
        "absolute_delta": delta,
        "absolute_tolerance": contract.metric_absolute_tolerance,
        "protected_test_allowed": passed,
        "no_favourable_tie_breaker": True,
    }


def reproducibility_environment(seed: int | str | None) -> dict[str, str]:
    resolved = str(0 if seed is None else seed)
    return {
        "PYTHONHASHSEED": resolved,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "FML_EXPERIMENTAL_SEED": resolved,
        "FML_CUDNN_DETERMINISTIC": "1",
        "FML_CUDNN_BENCHMARK": "0",
    }


def capture_reproducibility_evidence(
    seed: int | str | None,
    *,
    execution_backend: str,
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("numpy", "torch", "PyYAML", "openai", "backoff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED"
    torch_state: dict[str, Any] = {}
    try:
        import torch
        torch_state = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
    except Exception as exc:
        torch_state = {"status": "UNAVAILABLE", "detail": str(exc)}
    return {
        "schema_version": "fml-reproducibility-evidence-v1",
        "execution_backend": execution_backend,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "torch_backend": torch_state,
        "seed_sources": {
            "experimental_seed": seed,
            "pythonhashseed": str(0 if seed is None else seed),
            "provider_seed_environment": "FML_EXPERIMENT_SEED",
        },
        "deterministic_environment": reproducibility_environment(seed),
    }


def normalize_summary_contract(summary: dict[str, Any]) -> dict[str, Any]:
    """Read v1 summaries without pretending they contain v2 resource evidence."""
    normalized = dict(summary)
    if normalized.get("schema_version") == "fml-summary-v2":
        normalized.setdefault("resource_accounting_status", "RESOURCE_ACCOUNTING_INCOMPLETE")
    else:
        normalized.setdefault("schema_version", "fml-summary-v1-legacy")
        normalized["resource_accounting_status"] = "RESOURCE_ACCOUNTING_INCOMPLETE"
    normalized.setdefault("budget_ledger", {})
    normalized.setdefault("execution_counts", {})
    normalized.setdefault("candidate_activation", [])
    normalized.setdefault("reproducibility", {})
    normalized.setdefault("gpu_telemetry", [])
    normalized["resource_matched_comparable"] = (
        normalized["resource_accounting_status"] == "COMPLETE_V2"
        and bool(normalized["budget_ledger"])
        and bool(normalized["execution_counts"])
    )
    return normalized


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
