"""Learn versioned benchmark and metric contracts without leaking protected scores."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .contracts import MemoryMaturity, MetricContract, MetricObservability


PROCESS_OBSERVABILITY: dict[str, tuple[MetricObservability, tuple[str, ...]]] = {
    "Exploration Spread": (MetricObservability.ONLINE_VISIBLE, ("hypothesis", "search_trajectory")),
    "Exploration Uniqueness": (MetricObservability.ONLINE_VISIBLE, ("hypothesis", "search_trajectory")),
    "Exploration Reach": (MetricObservability.ONLINE_VISIBLE, ("hypothesis", "search_trajectory")),
    "Effective dim": (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory",)),
    "Val-test |gap|": (MetricObservability.PROTECTED_FINAL_ONLY, ("confirmatory_analysis", "paper_review")),
    "Valid step ratio": (MetricObservability.ONLINE_VISIBLE, ("code", "experiment", "search_trajectory")),
    "AUC-over-steps": (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory", "confirmatory_analysis")),
    "First-improvement step": (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory",)),
    "Late-gain fraction": (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory",)),
    "Best-improvement step": (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory",)),
    "Token cost (M)": (MetricObservability.ONLINE_VISIBLE, ("all_runtime_nodes",)),
    "Wall-clock time (h)": (MetricObservability.ONLINE_VISIBLE, ("all_runtime_nodes",)),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric_contracts_from_catalog(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    contracts: list[MetricContract] = []
    for metric in catalog["metrics"]:
        if metric["category"] == "task_performance":
            common = {
                "name": metric["name"],
                "category": metric["category"],
                "direction": metric.get("direction"),
                "definition": metric["definition"],
                "source": metric["source"],
                "maturity": MemoryMaturity.VERIFIED,
                "task_id": metric.get("task_id"),
                "metadata": {
                    "dataset": metric.get("dataset"),
                    "baseline_validation": metric.get("baseline_validation"),
                    "baseline_test": metric.get("baseline_test"),
                },
            }
            contracts.append(
                MetricContract(
                    metric_id=metric["metric_id"] + "::validation",
                    observability=MetricObservability.ONLINE_VISIBLE,
                    stages=("baseline_reproduction", "experiment", "replication", "ablation"),
                    split="validation",
                    protected=False,
                    claim_eligible=False,
                    **common,
                )
            )
            contracts.append(
                MetricContract(
                    metric_id=metric["metric_id"] + "::protected_test",
                    observability=MetricObservability.PROTECTED_FINAL_ONLY,
                    stages=("protected_final_test", "confirmatory_analysis", "paper_writing", "paper_review"),
                    split="test",
                    protected=True,
                    claim_eligible=True,
                    **common,
                )
            )
            continue
        observability, stages = PROCESS_OBSERVABILITY.get(
            metric["name"],
            (MetricObservability.POST_STAGE_VISIBLE, ("search_trajectory",)),
        )
        contracts.append(
            MetricContract(
                metric_id=metric["metric_id"],
                name=metric["name"],
                category=metric["category"],
                direction=metric.get("direction"),
                observability=observability,
                stages=stages,
                definition=metric["definition"],
                source=metric["source"],
                maturity=MemoryMaturity.VERIFIED,
                protected=observability is MetricObservability.PROTECTED_FINAL_ONLY,
                claim_eligible=observability in {
                    MetricObservability.POST_STAGE_VISIBLE,
                    MetricObservability.PROTECTED_FINAL_ONLY,
                },
            )
        )
    return [contract.to_dict() for contract in contracts]


def build_fml_benchmark_contract(catalog: dict[str, Any]) -> dict[str, Any]:
    """Create the source-audited FML contract used by planning and evaluation."""
    contract = {
        "schema_version": "ml-scientist-benchmark-contract-v1",
        "benchmark_id": "fml-bench",
        "benchmark_version": catalog["repository_commit"],
        "source_kind": "checked_out_repository",
        "maturity": "verified",
        "activation_eligible": True,
        "tasks": [
            {
                "task_id": task["task_id"],
                "domain": task["domain"],
                "dataset": task["canonical_dataset"],
                "editable_target_files": task["target_files"],
                "validation_command": task["validation_command"],
                "protected_test_command": task["protected_test_command"],
                "canonical_metric": task["canonical_metric"],
                "direction": task["canonical_direction"],
                "config_sha256": task["config_sha256"],
                "task_config_sha256": task["task_config_sha256"],
                "baseline_validation_sha256": task["baseline_validation_sha256"],
                "baseline_test_sha256": task["baseline_test_sha256"],
            }
            for task in catalog["tasks"]
        ],
        "metrics": _metric_contracts_from_catalog(catalog),
        "evaluation_boundary": catalog["evaluation_boundary"],
        "activation_gates": [
            "unique_metric_ids",
            "known_observability",
            "task_validation_and_test_views_separated",
            "protected_metrics_never_online",
            "task_commands_and_source_hashes_present",
        ],
        "learning_rule": "New or changed metrics are observations until their implementation, direction, split boundary, and boundary tests are verified for a future arm.",
    }
    report = validate_benchmark_contract(contract)
    contract["validation"] = report
    return contract


def learn_manifest(path: Path, *, activate: bool = False) -> dict[str, Any]:
    """Learn an external JSON contract conservatively; structural validity is not runtime proof."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Benchmark manifest must be a JSON object")
    contract = dict(payload)
    contract.setdefault("schema_version", "ml-scientist-benchmark-contract-v1")
    contract.setdefault("benchmark_id", path.stem)
    contract.setdefault("benchmark_version", _sha256(path)[:16])
    contract["source_kind"] = "user_supplied_manifest"
    contract["source"] = {"path": str(path.resolve()), "sha256": _sha256(path)}
    contract.setdefault("tasks", [])
    contract.setdefault("metrics", [])
    contract["maturity"] = "observation"
    contract["activation_eligible"] = False
    report = validate_benchmark_contract(contract, require_fml_views=False)
    contract["validation"] = report
    if activate:
        if not report["passed"]:
            raise ValueError("Cannot activate an invalid benchmark contract")
        if not contract.get("implementation_audit") or not contract.get("boundary_tests"):
            raise ValueError("Activation requires implementation_audit and boundary_tests evidence")
        contract["maturity"] = "verified"
        contract["activation_eligible"] = True
    return contract


def validate_benchmark_contract(
    contract: dict[str, Any], *, require_fml_views: bool = True
) -> dict[str, Any]:
    errors: list[str] = []
    metrics = contract.get("metrics")
    tasks = contract.get("tasks")
    if not contract.get("benchmark_id"):
        errors.append("missing benchmark_id")
    if not isinstance(tasks, list):
        errors.append("tasks must be a list")
        tasks = []
    if not isinstance(metrics, list) or not metrics:
        errors.append("metrics must be a non-empty list")
        metrics = []
    metric_ids = [metric.get("metric_id") for metric in metrics]
    if any(not metric_id for metric_id in metric_ids):
        errors.append("every metric requires metric_id")
    if len(metric_ids) != len(set(metric_ids)):
        errors.append("duplicate metric_id")
    allowed = {item.value for item in MetricObservability}
    for metric in metrics:
        observability = metric.get("observability")
        if observability not in allowed:
            errors.append(f"unknown observability for {metric.get('metric_id')}")
        if metric.get("protected") and observability != MetricObservability.PROTECTED_FINAL_ONLY.value:
            errors.append(f"protected metric is not final-only: {metric.get('metric_id')}")
        if observability == MetricObservability.PROTECTED_FINAL_ONLY.value and "search_trajectory" in metric.get("stages", []):
            errors.append(f"protected metric exposed to search: {metric.get('metric_id')}")
    if require_fml_views:
        for task in tasks:
            prefix = f"task::{task.get('task_id')}::{task.get('canonical_metric')}"
            views = {metric.get("split") for metric in metrics if str(metric.get("metric_id", "")).startswith(prefix)}
            if views != {"validation", "test"}:
                errors.append(f"task metric views incomplete for {task.get('task_id')}")
            for field in ("validation_command", "protected_test_command", "task_config_sha256"):
                if not task.get(field):
                    errors.append(f"missing {field} for {task.get('task_id')}")
    return {
        "schema_version": "ml-scientist-benchmark-validation-v1",
        "passed": not errors,
        "errors": errors,
        "task_count": len(tasks),
        "metric_count": len(metrics),
        "protected_metric_count": sum(bool(metric.get("protected")) for metric in metrics),
    }


def write_benchmark_contract(contract: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{contract['benchmark_id']}.json"
    path.write_text(json.dumps(contract, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    index = {
        "schema_version": "ml-scientist-benchmark-index-v1",
        "contracts": [
            {
                "benchmark_id": contract["benchmark_id"],
                "benchmark_version": contract.get("benchmark_version"),
                "maturity": contract.get("maturity"),
                "activation_eligible": contract.get("activation_eligible", False),
                "path": path.name,
                "sha256": _sha256(path),
            }
        ],
    }
    (out_dir / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return index
