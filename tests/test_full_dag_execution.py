from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.adaptive_runtime import AdaptiveResearchRuntime
from ml_scientist.catalog import build_catalog
from ml_scientist.node_execution import (
    ArtifactVerifier,
    ExecutionManifest,
    FullDagRunner,
    NodeExecutionResult,
)
from ml_scientist.planner import build_research_paper_plan, validate_plan


ROOT = Path(__file__).resolve().parents[1]


class _FakeEveryNodeExecutor:
    def __init__(self):
        self.review_reopened_once = False

    def execute(self, context):
        context.output_dir.mkdir(parents=True, exist_ok=True)
        for relative in context.node.get("output_artifacts", []):
            path = context.output_dir / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text(json.dumps({
                    "schema_version": "fake-node-artifact-v1",
                    "node_id": context.node["node_id"],
                    "evidence_version": 1,
                }), encoding="utf-8")
            else:
                path.write_text(f"fake evidence for {context.node['node_id']}\n", encoding="utf-8")
        issues = []
        if context.node["node_id"] == "paper-peer-review" and not self.review_reopened_once:
            self.review_reopened_once = True
            issues = [{"issue_id": "fake-implementation-defect", "issue_type": "implementation"}]
        return NodeExecutionResult(context.node["node_id"], "PASSED_GATE", review_issues=issues)


class _FakeRegistry:
    def __init__(self):
        self.executor = _FakeEveryNodeExecutor()

    def resolve(self, _node):
        return self.executor


class FullDagExecutionTests(unittest.TestCase):
    def test_missing_promised_artifact_rejects_node_credit(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifacts, errors = ArtifactVerifier.descriptors(
                Path(tmp), ["promised_but_missing.json"]
            )
            self.assertFalse(artifacts)
            self.assertEqual(errors, ["missing artifact: promised_but_missing.json"])

    def test_fake_executor_completes_every_node_and_rebuilds_after_review(self):
        catalog = build_catalog(ROOT)
        graph = json.loads(
            (ROOT / "artifacts/ml_scientist/post_arm/knowledge_graph/"
             "final_graph_snapshot/knowledge_graph.json").read_text()
        )
        plan = build_research_paper_plan(
            catalog["tasks"][0], catalog["agents"], knowledge_graph=graph,
            research_request="test a small attributable improvement",
        )
        validate_plan(plan)
        self.assertEqual(len(plan["nodes"]), 29)
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            runtime = AdaptiveResearchRuntime(
                plan=plan,
                graph=graph,
                run_id="fake-full-dag",
                state_path=root / "runtime_state.json",
            )
            manifest = ExecutionManifest.create(
                run_id="fake-full-dag",
                plan=plan,
                graph=graph,
                result_root=root / "results",
                workspace=ROOT,
                authorized=True,
                budget_profile="legacy-unbounded",
                budget_limits={},
            )
            runner = FullDagRunner(
                runtime=runtime,
                manifest=manifest,
                registry=_FakeRegistry(),
            )
            # This integration isolates scheduling/artifact/amendment semantics;
            # controller composition selection has separate deterministic tests.
            runner._decision = lambda _node_id: None
            report = runner.run()
            self.assertTrue(report["complete"])
            self.assertTrue(all(status == "COMPLETE" for status in report["node_status"].values()))
            self.assertGreater(report["executed_node_count"], 29)
            self.assertEqual(len(runtime.state["amendments"]), 1)
            amendment = runtime.state["amendments"][0]
            self.assertEqual(amendment["target_node"], "code-modification")
            self.assertGreater(runtime.state["evidence_version"], 1)
            manifests = list((root / "results" / "fake-full-dag").rglob("artifact_manifest.json"))
            self.assertGreaterEqual(len(manifests), report["executed_node_count"])


if __name__ == "__main__":
    unittest.main()
