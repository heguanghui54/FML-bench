from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.governance import EvidenceGatedSkillRegistry
from ml_scientist.skill_evolution import SkillEvolutionEngine


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _episode(episode_id: str, *, policy_evaluations=None, skill_validations=None, outcome="PASSED_GATE") -> dict:
    return {
        "schema_version": "ml-scientist-adaptive-episode-v2",
        "episode_id": episode_id,
        "run_id": episode_id.split(":")[0],
        "task_id": "Privacy_privacymeter",
        "node_id": "hypothesis-frontier",
        "node_type": "method_search",
        "decision_id": f"decision:{episode_id}",
        "selected_policy_ids": ["policy:autoresearch:proposal"],
        "selected_policy_by_role": {"proposal": "policy:autoresearch:proposal"},
        "visible_evidence": [
            {"metric_name": "Exploration Spread", "observability": "ONLINE_VISIBLE", "value": 0.2}
        ],
        "research_context": {
            "research_request": "privacy attack evaluation with limited compute",
            "task_id": "Privacy_privacymeter",
            "task_domain": "Privacy",
            "node_type": "method_search",
        },
        "state_before": {"node_status": "READY", "budget": {"tokens_remaining": 1000}},
        "action": {"decision_id": f"decision:{episode_id}"},
        "outcome": outcome,
        "next_state": {"node_status": "COMPLETE", "route": {"action": "ADVANCE"}},
        "resource_telemetry": {"tokens_consumed": 200, "diversity_delta": 0.3},
        "evidence_artifacts": [{"path": "hypotheses.json"}],
        "policy_evaluations": policy_evaluations or [],
        "skill_validation_evaluations": skill_validations or [],
        "snapshot_sha256": "frozen-graph",
        "protected_metric_used_for_routing": False,
        "protected_metric_used_for_skill_evolution": False,
    }


class SkillEvolutionLifecycleTests(unittest.TestCase):
    @staticmethod
    def _signal() -> dict:
        return {
            "intent": "CREATE",
            "branch": "task_specific",
            "eligible_stages": ["method_search"],
            "reusable_pattern": "Switch away from repeated confidence smoothing",
            "trigger": {"failed_family": "label_smoothing"},
            "procedure": ["Retrieve failed interventions", "Select a distinct mechanism"],
            "expected_effect": {"novelty": "increase"},
            "do_not_use_when": [],
            "requires_capabilities": ["negative_memory"],
            "provides_capabilities": ["candidate_pool"],
            "acceptance_gates": ["held_out_no_regression"],
            "rollback_condition": "held-out validation regresses",
        }

    def test_novelty_routes_failure_to_negative_memory_and_exact_repeat_to_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = EvidenceGatedSkillRegistry(root / "governance", "commit")
            failed = _episode(
                "failed:1",
                outcome="VALID_REGRESSION",
                policy_evaluations=[{
                    "policy_id": "policy:autoresearch:proposal", "role": "proposal",
                    "applied": True, "status": "FAILED",
                    "evidence_artifact_refs": ["failed.json"],
                    "learning_signal": self._signal(),
                }],
            )
            failed_path = root / "failed.jsonl"
            _write_jsonl(failed_path, [failed])
            negative = SkillEvolutionEngine(
                registry=registry, out_dir=root / "negative"
            ).evolve(failed_path)
            self.assertEqual(negative["write_actions"][0]["intent"], "NEGATIVE_MEMORY")

            author = _episode(
                "author:2",
                policy_evaluations=[{
                    "policy_id": "policy:autoresearch:proposal", "role": "proposal",
                    "applied": True, "status": "PASSED",
                    "evidence_artifact_refs": ["author.json"],
                    "learning_signal": self._signal(),
                }],
            )
            author_path = root / "author.jsonl"
            _write_jsonl(author_path, [author])
            created = SkillEvolutionEngine(
                registry=registry, out_dir=root / "create"
            ).evolve(author_path)
            self.assertEqual(created["write_actions"][0]["intent"], "CREATE")
            repeated = _episode(
                "repeat:3",
                policy_evaluations=[{
                    "policy_id": "policy:autoresearch:proposal", "role": "proposal",
                    "applied": True, "status": "PASSED",
                    "evidence_artifact_refs": ["repeat.json"],
                    "learning_signal": self._signal(),
                }],
            )
            repeat_path = root / "repeat.jsonl"
            _write_jsonl(repeat_path, [repeated])
            duplicate = SkillEvolutionEngine(
                registry=registry, out_dir=root / "duplicate"
            ).evolve(repeat_path)
            self.assertEqual(duplicate["write_actions"][0]["intent"], "DUPLICATE")

    def test_create_then_later_held_out_promote_and_repeat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = EvidenceGatedSkillRegistry(root / "governance", "commit")
            signal = {
                "intent": "CREATE",
                "branch": "task_specific",
                "policy_role": "proposal",
                "eligible_stages": ["method_search"],
                "reusable_pattern": "Recover hypothesis diversity after local-search stagnation",
                "trigger": {"exploration_spread_lt": 0.3, "failed_attempts_gte": 2},
                "procedure": ["Freeze the current candidates", "Generate alternatives from distinct mechanisms"],
                "expected_effect": {"Exploration Spread": "increase"},
                "do_not_use_when": ["remaining budget is below one candidate evaluation"],
                "requires_capabilities": ["candidate_state"],
                "provides_capabilities": ["candidate_pool"],
                "acceptance_gates": ["held_out_spread_increases", "no_validation_regression"],
                "rollback_condition": "comparable held-out evidence shows regression",
            }
            authoring = _episode(
                "author:1",
                policy_evaluations=[{
                    "policy_id": "policy:autoresearch:proposal",
                    "role": "proposal",
                    "applied": True,
                    "status": "PASSED",
                    "evidence_artifact_refs": ["hypotheses.json"],
                    "gate_checks": ["diversity_increased"],
                    "learning_signal": signal,
                }],
            )
            authoring_path = root / "authoring.jsonl"
            _write_jsonl(authoring_path, [authoring])
            report = SkillEvolutionEngine(registry=registry, out_dir=root / "evolution-1").evolve(authoring_path)
            created = next(row for row in report["write_actions"] if row["intent"] == "CREATE")
            skill_id = created["skill_id"]
            skill = registry._find(skill_id)
            self.assertEqual(skill["versions"][-1]["maturity"], "observation")
            self.assertIsNone(skill["active_version"])
            self.assertFalse(report["new_writes_visible_to_authoring_window"])
            for name in (
                "trajectory_buffer_report.json", "skill_distillation_report.json",
                "utility_assessment_report.json", "skill_governance_report.json",
            ):
                self.assertTrue((root / "evolution-1" / name).is_file())

            validation_one = _episode(
                "heldout-a:1",
                skill_validations=[{
                    "skill_id": skill_id,
                    "role": "proposal",
                    "applied": True,
                    "status": "PASSED",
                    "evidence_artifact_refs": ["heldout-a.json"],
                    "held_out_validation": True,
                    "real_downstream_complete": True,
                    "no_regression": True,
                    "reward": 0.8,
                    "baseline_reward": 0.5,
                    "context_id": "Privacy::method_search",
                }],
            )
            validation_path = root / "validation-one.jsonl"
            _write_jsonl(validation_path, [validation_one])
            report_two = SkillEvolutionEngine(registry=registry, out_dir=root / "evolution-2").evolve(validation_path)
            self.assertEqual(report_two["assessment_governance_actions"][0]["action"], "PROMOTE")
            self.assertEqual(registry._find(skill_id)["versions"][-1]["maturity"], "provisional_success")

            validation_two = _episode(
                "heldout-b:1",
                skill_validations=[{
                    "skill_id": skill_id,
                    "role": "proposal",
                    "applied": True,
                    "status": "PASSED",
                    "evidence_artifact_refs": ["heldout-b.json"],
                    "held_out_validation": True,
                    "real_downstream_complete": True,
                    "no_regression": True,
                    "reward": 0.7,
                    "baseline_reward": 0.5,
                    "context_id": "Representation_Learning::method_search",
                }],
            )
            validation_two_path = root / "validation-two.jsonl"
            _write_jsonl(validation_two_path, [validation_two])
            SkillEvolutionEngine(registry=registry, out_dir=root / "evolution-3").evolve(validation_two_path)
            version = registry._find(skill_id)["versions"][-1]
            self.assertEqual(version["maturity"], "repeated_success")
            self.assertEqual(set(version["utility"]["by_context"]), {
                "Privacy::method_search", "Representation_Learning::method_search",
            })

    def test_invalid_execution_is_filtered_before_writing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = EvidenceGatedSkillRegistry(root / "governance", "commit")
            path = root / "invalid.jsonl"
            _write_jsonl(path, [_episode("bad:1", outcome="INVALID_EXECUTION")])
            report = SkillEvolutionEngine(registry=registry, out_dir=root / "evolution").evolve(path)
            self.assertFalse(report["write_actions"])
            self.assertEqual(report["rejected_episodes"][0]["reason"], "INFRASTRUCTURE_OR_INVALID_EXECUTION")


if __name__ == "__main__":
    unittest.main()
