from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.artifact_contract import materialize_planner_artifacts


class ArtifactContractTests(unittest.TestCase):
    def test_materializes_diff_journal_selection_and_hash_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = {"train.py": "def train():\n    return 1\n"}
            after = {"train.py": "def train():\n    return 2\n"}
            planner = {
                "action": {"hypothesis": "change return", "comparison": [{"idea": "one"}, {"idea": "two"}]},
                "rationale": "minimal edit",
                "planned_policy_use": [{"role": "proposal", "planned_use": "one edit"}],
            }
            pre = materialize_planner_artifacts(
                run_dir=root, step_id=1, planner_output=planner,
                before=before, after=after, research_context={"task_id": "demo"},
            )
            post = materialize_planner_artifacts(
                run_dir=root, step_id=1, planner_output=planner,
                before=before, after=after, research_context={"task_id": "demo"},
                validation={"success": True, "primary_metric": 0.8},
                resources={"tokens_consumed": 10, "wall_clock_seconds": 2},
            )
            artifact_dir = root / "planner_artifacts" / "step_0001"
            patch = (artifact_dir / "candidate.patch").read_text(encoding="utf-8")
            manifest = json.loads((artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8"))
            selection = json.loads((artifact_dir / "selection_record.json").read_text(encoding="utf-8"))
        self.assertIn("-    return 1", patch)
        self.assertIn("+    return 2", patch)
        self.assertEqual(len(pre), 5)
        self.assertEqual(len(post), 5)
        self.assertTrue(manifest["validation_appended"])
        self.assertEqual(selection["eligibility_status"], "ELIGIBLE")
        self.assertTrue(all(len(row["sha256"]) == 64 for row in post))

    def test_rejects_noop_candidate_before_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                materialize_planner_artifacts(
                    run_dir=Path(tmp), step_id=1, planner_output={},
                    before={"a.py": "x=1\n"}, after={"a.py": "x=1\n"},
                    research_context={"task_id": "demo"},
                )


if __name__ == "__main__":
    unittest.main()
