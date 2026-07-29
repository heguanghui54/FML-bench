from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ml_scientist.episode_merge import merge_episode_files


class EpisodeMergeTests(unittest.TestCase):
    def test_merges_deduplicates_and_sorts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right, out = root / "left.jsonl", root / "right.jsonl", root / "out.jsonl"
            row_a = {"episode_id": "a", "outcome": "PASSED_GATE"}
            row_b = {"episode_id": "b", "outcome": "FAILED_GATE"}
            left.write_text(json.dumps(row_b) + "\n" + json.dumps(row_a) + "\n", encoding="utf-8")
            right.write_text(json.dumps(row_a) + "\n", encoding="utf-8")
            report = merge_episode_files([left, right], out)
            rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual([row["episode_id"] for row in rows], ["a", "b"])
        self.assertEqual(report["episode_count"], 2)
        self.assertEqual(report["duplicate_count"], 1)

    def test_rejects_conflicting_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            left, right = root / "left.jsonl", root / "right.jsonl"
            left.write_text('{"episode_id":"same","value":1}\n', encoding="utf-8")
            right.write_text('{"episode_id":"same","value":2}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                merge_episode_files([left, right], root / "out.jsonl")


if __name__ == "__main__":
    unittest.main()
