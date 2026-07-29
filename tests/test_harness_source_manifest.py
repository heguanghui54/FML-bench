import tempfile
import unittest
from pathlib import Path

from run_agent_benchmark import build_harness_source_manifest


class HarnessSourceManifestTests(unittest.TestCase):
    def test_manifest_includes_untracked_source_and_changes_with_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "agents" / "new_agent").mkdir(parents=True)
            source = root / "agents" / "new_agent" / "agent.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            first = build_harness_source_manifest(str(root))
            self.assertEqual(first["source_file_count"], 1)
            self.assertEqual(first["files"][0]["path"], "agents/new_agent/agent.py")
            source.write_text("VALUE = 2\n", encoding="utf-8")
            second = build_harness_source_manifest(str(root))
            self.assertNotEqual(first["content_sha256"], second["content_sha256"])

    def test_manifest_excludes_regenerable_baseline_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ml_tasks" / "task" / "baseline_results").mkdir(parents=True)
            (root / "ml_tasks" / "task" / "baseline_results" / "score.json").write_text("{}")
            (root / "ml_tasks" / "task" / "postprocess.py").write_text("VALUE = 1\n")
            manifest = build_harness_source_manifest(str(root))
            self.assertEqual([row["path"] for row in manifest["files"]], ["ml_tasks/task/postprocess.py"])


if __name__ == "__main__":
    unittest.main()
