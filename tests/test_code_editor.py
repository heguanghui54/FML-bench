from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from agents.code_editor import CodeEditor


class CodeEditorPathParsingTests(unittest.TestCase):
    def test_whole_file_path_may_contain_unicode_and_spaces(self):
        path = "/Volumes/mac外接盘/self-evolve ml/workspace/train.py"
        response = f"### FILE: {path}\n```python\nvalue = 2\n```"
        self.assertEqual(CodeEditor._extract_whole_files(response), [(path, "value = 2")])

    def test_search_replace_path_may_contain_unicode_spaces_and_crlf(self):
        path = "/Volumes/mac外接盘/self-evolve ml/workspace/large model.py"
        response = (
            f"### FILE: {path}\r\n"
            "<<<<<<< SEARCH\r\n"
            "value = 1\r\n"
            "=======\r\n"
            "value = 2\r\n"
            ">>>>>>> REPLACE"
        )
        self.assertEqual(
            CodeEditor._extract_search_replace(response),
            [(path, "value = 1", "value = 2")],
        )

    def test_apply_reports_only_files_with_real_text_changes(self):
        with tempfile.TemporaryDirectory(prefix="editor path ") as tmp:
            changed = Path(tmp) / "changed file.py"
            unchanged = Path(tmp) / "unchanged file.py"
            changed.write_text("value = 1\n")
            unchanged.write_text("stable = True\n")
            targets = {
                str(changed): changed.read_text(),
                str(unchanged): unchanged.read_text(),
            }
            response = (
                f"### FILE: {changed}\n```python\nvalue = 2\n```\n"
                f"### FILE: {unchanged}\n```python\nstable = True\n```"
            )
            editor = object.__new__(CodeEditor)
            modified = editor._parse_and_apply(response, targets)
            self.assertEqual(modified, [str(changed)])
            self.assertEqual(changed.read_text(), "value = 2\n")
            self.assertEqual(unchanged.read_text(), "stable = True\n")


if __name__ == "__main__":
    unittest.main()
