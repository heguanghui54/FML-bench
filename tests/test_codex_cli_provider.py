import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / "agents" / "codex_cli.py"
SPEC = importlib.util.spec_from_file_location("fml_codex_cli_under_test", MODULE_PATH)
codex_cli = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(codex_cli)
CodexCLIClient = codex_cli.CodexCLIClient
CodexCLIError = codex_cli.CodexCLIError


class _Result:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


class CodexCLIProviderTests(unittest.TestCase):
    def _client(self, audit_dir):
        with patch.object(codex_cli.subprocess, "run") as run:
            run.side_effect = [
                _Result(stdout="codex-cli 1.2.3\n"),
                _Result(stdout="Logged in using ChatGPT\n"),
            ]
            return CodexCLIClient(
                executable="/test/codex",
                audit_dir=audit_dir,
                validate_login=True,
            )

    def test_openai_compatible_response_and_usage_are_audited(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            events = "\n".join(
                [
                    json.dumps({"type": "thread.started", "thread_id": "t1"}),
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 12, "cached_input_tokens": 4, "output_tokens": 3, "reasoning_output_tokens": 1}}),
                ]
            )
            with patch.object(codex_cli.subprocess, "run", return_value=_Result(stdout=events)) as run:
                response = client.chat.completions.create(
                    model="gpt-test",
                    messages=[{"role": "user", "content": "question"}],
                    seed=7,
                )
            self.assertEqual(response.choices[0].message.content, "answer")
            self.assertEqual(response.usage.prompt_tokens, 12)
            self.assertEqual(response.usage.completion_tokens, 3)
            self.assertEqual(response.usage.total_tokens, 15)
            command = run.call_args.args[0]
            self.assertIn("--ephemeral", command)
            self.assertIn("read-only", command)
            self.assertEqual(run.call_args.kwargs["input"].count("question"), 1)
            audits = list(Path(tmp).glob("call-*.json"))
            self.assertEqual(len(audits), 1)
            audit = json.loads(audits[0].read_text())
            self.assertEqual(audit["status"], "PASS")
            self.assertEqual(audit["codex_cli_version"], "codex-cli 1.2.3")

    def test_tool_activity_is_rejected_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            client = self._client(tmp)
            events = "\n".join(
                [
                    json.dumps({"type": "item.completed", "item": {"type": "command_execution", "command": "pwd"}}),
                    json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}}),
                    json.dumps({"type": "turn.completed", "usage": {"input_tokens": 2, "output_tokens": 1}}),
                ]
            )
            with patch.object(codex_cli.subprocess, "run", return_value=_Result(stdout=events)):
                with self.assertRaisesRegex(CodexCLIError, "Forbidden Codex item types"):
                    client.chat.completions.create(
                        model="gpt-test",
                        messages=[{"role": "user", "content": "question"}],
                    )
            audit = json.loads(next(Path(tmp).glob("call-*.json")).read_text())
            self.assertEqual(audit["status"], "REJECTED_TOOL_ACTIVITY")
            self.assertEqual(audit["tool_activity"], ["command_execution"])

    def test_missing_login_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(codex_cli.subprocess, "run") as run:
                run.side_effect = [
                    _Result(stdout="codex-cli 1.2.3\n"),
                    _Result(stderr="Not logged in", returncode=1),
                ]
                with self.assertRaisesRegex(CodexCLIError, "not authenticated"):
                    CodexCLIClient(executable="/test/codex", audit_dir=tmp)


if __name__ == "__main__":
    unittest.main()
