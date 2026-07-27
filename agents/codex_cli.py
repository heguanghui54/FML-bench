"""OpenAI-compatible, text-only adapter for the local Codex CLI.

The FML agents expect ``client.chat.completions.create``.  This module keeps
that interface while executing an authenticated local ``codex exec`` process.
Every call is ephemeral, runs in an empty read-only directory, records the
Codex JSONL event stream, and rejects any tool activity.  Codex is therefore a
language-model condition inside the benchmark rather than a second coding
agent that can bypass an FML baseline's search policy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable


class CodexCLIError(RuntimeError):
    """Raised when Codex cannot produce an auditable text-only response."""


_DISABLED_FEATURES = (
    "plugins",
    "apps",
    "memories",
    "skill_search",
    "tool_suggest",
    "multi_agent",
    "multi_agent_v2",
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "computer_use",
    "workspace_dependencies",
    "image_generation",
    "goals",
    "personality",
)

_ALLOWED_ITEM_TYPES = {"agent_message", "reasoning", "error"}


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(content)


def _build_text_only_prompt(messages: Iterable[dict[str, Any]]) -> str:
    transcript = []
    for message in messages:
        role = str(message.get("role", "user")).upper()
        transcript.append(f"<{role}>\n{_message_text(message.get('content', ''))}\n</{role}>")
    return (
        "You are the text-only language-model backend inside a controlled ML "
        "research benchmark. Do not call tools, inspect files, browse, execute "
        "commands, modify state, or delegate work. Use only the transcript below. "
        "Return only the assistant response requested by the final USER message.\n\n"
        + "\n\n".join(transcript)
    )


def _safe_environment() -> dict[str, str]:
    """Inherit Codex authentication but remove unrelated provider credentials."""
    env = dict(os.environ)
    secret_names = (
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "DEEPSEEK_API_KEY",
    )
    for name in secret_names:
        env.pop(name, None)
    env["NO_COLOR"] = "1"
    return env


class _Completions:
    def __init__(self, owner: "CodexCLIClient"):
        self._owner = owner

    def create(self, *, model: str, messages: list[dict[str, Any]], n: int = 1, **kwargs):
        if n < 1:
            raise ValueError("n must be at least 1")
        choices = []
        totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
        }
        for index in range(n):
            content, usage = self._owner._run_once(
                model=model,
                messages=messages,
                request_index=index,
                request_options=kwargs,
            )
            choices.append(SimpleNamespace(message=SimpleNamespace(content=content)))
            for key in totals:
                totals[key] += int(usage.get(key, 0) or 0)
        usage_obj = SimpleNamespace(**totals)
        return SimpleNamespace(choices=choices, usage=usage_obj)


class CodexCLIClient:
    """Small OpenAI-compatible facade over ``codex exec --json``."""

    provider_name = "CodexCLI"

    def __init__(
        self,
        executable: str | None = None,
        *,
        audit_dir: str | os.PathLike[str] | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: int | None = None,
        validate_login: bool = True,
    ):
        self.executable = executable or os.environ.get("FML_CODEX_BIN") or shutil.which("codex")
        if not self.executable:
            raise CodexCLIError("Codex CLI executable was not found")
        self.reasoning_effort = reasoning_effort or os.environ.get(
            "FML_CODEX_REASONING_EFFORT", "medium"
        )
        self.timeout_seconds = timeout_seconds or int(os.environ.get("FML_CODEX_TIMEOUT", "1200"))
        self.audit_dir = Path(
            audit_dir
            or os.environ.get("FML_CODEX_AUDIT_DIR", "benchmark_results/codex_cli_audit")
        ).resolve()
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._call_number = 0
        self.version = self._read_version()
        if validate_login:
            self._validate_login()
        self.chat = SimpleNamespace(completions=_Completions(self))

    def _read_version(self) -> str:
        try:
            result = subprocess.run(
                [self.executable, "--version"],
                text=True,
                capture_output=True,
                timeout=30,
                check=True,
                env=_safe_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CodexCLIError(f"Unable to read Codex CLI version: {exc}") from exc
        return result.stdout.strip()

    def _validate_login(self) -> None:
        try:
            result = subprocess.run(
                [self.executable, "login", "status"],
                text=True,
                capture_output=True,
                timeout=30,
                env=_safe_environment(),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CodexCLIError(f"Unable to check Codex login status: {exc}") from exc
        status = (result.stdout + result.stderr).strip()
        if result.returncode != 0 or "logged in" not in status.lower():
            raise CodexCLIError(f"Codex CLI is not authenticated: {status or 'unknown status'}")

    def _next_call_id(self) -> str:
        with self._lock:
            self._call_number += 1
            number = self._call_number
        return f"call-{number:06d}-{uuid.uuid4().hex[:12]}"

    def _command(self, model: str, workdir: str) -> list[str]:
        command = [
            self.executable,
            "exec",
            "--ephemeral",
            "--ignore-user-config",
            "--ignore-rules",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "-c",
            f'model_reasoning_effort="{self.reasoning_effort}"',
            "--cd",
            workdir,
        ]
        for feature in _DISABLED_FEATURES:
            command.extend(["--disable", feature])
        command.extend(["--json", "-"])
        return command

    @staticmethod
    def _parse_events(stdout: str) -> tuple[list[dict[str, Any]], list[str]]:
        events = []
        malformed = []
        for raw in stdout.splitlines():
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed.append(line)
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events, malformed

    @staticmethod
    def _extract_response(events: list[dict[str, Any]]) -> tuple[str, dict[str, int], list[str]]:
        messages = []
        violations = []
        usage: dict[str, int] = {}
        for event in events:
            if event.get("type") in {"item.started", "item.completed"}:
                item = event.get("item") or {}
                item_type = item.get("type")
                if item_type not in _ALLOWED_ITEM_TYPES:
                    violations.append(str(item_type or "unknown"))
                if item_type == "agent_message" and isinstance(item.get("text"), str):
                    messages.append(item["text"])
            if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                raw_usage = event["usage"]
                prompt = int(raw_usage.get("input_tokens", 0) or 0)
                completion = int(raw_usage.get("output_tokens", 0) or 0)
                usage = {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "total_tokens": prompt + completion,
                    "cached_input_tokens": int(raw_usage.get("cached_input_tokens", 0) or 0),
                    "reasoning_output_tokens": int(raw_usage.get("reasoning_output_tokens", 0) or 0),
                }
        return (messages[-1] if messages else ""), usage, violations

    def _run_once(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        request_index: int,
        request_options: dict[str, Any],
    ) -> tuple[str, dict[str, int]]:
        call_id = self._next_call_id()
        prompt = _build_text_only_prompt(messages)
        started = time.time()
        with tempfile.TemporaryDirectory(prefix="fml-codex-text-only-") as workdir:
            command = self._command(model, workdir)
            try:
                result = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    env=_safe_environment(),
                )
            except subprocess.TimeoutExpired as exc:
                self._write_audit(
                    call_id,
                    {
                        "status": "TIMEOUT",
                        "model": model,
                        "prompt": prompt,
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "codex_cli_version": self.version,
                        "reasoning_effort": self.reasoning_effort,
                        "timeout_seconds": self.timeout_seconds,
                    },
                )
                raise CodexCLIError(f"Codex CLI timed out after {self.timeout_seconds}s") from exc

        events, malformed = self._parse_events(result.stdout)
        content, usage, violations = self._extract_response(events)
        status = "PASS"
        error = None
        if result.returncode != 0:
            status = "CLI_ERROR"
            error = f"Codex CLI exited with {result.returncode}"
        elif malformed:
            status = "MALFORMED_JSONL"
            error = "Codex emitted non-JSON stdout"
        elif violations:
            status = "REJECTED_TOOL_ACTIVITY"
            error = f"Forbidden Codex item types: {sorted(set(violations))}"
        elif not content:
            status = "EMPTY_RESPONSE"
            error = "Codex emitted no assistant message"

        metadata = {
            "schema_version": "fml-codex-cli-call-v1",
            "call_id": call_id,
            "status": status,
            "started_at_epoch": started,
            "duration_seconds": time.time() - started,
            "model": model,
            "provider": self.provider_name,
            "codex_cli_version": self.version,
            "reasoning_effort": self.reasoning_effort,
            "request_index": request_index,
            "request_options": {
                key: value
                for key, value in request_options.items()
                if key not in {"api_key", "headers", "extra_headers"}
            },
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "response": content,
            "response_sha256": hashlib.sha256(content.encode()).hexdigest(),
            "usage": usage,
            "tool_activity": sorted(set(violations)),
            "malformed_stdout_lines": malformed,
            "returncode": result.returncode,
            "stderr": result.stderr,
            "events": events,
            "error": error,
        }
        self._write_audit(call_id, metadata)
        if error:
            raise CodexCLIError(error)
        return content, usage

    def _write_audit(self, call_id: str, payload: dict[str, Any]) -> None:
        destination = self.audit_dir / f"{call_id}.json"
        destination.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )

