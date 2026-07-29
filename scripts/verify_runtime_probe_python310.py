#!/usr/bin/env python3
"""Verify generated runtime attribution under the Ubuntu Python 3.10 env."""

import json
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Load the module by path so this focused smoke test does not import the full
# benchmark/agent registry (the task environments intentionally contain only
# the dependencies needed by that task).
runtime_probe_spec = importlib.util.spec_from_file_location(
    "fml_runtime_probe_module", PROJECT_ROOT / "benchmark" / "runtime_probe.py"
)
if runtime_probe_spec is None or runtime_probe_spec.loader is None:
    raise RuntimeError("Could not load benchmark/runtime_probe.py")
runtime_probe_module = importlib.util.module_from_spec(runtime_probe_spec)
runtime_probe_spec.loader.exec_module(runtime_probe_module)
finalize_runtime_probe = runtime_probe_module.finalize_runtime_probe
prepare_runtime_probe = runtime_probe_module.prepare_runtime_probe


BASELINE = """import torch\ntorch.backends.cudnn.benchmark = True\nclass Candidate:\n    @staticmethod\n    def transform(x):\n        return x\n"""
CANDIDATE = """import torch\ntorch.backends.cudnn.benchmark = True\nclass Candidate:\n    @staticmethod\n    def transform(x):\n        return x + 1\n"""


def run(*args, cwd):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        repo.mkdir()
        (repo / "candidate.py").write_text(BASELINE, encoding="utf-8")
        run("git", "init", "-q", cwd=repo)
        run("git", "config", "user.email", "probe@example.invalid", cwd=repo)
        run("git", "config", "user.name", "Probe", cwd=repo)
        run("git", "add", "candidate.py", cwd=repo)
        run("git", "commit", "-qm", "baseline", cwd=repo)
        (repo / "candidate.py").write_text(CANDIDATE, encoding="utf-8")
        contract = prepare_runtime_probe(
            repo=repo,
            target_files=["candidate.py"],
            run_id="python310-smoke",
            configuration={"runtime": sys.version},
        )
        marker = repo / "results_tmp" / "runtime_activation.jsonl"
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo / ".fml_runtime_probe")
        env["FML_RUNTIME_MARKER_PATH"] = str(marker)
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from candidate import Candidate; import torch; "
                    "assert Candidate.transform(1) == 2; "
                    "assert torch.backends.cudnn.benchmark is False"
                ),
            ],
            cwd=repo,
            env=env,
            check=True,
        )
        evidence = Path(tmp) / "evidence"
        evidence.mkdir()
        result = finalize_runtime_probe(repo=repo, execution_dir=evidence, contract=contract)
        runtime_rows = json.loads((evidence / "runtime_reproducibility.json").read_text(encoding="utf-8"))
        correction_count = sum(row.get("determinism_correction_count", 0) for row in runtime_rows)
        print(json.dumps({
            "python": sys.version.split()[0],
            "passed": result["passed"],
            "valid_marker_count": result["valid_marker_count"],
            "missing_activation_targets": result["missing_activation_targets"],
            "runtime_reproducibility_passed": result["runtime_reproducibility_passed"],
            "runtime_reproducibility_process_count": result["runtime_reproducibility_process_count"],
            "determinism_correction_count": correction_count,
        }, sort_keys=True))
        if not result["passed"] or correction_count < 3:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
