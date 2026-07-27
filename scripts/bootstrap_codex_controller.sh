#!/usr/bin/env bash
: <<'DOC'
Create the core Mac-side controller environment for CodexCLI runs.

The environment lives beside the repository on the mounted external disk. It
does not contain task datasets, checkpoints, or task environments; those remain
on the Ubuntu external disk. AdaptiveSearch additionally needs the fixed,
offline GraphCodeBERT bundle prepared by prepare_adaptivesearch_assets.sh.

Optional environment:
  FML_CONTROLLER_PYTHON=/opt/homebrew/bin/python3.11
  FML_DOWNLOAD_PROXY=http://127.0.0.1:17891
DOC

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
venv="$project_root/.controller-venv"
controller_python="${FML_CONTROLLER_PYTHON:-$(command -v python3.11 || true)}"

case "$venv" in
  /Volumes/*/.controller-venv) ;;
  *) echo "Controller environment must be on a mounted external volume: $venv" >&2; exit 90 ;;
esac
[[ -n "$controller_python" && -x "$controller_python" ]] || {
  echo "Python 3.11 is required; set FML_CONTROLLER_PYTHON." >&2
  exit 91
}

if [[ ! -x "$venv/bin/python" ]]; then
  "$controller_python" -m venv "$venv"
fi
"$venv/bin/python" - <<'PY'
import sys
if sys.version_info[:2] != (3, 11):
    raise SystemExit(f"Controller venv must use Python 3.11, got {sys.version}")
PY

if [[ -n "${FML_DOWNLOAD_PROXY:-}" ]]; then
  export HTTP_PROXY="$FML_DOWNLOAD_PROXY"
  export HTTPS_PROXY="$FML_DOWNLOAD_PROXY"
  export NO_PROXY="localhost,127.0.0.1"
fi

"$venv/bin/python" -m pip install \
  pyyaml==6.0.2 \
  backoff==2.2.1 \
  openai==1.91.0 \
  numpy==1.26.4 \
  requests==2.32.4

"$venv/bin/python" "$project_root/run_agent_benchmark.py" --help >/dev/null
printf 'controller_python=%s\n' "$venv/bin/python"
printf 'next_for_all_baselines=bash scripts/prepare_adaptivesearch_assets.sh\n'
