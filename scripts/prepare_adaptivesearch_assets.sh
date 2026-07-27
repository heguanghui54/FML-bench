#!/usr/bin/env bash
: <<'DOC'
Prepare AdaptiveSearch's offline GraphCodeBERT controller runtime.

Heavy downloads are staged on the Ubuntu external disk, then copied to the
Mac external-volume repository. The controller installs fixed macOS arm64
Python 3.11 wheels without contacting PyPI, and GraphCodeBERT is loaded with
local_files_only=true during experiments.

Optional environment:
  FML_SSH_HOST=ubuntu-heshi
  FML_REMOTE_BASE=/media/heshi/game/fml-scientist
  FML_DOWNLOAD_PROXY=http://127.0.0.1:17891
DOC

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
ssh_host="${FML_SSH_HOST:-ubuntu-heshi}"
remote_base="${FML_REMOTE_BASE:-/media/heshi/game/fml-scientist}"
download_proxy="${FML_DOWNLOAD_PROXY:-}"
controller_python="$project_root/.controller-venv/bin/python"
local_wheels="$project_root/.cache/wheels"
local_hf_cache="$project_root/.cache/huggingface"

case "$project_root" in
  /Volumes/*) ;;
  *) echo "Controller repository must be on a mounted external volume: $project_root" >&2; exit 90 ;;
esac
case "$remote_base" in
  /media/*) ;;
  *) echo "Remote asset root must be on an external mount: $remote_base" >&2; exit 91 ;;
esac
[[ "$(uname -m)" == "arm64" ]] || {
  echo "This asset bundle targets macOS arm64; found $(uname -m)." >&2
  exit 92
}
[[ -x "$controller_python" ]] || {
  echo "Run scripts/bootstrap_codex_controller.sh first." >&2
  exit 93
}

ssh -o BatchMode=yes -o ConnectTimeout=8 "$ssh_host" bash -s -- \
  "$remote_base" "$download_proxy" <<'REMOTE'
set -euo pipefail
remote_base="$1"
download_proxy="$2"
asset_venv="$remote_base/asset-tools"
wheel_dir="$remote_base/packages/controller-macos-arm64-py311"
hf_cache="$remote_base/cache/huggingface"
base_python="$remote_base/miniforge3/bin/python"

[[ -x "$base_python" ]] || {
  echo "Run scripts/bootstrap_ubuntu_external.sh before preparing assets." >&2
  exit 94
}
mkdir -p "$wheel_dir" "$hf_cache"
if [[ ! -x "$asset_venv/bin/python" ]]; then
  "$base_python" -m venv "$asset_venv"
fi
if [[ -n "$download_proxy" ]]; then
  export HTTP_PROXY="$download_proxy"
  export HTTPS_PROXY="$download_proxy"
  export NO_PROXY="localhost,127.0.0.1"
fi
"$asset_venv/bin/pip" install "huggingface-hub==1.24.0"

target=(
  --no-deps
  --dest "$wheel_dir"
  --platform macosx_11_0_arm64
  --python-version 3.11
  --implementation cp
  --abi cp311
  --only-binary=:all:
)
"$asset_venv/bin/pip" download "${target[@]}" \
  "torch==2.10.0" "transformers==5.3.0"
"$asset_venv/bin/pip" download "${target[@]}" \
  filelock "sympy>=1.13.3" "networkx>=2.5.1" jinja2 "fsspec>=0.8.5" \
  "huggingface-hub>=1.3.0,<2.0" "packaging>=20.0" "regex!=2019.12.17" \
  "tokenizers>=0.22.0,<=0.23.0" typer "safetensors>=0.4.3" markupsafe \
  "mpmath>=1.1.0,<1.4" click shellingham rich markdown-it-py pygments mdurl \
  hf-xet "annotated-doc>=0.0.2"

HF_HOME="$hf_cache" HF_HUB_DISABLE_XET=1 "$asset_venv/bin/python" - <<PY
from huggingface_hub import snapshot_download
print(snapshot_download(
    repo_id="microsoft/graphcodebert-base",
    cache_dir=r"$hf_cache",
    allow_patterns=["*.json", "*.txt", "*.bin", "*.safetensors"],
))
PY
REMOTE

mkdir -p "$local_wheels" "$local_hf_cache"
rsync -a "$ssh_host:$remote_base/packages/controller-macos-arm64-py311/" "$local_wheels/"
rsync -a "$ssh_host:$remote_base/cache/huggingface/" "$local_hf_cache/"
"$controller_python" -m pip install --no-index --find-links "$local_wheels" \
  "torch==2.10.0" "transformers==5.3.0"
"$controller_python" -m pip check

cd "$project_root"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$controller_python" - <<'PY'
import hashlib
import numpy as np
from agents.adaptivesearch.embeddings import GraphCodeBERTEmbedder

embedder = GraphCodeBERTEmbedder(
    model_id="microsoft/graphcodebert-base",
    device="cpu",
    max_len=512,
    cache_dir=".cache/huggingface",
    local_files_only=True,
)
vector = embedder.embed_files({"probe.py": "def probe(x):\n    return x + 1\n"})
if vector.shape != (768,) or vector.dtype != np.float32 or not np.isfinite(vector).all():
    raise SystemExit("GraphCodeBERT offline embedding smoke failed")
print(f"graphcodebert_vector_sha256={hashlib.sha256(vector.tobytes()).hexdigest()}")
PY

printf 'controller_wheels=%s\n' "$local_wheels"
printf 'graphcodebert_cache=%s\n' "$local_hf_cache"
