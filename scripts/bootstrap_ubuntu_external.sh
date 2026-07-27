#!/usr/bin/env bash
: <<'DOC'
Install Miniforge and create all heavyweight cache roots on the Ubuntu external disk.

Optional environment:
  FML_REMOTE_BASE=/media/heshi/game/fml-scientist
  FML_DOWNLOAD_PROXY=http://127.0.0.1:17891
DOC

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/ubuntu_external_env.sh"

installer="$FML_REMOTE_BASE/bootstrap/Miniforge3-Linux-x86_64.sh"
release_json="$FML_REMOTE_BASE/bootstrap/miniforge-latest-release.json"
api_url="https://api.github.com/repos/conda-forge/miniforge/releases/latest"
curl_args=(--fail --show-error --location --retry 4 --retry-all-errors)
if [[ -n "${FML_DOWNLOAD_PROXY:-}" ]]; then
  curl_args+=(--proxy "$FML_DOWNLOAD_PROXY")
  export HTTP_PROXY="$FML_DOWNLOAD_PROXY"
  export HTTPS_PROXY="$FML_DOWNLOAD_PROXY"
  export NO_PROXY="localhost,127.0.0.1"
fi

if [[ ! -x "$FML_REMOTE_BASE/miniforge3/bin/conda" ]]; then
  curl "${curl_args[@]}" --output "$release_json" "$api_url"
  read -r url expected < <(python3 - "$release_json" <<'PY'
import json, sys
release = json.load(open(sys.argv[1]))
asset = next(a for a in release["assets"] if a["name"] == "Miniforge3-Linux-x86_64.sh")
digest = asset.get("digest") or ""
if not digest.startswith("sha256:"):
    raise SystemExit("GitHub release asset has no sha256 digest")
print(asset["browser_download_url"], digest.split(":", 1)[1])
PY
  )
  if [[ ! -f "$installer" ]]; then
    curl "${curl_args[@]}" --output "$installer" "$url"
  fi
  actual="$(sha256sum "$installer" | awk '{print $1}')"
  [[ -n "$expected" && "$actual" == "$expected" ]] || {
    echo "Miniforge checksum verification failed" >&2
    exit 91
  }
  bash "$installer" -b -p "$FML_REMOTE_BASE/miniforge3"
fi

source "$FML_REMOTE_BASE/miniforge3/etc/profile.d/conda.sh"
conda config --set auto_activate_base false
conda --version
python --version
printf 'FML_REMOTE_BASE=%s\n' "$FML_REMOTE_BASE"
printf 'CONDA_ENVS_PATH=%s\n' "$CONDA_ENVS_PATH"
printf 'CONDA_PKGS_DIRS=%s\n' "$CONDA_PKGS_DIRS"
