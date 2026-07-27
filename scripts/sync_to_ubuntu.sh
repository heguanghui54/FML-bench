#!/usr/bin/env bash
: <<'DOC'
Synchronize the controller source tree to the prepared Ubuntu external-disk root.
Generated experiments, local task workspaces, and bootstrap artifacts are excluded.
DOC

set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
ssh_host="${FML_SSH_HOST:-ubuntu-heshi}"
remote_root="${FML_SSH_REMOTE_ROOT:-/media/heshi/game/fml-scientist/repo}"
case "$remote_root" in
  /media/heshi/game/*/repo) ;;
  *) echo "Unsafe remote_root: $remote_root" >&2; exit 90 ;;
esac

ssh -o BatchMode=yes -o ConnectTimeout=8 "$ssh_host" \
  "mkdir -p $(printf '%q' "$remote_root")"
rsync -a \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='artifacts/' \
  --exclude='benchmark_results/' \
  --exclude='metric_reports/' \
  --include='workspace/' \
  --include='workspace/README.md' \
  --exclude='workspace/***' \
  -e 'ssh -o BatchMode=yes -o ConnectTimeout=8 -o ServerAliveInterval=30 -o ServerAliveCountMax=3' \
  "$project_root/" "$ssh_host:$remote_root/"

ssh -o BatchMode=yes -o ConnectTimeout=8 "$ssh_host" \
  "cd $(printf '%q' "$remote_root") && git rev-parse HEAD && git status --short --branch"
