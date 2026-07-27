#!/usr/bin/env bash
: <<'DOC'
Install the eight FML-Lite task environments and datasets on the Ubuntu
external disk. The Mac controller owns the benchmark harness, so this skips
the general fmlbench environment. Each task has an independent log and a
failure does not prevent later tasks from being attempted.

Optional environment:
  FML_DOWNLOAD_PROXY=http://127.0.0.1:17891
DOC

set -uo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
source "$script_dir/ubuntu_external_env.sh"

if [[ -n "${FML_DOWNLOAD_PROXY:-}" ]]; then
  export HTTP_PROXY="$FML_DOWNLOAD_PROXY"
  export HTTPS_PROXY="$FML_DOWNLOAD_PROXY"
  export NO_PROXY="localhost,127.0.0.1"
fi

tasks=(
  Generalization_domainbed
  Continual_Learning_pycil
  Data_Efficiency_usb
  Generalization_domainbed_officehome
  Privacy_opacus
  Privacy_privacymeter
  Robustness_and_Reliability_art
  Robustness_openood
)
log_dir="$FML_REMOTE_BASE/results/setup-lite"
mkdir -p "$log_dir"
failures=()

cd "$project_root"
for task in "${tasks[@]}"; do
  log="$log_dir/$task.log"
  printf '[lite-setup] task=%s log=%s\n' "$task" "$log"
  if python setup.py --skip-harness-env --task "$task" 2>&1 | tee "$log"; then
    printf 'PASS\n' > "$log_dir/$task.status"
  else
    rc=${PIPESTATUS[0]}
    printf 'FAIL rc=%s\n' "$rc" > "$log_dir/$task.status"
    failures+=("$task:$rc")
  fi
done

if ((${#failures[@]})); then
  printf '[lite-setup] failures=%s\n' "${failures[*]}" >&2
  exit 1
fi
printf '[lite-setup] all eight tasks ready\n'
