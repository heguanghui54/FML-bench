#!/usr/bin/env bash
# Source this file on ubuntu-heshi before setup.py or manual task execution.

set -euo pipefail

FML_REMOTE_BASE="${FML_REMOTE_BASE:-/media/heshi/game/fml-scientist}"
case "$FML_REMOTE_BASE" in
  /media/heshi/game/*) ;;
  *) echo "FML_REMOTE_BASE must be below /media/heshi/game: $FML_REMOTE_BASE" >&2; return 90 2>/dev/null || exit 90 ;;
esac

export FML_REMOTE_BASE
export FML_SSH_REMOTE_ROOT="${FML_SSH_REMOTE_ROOT:-$FML_REMOTE_BASE/repo}"
export FML_DATASETS_ROOT="${FML_DATASETS_ROOT:-$FML_REMOTE_BASE/datasets}"
# These mirrors are faster from this host. setup.py verifies the canonical
# torchvision MD5 values before an archive can enter a task workspace.
export FML_CIFAR10_URL="${FML_CIFAR10_URL:-https://huggingface.co/datasets/VerisimilitudeX/cifar10/resolve/main/cifar-10-python.tar.gz}"
export FML_CIFAR100_URL="${FML_CIFAR100_URL:-https://huggingface.co/datasets/nakroy/cifar100-python/resolve/main/cifar-100-python.tar.gz}"
export PATH="$FML_REMOTE_BASE/miniforge3/bin:$PATH"
export CONDA_ENVS_PATH="$FML_REMOTE_BASE/conda-envs"
export CONDA_PKGS_DIRS="$FML_REMOTE_BASE/conda-pkgs"
export HF_HOME="$FML_REMOTE_BASE/cache/huggingface"
export TORCH_HOME="$FML_REMOTE_BASE/cache/torch"
export XDG_CACHE_HOME="$FML_REMOTE_BASE/cache/xdg"
export PIP_CACHE_DIR="$FML_REMOTE_BASE/cache/pip"
export TMPDIR="$FML_REMOTE_BASE/tmp"
export CUDA_CACHE_PATH="$FML_REMOTE_BASE/cache/cuda"
export PYTHONNOUSERSITE=1

mkdir -p \
  "$CONDA_ENVS_PATH" "$CONDA_PKGS_DIRS" "$HF_HOME" "$TORCH_HOME" \
  "$XDG_CACHE_HOME" "$PIP_CACHE_DIR" "$TMPDIR" "$CUDA_CACHE_PATH" \
  "$FML_DATASETS_ROOT" "$FML_REMOTE_BASE/checkpoints" \
  "$FML_REMOTE_BASE/results" "$FML_REMOTE_BASE/bootstrap"
