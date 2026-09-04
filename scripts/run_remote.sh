#!/usr/bin/env bash
set -eo pipefail
ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if test -n "$WORLDCLAW_MODEL_ROOT"; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
export HF_HOME="$ROOT/hf"
export TORCH_HOME="$ROOT/torch"
export XDG_CACHE_HOME="$ROOT/xdg"
export PIP_CACHE_DIR="$ROOT/pip"
export TMPDIR="$ROOT/tmp"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
if ! test -n "$WORLDCLAW_VLLM_URL"; then export WORLDCLAW_VLLM_URL=http://127.0.0.1:8000; fi
export BLENDER_BIN="$HOME/apps/blender-4.2.0-linux-x64/blender"
set -u
exec "$@"
