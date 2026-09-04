#!/usr/bin/env bash
set -eo pipefail
ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if test -n "$WORLDCLAW_MODEL_ROOT"; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
LOCK=models.lock.json
if test -n "$1"; then LOCK="$1"; fi
set -u

test "$(id -un)" = "v-huguangyu"
mkdir -p "$ROOT"/{hf,torch,xdg,pip,tmp,runs,cache,conda-pkgs,envs}
chmod 700 "$ROOT" "$ROOT"/{hf,torch,xdg,pip,tmp,runs,cache,conda-pkgs,envs}
test -w "$ROOT"
df -hT / /tmp "$ROOT"
findmnt -T "$ROOT"

export HF_HOME="$ROOT/hf"
export TORCH_HOME="$ROOT/torch"
export XDG_CACHE_HOME="$ROOT/xdg"
export PIP_CACHE_DIR="$ROOT/pip"
export TMPDIR="$ROOT/tmp"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"

python3 - "$LOCK" "$ROOT" <<'PY'
import json, shutil, sys
lock = json.load(open(sys.argv[1], encoding="utf-8"))
unresolved = [name for name, item in lock["models"].items() if not item["resolved"]]
if unresolved:
    raise SystemExit("unresolved lock entries: " + ", ".join(unresolved))
need = sum(item["size_bytes"] for item in lock["models"].values())
free = shutil.disk_usage(sys.argv[2]).free
print({"estimated_weight_bytes": need, "nfs_free_bytes": free, "required_headroom_bytes": 2 * need})
if need > 90 * 1024**3 and free < 2 * need:
    raise SystemExit("NFS free space is less than twice the estimated model weight size")
PY
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
"$HOME/apps/blender-4.2.0-linux-x64/blender" --version | head -n 2
