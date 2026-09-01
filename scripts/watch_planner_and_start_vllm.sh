#!/usr/bin/env bash
set -euo pipefail

ROOT="${WORLDCLAW_MODEL_ROOT:-/mnt/data/v-huguangyu/worldclaw-oss}"
SRC="${WORLDCLAW_SOURCE_ROOT:-$ROOT/src}"
echo "planner download watcher disabled: planner and asset routing are forced to gpt-5.6-sol API"
exec bash "$SRC/scripts/start_vllm_services.sh"
