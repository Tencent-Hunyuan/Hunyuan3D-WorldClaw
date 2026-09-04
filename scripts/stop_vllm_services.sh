#!/usr/bin/env bash
set -eo pipefail
ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if [ -n "$WORLDCLAW_MODEL_ROOT" ]; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
set -u
for name in planner vlm; do
  pid_file="$ROOT/pids/$name.pid"
  if [ -f "$pid_file" ]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid"
      echo "stopped $name pid=$pid"
    fi
    rm -f "$pid_file"
  fi
done

