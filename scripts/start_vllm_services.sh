#!/usr/bin/env bash
set -eo pipefail
ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if [ -n "$WORLDCLAW_MODEL_ROOT" ]; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
SRC="$ROOT/src"
if [ -n "$WORLDCLAW_SOURCE_ROOT" ]; then SRC="$WORLDCLAW_SOURCE_ROOT"; fi
set -u
export HF_HOME="$ROOT/hf"
export TORCH_HOME="$ROOT/torch"
export XDG_CACHE_HOME="$ROOT/xdg"
export PIP_CACHE_DIR="$ROOT/pip"
export TMPDIR="$ROOT/tmp"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
ENV="$ROOT/envs/llm-vlm"
LOCK="$SRC/models.lock.json"
LOGS="$ROOT/logs"
PIDS="$ROOT/pids"
mkdir -p "$LOGS" "$PIDS"

read_model() {
  "$ENV/bin/python" - "$LOCK" "$1" "$2" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))["models"][sys.argv[2]]
print(value[sys.argv[3]])
PY
}

VLM_ID="$(read_model vlm model_id)"
VLM_REV="$(read_model vlm revision)"

start_service() {
  local name="$1"
  local gpu="$2"
  local port="$3"
  local model="$4"
  local revision="$5"
  local served_model="$4"
  shift 5
  local pid_file="$PIDS/$name.pid"
  local snapshot="$HF_HOME/hub/models--${model//\//--}/snapshots/$revision"
  if [ -d "$snapshot" ]; then
    model="$snapshot"
  elif [ "${WORLDCLAW_ALLOW_REMOTE_MODEL_FETCH:-0}" != "1" ]; then
    echo "$name model is not present in local HF cache: $model@$revision" >&2
    return 2
  fi
  if [ -f "$pid_file" ] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  CUDA_VISIBLE_DEVICES="$gpu" HF_HUB_OFFLINE="$([[ -d "$snapshot" ]] && echo 1 || echo 0)" \
    nohup "$ENV/bin/vllm" serve "$model" \
    --served-model-name "$served_model" \
    --revision "$revision" \
    --host 127.0.0.1 \
    --port "$port" \
    --dtype half \
    --gpu-memory-utilization 0.90 \
    "$@" > "$LOGS/$name.log" 2>&1 &
  echo "$!" > "$pid_file"
  echo "started $name pid=$! gpu=$gpu port=$port"
}

echo "planner vLLM service disabled: planner and asset routing are forced to gpt-5.6-sol API"
if [ "${WORLDCLAW_VALIDATION_PROVIDER:-openai}" = "vllm" ]; then
  start_service vlm "${WORLDCLAW_VLM_GPU:-1}" 8001 "$VLM_ID" "$VLM_REV" --max-model-len 16384 --limit-mm-per-prompt '{"image":4}' || \
    echo "vlm unavailable locally; continuing with planner only" >&2
else
  echo "skipping local vlm; validation provider is ${WORLDCLAW_VALIDATION_PROVIDER:-openai}"
fi

wait_ready() {
  local name="$1"
  local port="$2"
  for attempt in $(seq 1 120); do
    if curl -fsS "http://127.0.0.1:$port/v1/models" >/dev/null; then
      echo "$name ready"
      return
    fi
    if ! kill -0 "$(cat "$PIDS/$name.pid")" 2>/dev/null; then
      tail -n 100 "$LOGS/$name.log"
      return 1
    fi
    sleep 5
  done
  echo "$name readiness timeout" >&2
  return 1
}

if [ "${WORLDCLAW_VALIDATION_PROVIDER:-openai}" = "vllm" ]; then
  if [ -f "$PIDS/vlm.pid" ] && kill -0 "$(cat "$PIDS/vlm.pid")" 2>/dev/null; then
    wait_ready vlm 8001
  else
    echo "vlm not started; its local weights are unavailable" >&2
  fi
fi
