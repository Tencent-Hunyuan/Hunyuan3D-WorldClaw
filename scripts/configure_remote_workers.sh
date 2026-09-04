#!/usr/bin/env bash
ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if [ -n "$WORLDCLAW_MODEL_ROOT" ]; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
SRC="$ROOT/src"
if [ -n "$WORLDCLAW_SOURCE_ROOT" ]; then SRC="$WORLDCLAW_SOURCE_ROOT"; fi
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
export WORLDCLAW_MODEL_ROOT="$ROOT"
# The remote control host keeps the OpenAI credential in the user's Codex
# auth file. Export only its path; workers read the key at runtime and never
# persist its value in requests, responses, or manifests.
export WORLDCLAW_OPENAI_AUTH_FILE="${WORLDCLAW_OPENAI_AUTH_FILE:-$HOME/.codex/auth.json}"
# ziplab's OpenAI-compatible gateway is the default remote route. Keep this
# overrideable for other hosts; include /v1 because OPENAI_BASE_URL is used
# verbatim by the OpenAI SDK when explicitly set.
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://sub2api-hk.ziplab.co/v1}"
export HF_HOME="$ROOT/hf"
export TORCH_HOME="$ROOT/torch"
export XDG_CACHE_HOME="$ROOT/xdg"
export PIP_CACHE_DIR="$ROOT/pip"
export TMPDIR="$ROOT/tmp"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
export HUNYUAN3D_SOURCE="$ROOT/sources/Hunyuan3D-2.1"
export HY3DGEN_MODELS="$ROOT/hy3dgen-cache"
export WORLDCLAW_OPEN3D_PYTHON="$ROOT/envs/orchestrator/bin/python"
export WORLDCLAW_FLUX_GPU="${WORLDCLAW_FLUX_GPU:-2}"
export WORLDCLAW_LOCAL_FLUX=1
export WORLDCLAW_SEGMENT_GPU="${WORLDCLAW_SEGMENT_GPU:-3}"
export WORLDCLAW_IMAGE3D_GPU="${WORLDCLAW_IMAGE3D_GPU:-4}"
export WORLDCLAW_FLUX_WORKER="env HF_HOME=$ROOT/hf CUDA_VISIBLE_DEVICES=$WORLDCLAW_FLUX_GPU $ROOT/envs/image-seg/bin/python $SRC/workers/flux_worker.py"
# Environment asset references and regional composition use FLUX by default.
# Keep this overrideable for experiments, but do not route references through
# the GPT Image 2 API in the standard workflow.
export WORLDCLAW_REFERENCE_IMAGE_WORKER="${WORLDCLAW_REFERENCE_IMAGE_WORKER:-env HF_HOME=$ROOT/hf CUDA_VISIBLE_DEVICES=$WORLDCLAW_FLUX_GPU $ROOT/envs/image-seg/bin/python $SRC/workers/flux_worker.py}"
export WORLDCLAW_RECON_PREFLIGHT_WORKER="$ROOT/envs/orchestrator/bin/python $SRC/workers/reconstruction_preflight_worker.py"
export WORLDCLAW_SEGMENT_WORKER="env HF_HOME=$ROOT/hf CUDA_VISIBLE_DEVICES=$WORLDCLAW_SEGMENT_GPU $ROOT/envs/image-seg/bin/python $SRC/workers/segmentation_worker.py"
HUNYUAN_SITE="$ROOT/envs/hunyuan3d/lib/python3.10/site-packages/nvidia"
HUNYUAN_BPY_LIB="$ROOT/envs/hunyuan3d/lib/python3.10/site-packages/bpy/lib"
HUNYUAN_TORCH_LIB="$ROOT/envs/hunyuan3d/lib/python3.10/site-packages/torch/lib"
HUNYUAN_CUDA_LIBS="$HUNYUAN_BPY_LIB:$HUNYUAN_TORCH_LIB:$HUNYUAN_SITE/cudnn/lib:$HUNYUAN_SITE/cublas/lib:$HUNYUAN_SITE/cuda_runtime/lib:$HUNYUAN_SITE/cuda_nvrtc/lib:$HUNYUAN_SITE/cufft/lib:$HUNYUAN_SITE/curand/lib:$HUNYUAN_SITE/cusolver/lib:$HUNYUAN_SITE/cusparse/lib:$HUNYUAN_SITE/cusparselt/lib:$HUNYUAN_SITE/nccl/lib:$HUNYUAN_SITE/nvjitlink/lib:$HUNYUAN_SITE/nvtx/lib"
if [ -n "${LD_LIBRARY_PATH:-}" ]; then HUNYUAN_CUDA_LIBS="$HUNYUAN_CUDA_LIBS:$LD_LIBRARY_PATH"; fi
export HUNYUAN_CUDA_LIBS
export WORLDCLAW_IMAGE3D_WORKER="env HF_HOME=$ROOT/hf LD_LIBRARY_PATH=$HUNYUAN_CUDA_LIBS CUDA_VISIBLE_DEVICES=$WORLDCLAW_IMAGE3D_GPU $ROOT/envs/hunyuan3d/bin/python $SRC/workers/hunyuan3d_worker.py"
export WORLDCLAW_MESH_VALIDATION_WORKER="env HF_HOME=$ROOT/hf WORLDCLAW_MODEL_ROOT=$ROOT BLENDER_BIN=$HOME/apps/blender-4.2.0-linux-x64/blender $ROOT/envs/orchestrator/bin/python $SRC/workers/mesh_validation_worker.py"
if [ -x "$ROOT/envs/trellis/bin/python" ]; then
  export WORLDCLAW_TRELLIS_WORKER="env CUDA_VISIBLE_DEVICES=5 $ROOT/envs/trellis/bin/python $SRC/workers/trellis_worker.py"
fi
export WORLDCLAW_PLACEMENT_WORKER="$ROOT/envs/orchestrator/bin/python $SRC/workers/placement_worker.py"
export WORLDCLAW_REFINEMENT_WORKER="$ROOT/envs/orchestrator/bin/python $SRC/workers/refinement_worker.py"
export WORLDCLAW_EXPORT_WORKER="$ROOT/envs/orchestrator/bin/python $SRC/workers/export_worker.py"
export BLENDER_BIN="$HOME/apps/blender-4.2.0-linux-x64/blender"
export WORLDCLAW_VLLM_URL=http://127.0.0.1:8000
export WORLDCLAW_VLM_URL=http://127.0.0.1:8001
# Planner and asset routing always use the OpenAI-compatible gpt-5.6-sol API.
# Mesh/render validation uses the OpenAI-compatible endpoint by default. Set
# WORLDCLAW_VALIDATION_PROVIDER=vllm only for an explicitly offline Qwen-VL
# validation run.
export WORLDCLAW_VALIDATION_PROVIDER="${WORLDCLAW_VALIDATION_PROVIDER:-openai}"
export OPENAI_VALIDATION_MODEL="${OPENAI_VALIDATION_MODEL:-gpt-5.6-sol}"
