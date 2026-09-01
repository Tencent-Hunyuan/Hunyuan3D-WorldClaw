#!/usr/bin/env bash
set -eo pipefail

ROOT=/mnt/data/v-huguangyu/worldclaw-oss
if [ -n "$WORLDCLAW_MODEL_ROOT" ]; then ROOT="$WORLDCLAW_MODEL_ROOT"; fi
SRC="$ROOT/src"
if [ -n "$WORLDCLAW_SOURCE_ROOT" ]; then SRC="$WORLDCLAW_SOURCE_ROOT"; fi
PROXY=http://127.0.0.1:17890
if [ -n "$WORLDCLAW_REMOTE_PROXY" ]; then PROXY="$WORLDCLAW_REMOTE_PROXY"; fi
COMPONENTS="$*"
if [ -z "$COMPONENTS" ]; then COMPONENTS="orchestrator llm image hunyuan trellis"; fi
set -u
MINIFORGE_TAG=26.3.2-3
MINIFORGE_NAME=Miniforge3-26.3.2-3-Linux-x86_64.sh
MINIFORGE_SHA256=848194851a98903134187fbb4ab50efe87b003e0c0f808f97644b7524a62bf2c
MINIFORGE_URL="https://github.com/conda-forge/miniforge/releases/download/$MINIFORGE_TAG/$MINIFORGE_NAME"

mkdir -p "$ROOT"/{tmp,pip,xdg,hf,torch,conda-pkgs,envs,sources,environment-locks,logs}
chmod 700 "$ROOT" "$ROOT"/{tmp,pip,xdg,hf,torch,conda-pkgs,envs,sources,environment-locks,logs}
export HTTP_PROXY="$PROXY"
export HTTPS_PROXY="$PROXY"
export http_proxy="$PROXY"
export https_proxy="$PROXY"
export NO_PROXY="localhost,127.0.0.1"
export HF_HOME="$ROOT/hf"
export TORCH_HOME="$ROOT/torch"
export XDG_CACHE_HOME="$ROOT/xdg"
export PIP_CACHE_DIR="$ROOT/pip"
export TMPDIR="$ROOT/tmp"
export CONDA_PKGS_DIRS="$ROOT/conda-pkgs"
export PYTHONNOUSERSITE=1
# The host's global pip configuration points at a mirror whose TLS endpoint
# is not reachable through the localhost proxy. Keep the source overridable.
export PIP_INDEX_URL="${WORLDCLAW_PIP_INDEX_URL:-https://pypi.org/simple}"

INSTALLER="$ROOT/tmp/$MINIFORGE_NAME"
if [ ! -x "$ROOT/miniforge3/bin/conda" ]; then
  curl -fL --retry 5 --retry-all-errors -x "$PROXY" "$MINIFORGE_URL" -o "$INSTALLER"
  echo "$MINIFORGE_SHA256  $INSTALLER" | sha256sum -c -
  bash "$INSTALLER" -b -p "$ROOT/miniforge3"
fi
CONDA="$ROOT/miniforge3/bin/conda"
"$CONDA" config --system --set auto_activate_base false

has_component() {
  case " $COMPONENTS " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}

create_base() {
  local env_path="$1"
  local python_version="$2"
  if [ ! -x "$env_path/bin/python" ]; then
    "$CONDA" create -y -p "$env_path" "python=$python_version" pip
  fi
}

record_env() {
  local name="$1"
  local env_path="$2"
  "$CONDA" list -p "$env_path" --explicit > "$ROOT/environment-locks/$name.conda-explicit.txt"
  "$env_path/bin/python" -m pip freeze --all > "$ROOT/environment-locks/$name.pip-freeze.txt"
}

if has_component orchestrator; then
  ENV="$ROOT/envs/orchestrator"
  create_base "$ENV" 3.11
  "$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV/bin/python" -m pip install --no-build-isolation --prefer-binary -e "$SRC[test,live]" \
    "trimesh==4.4.7" "open3d==0.19.0" pillow requests huggingface-hub
  record_env orchestrator "$ENV"
fi

if has_component llm; then
  ENV="$ROOT/envs/llm-vlm"
  create_base "$ENV" 3.11
  "$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV/bin/python" -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" \
    --index-url https://download.pytorch.org/whl/cu128
  "$ENV/bin/python" -m pip install \
    "transformers==4.57.1" "accelerate==1.10.1" "huggingface-hub==0.36.2" \
    sentencepiece protobuf safetensors regex requests tqdm \
    "fastapi[standard]>=0.115,<0.137" "aiohttp" "openai<=1.90,>=1.52" \
    "prometheus_client>=0.18" "prometheus-fastapi-instrumentator>=7" \
    tiktoken "lm-format-enforcer>=0.10.11,<0.11" "outlines==0.1.11" \
    lark "xgrammar==0.1.19" "pyzmq>=25" msgspec gguf pyyaml six einops \
    compressed-tensors==0.10.2 depyf==0.18.0 cloudpickle watchfiles \
    python-json-logger scipy ninja pybase64 ray[cgraph] py-cpuinfo blake3 \
    partial-json-parser mistral_common opencv-python-headless cachetools \
    psutil
  "$ENV/bin/python" -m pip install --no-deps "vllm==0.9.2"
  record_env llm-vlm "$ENV"
fi

if has_component image; then
  ENV="$ROOT/envs/image-seg"
  create_base "$ENV" 3.11
  "$ENV/bin/python" -m pip install --upgrade pip
  "$ENV/bin/python" -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
  "$ENV/bin/python" -m pip install \
    "diffusers==0.35.2" "transformers==4.57.1" "accelerate==1.10.1" \
    sentencepiece protobuf pillow safetensors
  record_env image-seg "$ENV"
fi

if has_component hunyuan; then
  ENV="$ROOT/envs/hunyuan3d"
  create_base "$ENV" 3.10
  "$CONDA" install -y -p "$ENV" -c nvidia/label/cuda-12.8.1 -c conda-forge \
    cuda-nvcc=12.8 cuda-cudart-dev=12.8 gxx_linux-64=12
  "$ENV/bin/python" -m pip install --upgrade pip
  "$ENV/bin/python" -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" "torchaudio==2.7.1" \
    --index-url https://download.pytorch.org/whl/cu128
  "$ENV/bin/python" -m pip install -r "$SRC/environments/hunyuan3d-inference.txt"
  HYSRC="$ROOT/sources/Hunyuan3D-2.1"
  if [ ! -d "$HYSRC/.git" ]; then
    git -c http.proxy="$PROXY" clone \
      https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git "$HYSRC"
  fi
  git -C "$HYSRC" -c http.proxy="$PROXY" fetch --all
  git -C "$HYSRC" checkout --detach 82920d643c0dc2f7bfd7255f45f62d386edfe60c
  export CUDA_HOME="$ENV"
  export PATH="$ENV/bin:$PATH"
  set +u
  export LD_LIBRARY_PATH="$ENV/lib:$LD_LIBRARY_PATH"
  set -u
  "$ENV/bin/python" -m pip install -e "$HYSRC/hy3dpaint/custom_rasterizer"
  (
    cd "$HYSRC/hy3dpaint/DifferentiableRenderer"
    bash compile_mesh_painter.sh
  )
  ESRGAN="$HYSRC/hy3dpaint/ckpt/RealESRGAN_x4plus.pth"
  if [ ! -f "$ESRGAN" ]; then
    mkdir -p "$(dirname "$ESRGAN")"
    curl -fL --retry 5 --retry-all-errors -x "$PROXY" \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth \
      -o "$ESRGAN"
  fi
  record_env hunyuan3d "$ENV"
fi

if has_component trellis; then
  ENV="$ROOT/envs/trellis"
  create_base "$ENV" 3.11
  "$ENV/bin/python" -m pip install --upgrade pip setuptools wheel
  "$ENV/bin/python" -m pip install \
    "torch==2.7.1" "torchvision==0.22.1" --index-url https://download.pytorch.org/whl/cu128
  "$ENV/bin/python" -m pip install --prefer-binary \
    "transformers==4.57.1" "diffusers==0.35.2" "accelerate==1.10.1" \
    pillow safetensors einops omegaconf imageio
  TRSRC="$ROOT/sources/TRELLIS"
  if [ ! -d "$TRSRC/.git" ]; then
    git -c http.proxy="$PROXY" clone https://github.com/microsoft/TRELLIS.git "$TRSRC"
  fi
  git -C "$TRSRC" -c http.proxy="$PROXY" fetch --all
  git -C "$TRSRC" checkout --detach 442aa1e1afb9014e80681d3bf604e8d728a86ee7
  UTILSRC="$ROOT/sources/utils3d"
  if [ ! -d "$UTILSRC/.git" ]; then
    git -c http.proxy="$PROXY" clone https://github.com/EasternJournalist/utils3d.git "$UTILSRC"
  fi
  git -C "$UTILSRC" -c http.proxy="$PROXY" fetch --all
  git -C "$UTILSRC" checkout --detach 9a4eb15e4021b67b12c460c7057d642626897ec8
  if [ -f "$TRSRC/requirements.txt" ]; then
    "$ENV/bin/python" -m pip install --prefer-binary -r "$TRSRC/requirements.txt"
  fi
  "$ENV/bin/python" -m pip install --no-build-isolation -e "$TRSRC" -e "$UTILSRC"
  record_env trellis "$ENV"
fi

echo "BOOTSTRAP_OK components=$COMPONENTS root=$ROOT"
