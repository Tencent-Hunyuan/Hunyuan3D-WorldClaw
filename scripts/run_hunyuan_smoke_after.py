#!/usr/bin/env python3
"""Run the Hunyuan fixture smoke after a sequential model download exits."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--orchestrator-python", type=Path, required=True)
    parser.add_argument("--hunyuan-python", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--online", action="store_true", help="allow HF metadata lookups through the configured proxy")
    args = parser.parse_args()
    while Path(f"/proc/{args.wait_pid}").exists():
        time.sleep(30)

    nvidia = args.hunyuan_python.parent.parent / "lib" / "python3.10" / "site-packages" / "nvidia"
    bpy_lib = args.hunyuan_python.parent.parent / "lib" / "python3.10" / "site-packages" / "bpy" / "lib"
    torch_lib = args.hunyuan_python.parent.parent / "lib" / "python3.10" / "site-packages" / "torch" / "lib"
    library_dirs = [bpy_lib, torch_lib] + [nvidia / name / "lib" for name in (
        "cudnn", "cublas", "cuda_runtime", "cuda_nvrtc", "cufft", "curand",
        "cusolver", "cusparse", "cusparselt", "nccl", "nvjitlink", "nvtx",
    )]
    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(args.source_root),
        "WORLDCLAW_MODEL_ROOT": str(args.model_root),
        "HF_HOME": str(args.model_root / "hf"),
        "HY3DGEN_MODELS": str(args.model_root / "hy3dgen-cache"),
        "HUNYUAN3D_SOURCE": str(args.model_root / "sources" / "Hunyuan3D-2.1"),
        "WORLDCLAW_IMAGE3D_WORKER": (
            f"env LD_LIBRARY_PATH={':'.join(map(str, library_dirs))} "
            f"CUDA_VISIBLE_DEVICES=4 {args.hunyuan_python} "
            f"{args.source_root / 'workers' / 'hunyuan3d_worker.py'}"
        ),
    })
    if not args.online:
        env.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    command = [
        str(args.orchestrator_python), str(args.source_root / "scripts" / "hunyuan_worker_smoke.py"),
        "--root", str(args.root), "--lock", str(args.lock), "--image", str(args.image),
    ]
    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
