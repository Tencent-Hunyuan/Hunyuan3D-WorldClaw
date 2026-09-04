#!/usr/bin/env python3
"""Continue Hunyuan3D's large files sequentially after an existing download."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


FILES = (
    (
        "hunyuan3d-paintpbr-v2-1/text_encoder/pytorch_model.bin",
        "c3e254d7b61353497ea0be2c4013df4ea8f739ee88cffa0ba58cd085459ed565",
        1361671895,
    ),
    (
        "hunyuan3d-paintpbr-v2-1/unet/diffusion_pytorch_model.bin",
        "675a1b5cd0098b2002637c443946529c03c5cd54427f40245263350feb3dd5b8",
        3925293863,
    ),
)


def wait_for_pid(pid: int) -> None:
    while Path(f"/proc/{pid}").exists():
        time.sleep(20)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--hf-home", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--proxy", required=True)
    args = parser.parse_args()
    wait_for_pid(args.wait_pid)
    range_script = Path(__file__).with_name("download_hf_range.py")
    repo = "tencent/Hunyuan3D-2.1"
    revision = "0b94677654c57bb9a6b6845cd7b704ccf551d327"
    blob_dir = args.cache_root / "models--tencent--Hunyuan3D-2.1" / "blobs"
    env = os.environ.copy()
    env.update({"HTTPS_PROXY": args.proxy, "HTTP_PROXY": args.proxy, "HF_HOME": str(args.hf_home)})
    for filename, sha256, size in FILES:
        output = blob_dir / f"{sha256}.incomplete"
        command = [
            sys.executable, str(range_script), "--repo", repo, "--revision", revision,
            "--filename", filename, "--output", str(output), "--size", str(size),
            "--sha256", sha256, "--chunk-size", str(67108864),
            "--finalize-snapshot", "--hf-home", str(args.hf_home),
        ]
        subprocess.run(command, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
