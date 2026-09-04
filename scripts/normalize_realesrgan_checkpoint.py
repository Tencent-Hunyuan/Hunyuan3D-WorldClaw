#!/usr/bin/env python3
"""Wrap a bare RealESRGAN state dict for the Hunyuan paint loader."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    value = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(value, dict) and ("params" in value or "params_ema" in value):
        return 0
    temporary = args.checkpoint.with_suffix(args.checkpoint.suffix + ".tmp")
    torch.save({"params_ema": value}, temporary)
    temporary.replace(args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
