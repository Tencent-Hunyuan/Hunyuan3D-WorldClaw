from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import Pipeline, new_run_id
from .schemas import RunManifest
from .validation import validate_run


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        prog="python -m worldclaw_oss",
        description="Open-model WorldClaw replacement (not official WorldClaw)",
    )
    sub = value.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate")
    generate.add_argument("--prompt-file", type=Path, required=True)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--output", type=Path)
    generate.add_argument("--mode", choices=("live", "synthetic"), default="live")
    generate.add_argument("--models-lock", type=Path, default=Path("models.lock.json"))
    generate.add_argument("--cache-root", type=Path)
    resume = sub.add_parser("resume")
    resume.add_argument("--run-id", required=True)
    resume.add_argument("--runs-root", type=Path, default=Path("runs"))
    resume.add_argument("--models-lock", type=Path, default=Path("models.lock.json"))
    resume.add_argument("--cache-root", type=Path)
    validate = sub.add_parser("validate")
    validate.add_argument("--run", type=Path, required=True)
    validate.add_argument("--full", action="store_true")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "validate":
        report = validate_run(args.run, args.full)
        print(json.dumps(report, indent=2))
        raise SystemExit(0 if report["valid"] else 1)
    if args.command == "generate":
        prompt = args.prompt_file.read_text(encoding="utf-8").strip()
        output = args.output or Path("runs") / new_run_id(prompt)
        pipe = Pipeline(
            output, args.models_lock, args.mode, args.seed, prompt,
            [sys.executable, "-m", "worldclaw_oss", *sys.argv[1:]], args.cache_root,
        )
    else:
        output = args.runs_root / args.run_id
        manifest = RunManifest.model_validate_json(
            (output / "run_manifest.json").read_text(encoding="utf-8")
        )
        prompt_path = output / "prompt.txt"
        if prompt_path.is_file():
            prompt = prompt_path.read_text(encoding="utf-8").strip()
        else:
            command = manifest.command
            try:
                prompt_arg = command[command.index("--prompt-file") + 1]
                candidate = Path(prompt_arg)
                prompt = candidate.read_text(encoding="utf-8").strip()
            except (ValueError, IndexError, OSError) as exc:
                raise RuntimeError(
                    f"cannot resume {output}: prompt.txt and original --prompt-file are unavailable"
                ) from exc
        pipe = Pipeline(
            output, args.models_lock, manifest.mode, manifest.seed,
            prompt, manifest.command, args.cache_root, allow_retry_exhausted=True,
        )
    result = pipe.run()
    print(json.dumps({
        "run_id": pipe.run_id, "run": str(result), "mode": pipe.mode,
        "official_implementation": False,
    }, indent=2))
