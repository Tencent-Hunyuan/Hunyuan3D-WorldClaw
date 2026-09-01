# WorldClaw OSS Replacement

This directory is an independent, auditable replacement workflow inspired by
the public WorldClaw paper. It is not the official Tencent implementation and
does not claim the paper's visual quality, model weights, or internal code.

## Contents

- `worldclaw_oss/`: schemas, intent/planning, terrain and layout generation,
  structural-feature geometry, validation, caching, state management, export,
  and the pipeline CLI.
- `workers/`: isolated JSON worker contracts for image generation,
  segmentation, reconstruction, placement, refinement, validation, and export.
- `scripts/`: Blender finalization/inspection tools and deterministic audit
  helpers.
- `tests/`: unit and contract tests for the replacement implementation.
- `prompts/` and `environments/`: small example inputs and environment
  descriptors. Model weights are never stored in this repository.

## Quick start

Requires Python 3.10 or newer. Install the package and test dependencies:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

The deterministic synthetic path is intended for local verification and does
not invoke model inference:

```bash
python -m worldclaw_oss generate \
  --prompt-file prompts/forest_lake.txt \
  --seed 42 \
  --output outputs/oss_forest_synthetic \
  --mode synthetic
```

The live path requires the corresponding model environments and credentials in
the user's environment. Credentials, caches, generated runs, and model
weights are deliberately excluded from version control. See the worker source
and `models.lock.json`/`sources.lock.json` for the pinned interfaces and
provenance fields.

## Current implementation boundary

The replacement keeps terrain-dependent features on a structural branch. A
trail, road, river, lake, or shoreline is planned and generated as a
world-coordinate procedural surface with explicit occupancy and exclusion
masks. Ordinary asset placement consumes those masks, while export merges both
branches and validates terrain orientation. Mesh validation uses bounded
retry/fallback decisions and records the terminal result; a dropped raw asset
cannot be revived by placement.

The 100 m compact-world path normalizes authored plans while preserving
semantic category, count, density, and routing fields. The Blender finalizer
supports a diagnostic render profile for QA and an explicit full walkthrough
profile. Synthetic artifacts are marked as synthetic and cannot satisfy live
model-inference acceptance.

## Reproducibility

Meaningful runs should record the source revision, command, provider/model,
versions, seed, and output path in their run manifest. Generated output belongs
under an ignored `outputs/` or `runs/` directory and is not source-of-truth
code. The repository does not include private hostnames, user paths, auth
files, or secret values.
