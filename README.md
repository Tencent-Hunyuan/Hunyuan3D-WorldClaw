# worldclaw_oss

worldclaw_oss is an auditable open-model replacement workflow inspired by the
three stages described in arXiv:2608.05248. It is **not** Tencent's official
WorldClaw implementation and does not claim the paper's visual quality.
Week0/ remains the original dependency-free procedural baseline.

## Current Status

The repository contains:

- strict Pydantic schemas for plans, terrain, assets, defects, and manifests;
- separate intent/planning prompts and a vLLM JSON-schema client with at most
  three repair attempts;
- deterministic semantic masks and soft-boundary procedural terrain;
- camera/crop intrinsics, ray/mesh intersection, scale calibration, Z-up/glTF
  conversion, contact scoring, and joint depth/scale search;
- content-addressed asset cache and a twelve-stage SQLite state machine;
- isolated command-worker boundaries for FLUX, Grounding DINO + SAM 2.1,
  Hunyuan3D/TRELLIS, placement, refinement, and export;
- a concrete FLUX diffusers worker with the authorized dev-to-schnell fallback;
- Blender 4.2 finalization with independent nodes, diagnostic views, a Blend,
  GLB, preview, and 12-second 720p walkthrough. Repeated prototype geometry is
  exported as shared glTF Mesh data with per-node transforms; the finalizer
  also links duplicate Blender Mesh datablocks and adds conservative render
  LOD modifiers for expensive vegetation/rocks.
- a synthetic mode for deterministic tests. Synthetic artifacts are explicitly
  marked and can never satisfy model-inference acceptance.

Live execution is currently blocked by unresolved model commits. Public
Hugging Face metadata requests timed out from both the Windows control host and
ziplab-5090; models.lock.json deliberately uses UNRESOLVED instead of invented
revisions. The live CLI fails before inference until the lock is resolved.

## CLI

    python -m worldclaw_oss generate \
      --prompt-file prompts/castle.txt \
      --seed 42 \
      --output runs/castle \
      --mode live

    python -m worldclaw_oss resume --run-id castle --runs-root runs
    python -m worldclaw_oss validate --run runs/castle --full

For core algorithm and CI verification only:

    python -m worldclaw_oss generate \
      --prompt-file prompts/castle.txt \
      --seed 42 \
      --output outputs/oss_castle_synthetic \
      --mode synthetic

Live planning currently targets a compact `100 m x 100 m` world. The planner
first normalizes each object into a singular semantic `category` plus the
required `count`, `instance_strategy`, and `asset_role`. Optional `density` and
`appearance` are emitted only when supported by the description or entity
type; for example, dense trees retain `density=dense`, while a lake has no
`density` field. Asset routing and preflight then choose the concrete
representation/generation method without renaming that entity. Horizontal
coordinates and terrain operator scales are rescaled, and operational object
counts are scaled by world area so placement remains feasible. Override the target for a later experiment with
`WORLDCLAW_TARGET_WORLD_SIZE_M=WIDTH,DEPTH` (for example, `100,100`); the
normalized and authored sizes are recorded in the plan-stage manifest payload.

Terrain-dependent objects use a separate structural-feature branch. Trails,
roads, rivers, lakes, shorelines, and similar categories are normalized to the
high-level `structural_feature` route (with `linear` or `regional_surface`
subtypes). `StructuralFeatureAgent` emits planning parameters; deterministic
numpy geometry then writes world-coordinate meshes, optional basin/channel
terrain edits, occupied/exclusion masks, surface-type masks, and placement
constraints. Trail integration additionally persists a width-derived
`trail_clearance_field.npy`, hard `trail_exclusion_mask.npy`, and independent
`structural_placement_weights.npy`; these influence placement without rewriting
the base `layout_weights.npy`. Lake basin correction records a water level and
exports its water surface as a separate structural mesh. Structural validation
is representation-specific and does not invoke Hunyuan3D, mesh retry, or Stage
3 fallback. The placement barrier reads `terrain_structural.npz`, combines
continuous layout/influence weights with hard semantic gates, and excludes
structural masks before scattering normal assets; export merges both branches
into one coordinate-consistent scene.

## Remote Setup

The experiment host has no conda command at present. Install Conda/Miniforge
in a user-owned location, then create the four environments in environments/.
Do not install model dependencies into the root filesystem.

    export WORLDCLAW_MODEL_ROOT=/mnt/data/v-huguangyu/worldclaw-oss
    python scripts/resolve_model_lock.py --lock models.lock.json
    bash scripts/remote_preflight.sh models.lock.json

`scripts/configure_remote_workers.sh` exports
`WORLDCLAW_OPENAI_AUTH_FILE` to the remote user's `~/.codex/auth.json` by
default and uses the ziplab OpenAI-compatible gateway
`https://sub2api-hk.ziplab.co/v1` unless `OPENAI_BASE_URL` is overridden.

    conda env create -f environments/orchestrator.yml
    conda env create -f environments/llm-vlm.yml
    conda env create -f environments/image-seg.yml
    conda env create -f environments/hunyuan3d.yml

remote_preflight.sh redirects HF_HOME, TORCH_HOME, XDG_CACHE_HOME,
PIP_CACHE_DIR, and TMPDIR to NFS. It rejects an unresolved lock and, when
estimated weights exceed 90 GB, requires NFS free space of at least twice the
estimate. HF_TOKEN is read only from the environment; its value must never be
written to this repository or logs.

Each non-vLLM model environment exposes a JSON file contract through:

    WORLDCLAW_FLUX_WORKER
    WORLDCLAW_REFERENCE_IMAGE_WORKER
    WORLDCLAW_RECON_PREFLIGHT_WORKER
    WORLDCLAW_IMAGE3D_WORKER
    WORLDCLAW_SEGMENT_WORKER
    WORLDCLAW_PLACEMENT_WORKER
    WORLDCLAW_REFINEMENT_WORKER
    WORLDCLAW_EXPORT_WORKER

The command receives --request REQUEST.json --response RESPONSE.json.
Successful responses must contain a status value of ok. A failed stage is
persisted and retried up to three times; resume skips completed stages.

Environment asset reference images and regional terrain composition both use
the `WORLDCLAW_FLUX_WORKER` path by default. The reference worker remains
overrideable for experiments, but the standard workflow does not call GPT
Image 2.

Before either `environment_assets` or `reconstruction` invokes Hunyuan3D,
`WORLDCLAW_RECON_PREFLIGHT_WORKER` runs local crop/mask/bbox gates and a
structured GPT/VLM reconstructability review. It can request one bounded
reference prompt rewrite, one recrop/resegment pass, or reroute an input to a
procedural representation. Post-generation mesh validation remains the
unexpected-failure safety net. Configure `WORLDCLAW_PREFLIGHT_REWRITE_MAX=1`
and `WORLDCLAW_PREFLIGHT_RESEGMENT_MAX=1` to retain the default bounds.

Mesh validation Stage 1 now defaults to one reconstruction retry from an
immutable initial request with a new seed. Stage 2 prompt/input regeneration
also defaults to one pass; `WORLDCLAW_MESH_RETRY_MAX` and
`WORLDCLAW_MESH_INPUT_REGEN_MAX` can lower these limits but are bounded by the
worker.

Hunyuan3D image-to-mesh calls use a persistent daemon pool when the configured
worker command points to `hunyuan3d_worker.py` (the default is `auto` on Linux).
The first call mounts one daemon per selected GPU; later environment,
reconstruction, retry, and refinement calls reuse those resident pipelines.
When multiple GPUs are available, source images are sharded and processed in
parallel. Configure `WORLDCLAW_HUNYUAN_GPUS=4,5` to pin devices,
`WORLDCLAW_HUNYUAN_EXTRA_GPUS=6,7` to add explicit devices, or set
`WORLDCLAW_HUNYUAN_DAEMON_POOL=0` to retain one-shot workers. The daemon
sockets live under `/tmp/worldclaw-hunyuan` by default and can be relocated
with `WORLDCLAW_HUNYUAN_SOCKET_DIR`. Automatic discovery selects GPUs with at
least 12 GB free; override this with `WORLDCLAW_HUNYUAN_MIN_FREE_MB` or exclude
devices with `WORLDCLAW_HUNYUAN_GPU_EXCLUDE`.

Placement reconstruction uses a bounded four-worker process pool by default
for jobs with at least eight assets. Set `WORLDCLAW_PLACEMENT_WORKERS=4` to
choose a worker count (capped at eight), or `0`/`1` to force serial placement.
The placement response records the effective worker count and whether the pool
was used.

Mesh validation and render-feedback refinement default to the OpenAI-compatible
validation model `gpt-5.6-sol`:

    export WORLDCLAW_VALIDATION_PROVIDER=openai
export OPENAI_VALIDATION_MODEL=gpt-5.6-sol

Intent planning and asset routing are also explicitly forced to the
OpenAI-compatible `gpt-5.6-sol` model. `OPENAI_TEXT_MODEL` does not override
this planner/router selection, and the legacy local Qwen3 planner is disabled.

The OpenAI key and endpoint are read from the normal environment or user-owned
Codex configuration. To explicitly use the locked local Qwen-VL checker for
validation, set `WORLDCLAW_VALIDATION_PROVIDER=vllm`; this does not affect the
GPT-5.6 Sol planner/router.

## Tests

    python -m pytest -q

## Render QA and final video

The Blender finalizer defaults to the low-cost `diagnostic` profile during
testing. It renders four 640x360 views and does not create a video. Before
publishing a walkthrough, run the `full` profile explicitly:

    blender --background --python scripts/blender_finalize.py -- \
      --run runs/live_forest --profile diagnostic

This writes four 640x360 diagnostic views and `blender_metadata.json`, but no
MP4. Only after diagnostic QA passes, render the final walkthrough once:

    blender --background --python scripts/blender_finalize.py -- \
      --run runs/live_forest --profile full

`WORLDCLAW_RENDER_PROFILE=diagnostic|full` can be used when the finalizer is
launched by the pipeline. The pipeline passes this profile explicitly, so a
full video is never produced accidentally during diagnostic runs.
`WORLDCLAW_LOD_MIN_POLYGONS` controls the LOD gate,
and `WORLDCLAW_ATMOSPHERE=1` enables an opt-in world volume (it costs more per
frame). Metadata records shot beats, camera bounds, mesh reuse, estimated
render polygons, and whether video was generated.

The fixed synthetic audit outputs are under outputs/oss_*_synthetic.
outputs/oss_castle_synthetic additionally contains a Blend and MP4 rendered
on ziplab-5090. See [REPORT.md](REPORT.md) for exact verified and unverified
scope.
