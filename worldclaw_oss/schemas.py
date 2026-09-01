from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .asset_types import AssetType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Vec2(StrictModel):
    x: float
    y: float


class Polygon(StrictModel):
    points: list[Vec2] = Field(min_length=3)


class ObjectSpec(StrictModel):
    category: str = Field(min_length=1)
    count: int = Field(ge=0, le=10000)
    # ``category`` is the semantic entity (for example ``tree``).  The role
    # and instance strategy are planning hints; concrete representation and
    # generation method are selected later by asset routing/preflight.
    asset_role: AssetType | None = None
    instance_strategy: Literal["single", "scatter", "cluster", "path", "region"] = "scatter"
    # Density is a semantic modifier, not a universal object property.  Region
    # features such as lakes normally leave it unset.
    density: Literal["sparse", "medium", "dense"] | None = None
    appearance: str = ""

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_fields(cls, value):
        if not isinstance(value, dict):
            return value
        data = dict(value)
        # Old scene plans called the planner hint ``asset_type``.  Accept it
        # when reopening archived runs, but emit only the new ``asset_role``.
        if data.get("asset_role") in (None, "") and data.get("asset_type") not in (None, ""):
            data["asset_role"] = data["asset_type"]
        data.pop("asset_type", None)
        density = data.get("density")
        if isinstance(density, (int, float)):
            data["density"] = (
                "sparse" if float(density) <= 1.0 / 3.0
                else ("medium" if float(density) <= 2.0 / 3.0 else "dense")
            )
        elif isinstance(density, str):
            data["density"] = density.strip().lower()
        return data

    @property
    def asset_type(self) -> AssetType | None:
        """Compatibility accessor for workers and archived test fixtures."""
        return self.asset_role


class RegionPlan(StrictModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    function: str
    center: Vec2
    polygon: Polygon
    coverage: float = Field(gt=0.0, le=1.0)
    neighbors: list[str] = Field(default_factory=list)
    objects: list[ObjectSpec] = Field(default_factory=list)
    spatial_relations: list[str] = Field(default_factory=list)
    appearance: str = ""
    camera_hint: str = "wide"


class TerrainOperator(StrictModel):
    kind: Literal["peak", "dune", "terrace", "erosion"]
    strength: float = Field(ge=-100.0, le=100.0)
    scale: float = Field(gt=0.0)


class TerrainSpec(StrictModel):
    region_id: str
    mask_key: str
    base_height: float = Field(ge=-1000.0, le=1000.0)
    noise_octaves: list[float] = Field(default_factory=lambda: [1.0, 0.5, 0.25], min_length=1, max_length=8)
    operators: list[TerrainOperator] = Field(default_factory=list)
    boundary_blend: float = Field(default=0.12, gt=0.0, le=0.5)


class ScenePlan(StrictModel):
    theme: str
    world_size_m: tuple[float, float]
    regions: list[RegionPlan] = Field(min_length=3)
    terrain: list[TerrainSpec]
    materials: dict[str, str]
    explicit_constraints: list[str] = Field(default_factory=list)

    @field_validator("world_size_m")
    @classmethod
    def positive_size(cls, value: tuple[float, float]) -> tuple[float, float]:
        if value[0] <= 0 or value[1] <= 0:
            raise ValueError("world_size_m must be positive")
        return value

    @model_validator(mode="after")
    def references_are_consistent(self) -> "ScenePlan":
        ids = [r.id for r in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("region ids must be unique")
        known = set(ids)
        for region in self.regions:
            unknown = set(region.neighbors) - known
            if unknown or region.id in region.neighbors:
                raise ValueError(f"invalid neighbors for {region.id}: {sorted(unknown)}")
        if {t.region_id for t in self.terrain} != known:
            raise ValueError("terrain specs must cover every region exactly")
        coverage = sum(r.coverage for r in self.regions)
        if not 0.95 <= coverage <= 1.05:
            raise ValueError(f"region coverage must sum to approximately 1, got {coverage}")
        return self


class AssetInstance(StrictModel):
    id: str
    category: str
    asset_type: AssetType | None = None
    source_image: str
    mask: str
    mesh: str
    material: str
    transform_z_up: list[list[float]]
    region_id: str
    contact_ratio: float = Field(ge=0.0, le=1.0)
    source_model: str
    synthetic: bool = False

    @field_validator("transform_z_up")
    @classmethod
    def matrix_is_4x4(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) != 4 or any(len(row) != 4 for row in value):
            raise ValueError("transform_z_up must be 4x4")
        return value


class ModelRecord(StrictModel):
    model_id: str
    revision: str
    license: str
    size_bytes: int = Field(ge=0)
    purpose: str
    resolved: bool = True
    requested_model_id: str | None = None
    resolution_reason: str | None = None


class RunManifest(StrictModel):
    schema_version: str = "worldclaw-oss-run-v1"
    official_implementation: bool = False
    replacement_implementation: str = "worldclaw_oss"
    run_id: str
    prompt_sha256: str
    seed: int
    mode: Literal["live", "synthetic"]
    command: list[str]
    models: dict[str, ModelRecord]
    gpu: list[dict[str, Any]] = Field(default_factory=list)
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    elapsed_seconds: float | None = None
    errors: list[str] = Field(default_factory=list)
    retries: dict[str, int] = Field(default_factory=dict)
    stage_results: dict[str, dict[str, Any]] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)
    synthetic_outputs: bool = False


class Defect(StrictModel):
    asset_id: str | None = None
    kind: Literal["floating", "penetration", "overlap", "slope", "scale", "texture", "mesh", "other"]
    severity: float = Field(ge=0.0, le=1.0)
    recommendation: str


class DefectReport(StrictModel):
    defects: list[Defect] = Field(default_factory=list)
    acceptable: bool
    summary: str


class MeshValidationReport(StrictModel):
    """Asset-level quality gate result for one generated mesh."""
    status: Literal["pass", "fail"]
    severity: float = Field(ge=0.0, le=1.0)
    defects: list[str] = Field(default_factory=list)
    reason: str = ""
    retry_action: str = ""


class MeshValidationAssetResult(StrictModel):
    """Persisted per-asset result after validation and any route fallback."""
    asset_id: str | None = None
    status: Literal["pass", "fail"]
    final_status: Literal["pass", "fallback", "dropped"]
    fallback: str | None = None


class StructuralFeaturePlan(StrictModel):
    """Agent output describing a world-integrated feature, not a mesh asset."""
    feature_id: str
    region_id: str
    semantic_category: str
    generation_class: Literal["structural_feature"] = "structural_feature"
    structural_subtype: Literal["linear", "regional_surface"]
    representation: str
    instance_strategy: Literal["world_integrated"] = "world_integrated"
    geometry: dict[str, Any] = Field(default_factory=dict)
    region_relation: dict[str, Any] = Field(default_factory=dict)
    importance: str = "normal"


class StructuralValidationReport(StrictModel):
    status: Literal["pass", "fail"]
    structural_validation_failures: int = Field(ge=0)
    defects: list[str] = Field(default_factory=list)
    feature_checks: list[dict[str, Any]] = Field(default_factory=list)
    structural_retry_count: int = Field(ge=0, default=0)
    representation_specific: bool = True


class Stage(str, Enum):
    INTENT = "intent"
    PLAN = "plan"
    LAYOUT = "layout"
    TERRAIN = "terrain"
    STRUCTURAL_INPUT = "structural_input"
    STRUCTURAL_REPLAN = "structural_replan"
    STRUCTURAL_GENERATE = "structural_generate"
    STRUCTURAL_INTEGRATE = "structural_integrate"
    STRUCTURAL_VALIDATE = "structural_validate"
    ENV_ASSETS = "environment_assets"
    MESH_VALIDATE_ENV_ASSETS = "mesh_validation_environment_assets"
    REGION_COMPOSE = "region_composition"
    SEGMENT = "segment"
    RECONSTRUCT = "reconstruct"
    MESH_VALIDATE_RECONSTRUCT = "mesh_validation_reconstruct"
    PLACE = "place"
    REFINE = "refine"
    EXPORT = "export"
    STRUCTURAL_VIEW_PLAN = "structural_view_plan"
    STRUCTURAL_RENDER = "structural_render"
    STRUCTURAL_VISUAL_VALIDATE = "structural_visual_validate"
    STRUCTURAL_ADAPTIVE_RENDER = "structural_adaptive_render"
    STRUCTURAL_FINAL_VALIDATE = "structural_final_validate"
    VALIDATE = "validate"


def read_scene_plan(path: Path) -> ScenePlan:
    return ScenePlan.model_validate_json(path.read_text(encoding="utf-8"))
