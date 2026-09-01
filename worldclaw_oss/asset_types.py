"""Dependency-light asset route vocabulary shared by isolated workers."""
from enum import Enum


class AssetType(str, Enum):
    # Canonical high-level route for terrain-dependent world structures.
    STRUCTURAL_FEATURE = "structural_feature"
    SOLID_OBJECT = "solid_object"
    REUSABLE_PROTOTYPE = "reusable_prototype"
    PROCEDURAL_NATIVE = "procedural_native"
    SCATTER_DETAIL = "scatter_detail"
    LINEAR_STRUCTURE = "linear_structure"
    SURFACE_REGION_FEATURE = "surface_region_feature"

    @classmethod
    def _missing_(cls, value):
        """Read route names from archived plans while emitting canonical names."""
        legacy = {
            "tree_prototype": cls.REUSABLE_PROTOTYPE,
            "vegetation_cluster": cls.PROCEDURAL_NATIVE,
            "linear_structure": cls.STRUCTURAL_FEATURE,
            "surface_region_feature": cls.STRUCTURAL_FEATURE,
        }
        return legacy.get(str(value).strip().lower())
