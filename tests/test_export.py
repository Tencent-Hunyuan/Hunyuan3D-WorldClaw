import json
import struct

import numpy as np
import pytest

from worldclaw_oss.export import (
    box_mesh,
    orient_terrain_triangles_upward,
    vertex_normals,
    write_glb,
)


def test_vertex_normals_are_unit_and_upward_for_flat_mesh():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    normals = vertex_normals(vertices, triangles)
    assert np.allclose(np.linalg.norm(normals, axis=1), 1.0)
    assert np.all(normals[:, 2] > 0.0)


def test_glb_primitives_include_normal_accessor(tmp_path):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
    path = tmp_path / "scene.glb"
    write_glb(path, [{"name": "Terrain", "category": "terrain", "vertices": vertices, "triangles": triangles}])
    raw = path.read_bytes()
    json_length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + json_length].decode("utf-8"))
    attributes = document["meshes"][0]["primitives"][0]["attributes"]
    assert "POSITION" in attributes
    assert "NORMAL" in attributes
    normal_accessor = document["accessors"][attributes["NORMAL"]]
    assert normal_accessor["type"] == "VEC3"
    assert normal_accessor["count"] == len(vertices)


def test_terrain_export_orientation_flips_reversed_internal_mesh():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    reversed_triangles = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
    triangles, stats = orient_terrain_triangles_upward(vertices, reversed_triangles)
    assert stats["flipped"] is True
    assert stats["mean_normal_z"] > 0.0
    assert stats["upward_face_ratio"] > 0.99
    assert np.all(vertex_normals(vertices, triangles)[:, 2] > 0.0)


def test_terrain_export_orientation_rejects_mixed_winding():
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    mixed = np.array([[0, 1, 2], [0, 3, 2]], dtype=np.uint32)
    with pytest.raises(ValueError, match="Terrain winding validation failed"):
        orient_terrain_triangles_upward(vertices, mixed)


def test_glb_reuses_identical_meshes_and_preserves_node_transforms(tmp_path):
    vertices, triangles = box_mesh()
    transforms = []
    for index in range(3):
        matrix = np.eye(4)
        matrix[0, 0] = 1.0 + index
        matrix[1, 1] = 1.0 + index
        matrix[2, 2] = 2.0
        matrix[:3, 3] = (index * 4.0, 0.0, 1.0)
        transforms.append(matrix.tolist())
    path = tmp_path / "instances.glb"
    write_glb(path, [
        {"name": f"Tree_{i}", "category": "tree", "vertices": vertices,
         "triangles": triangles, "transform_z_up": transform}
        for i, transform in enumerate(transforms)
    ])
    raw = path.read_bytes()
    json_length = struct.unpack_from("<I", raw, 12)[0]
    document = json.loads(raw[20 : 20 + json_length].decode("utf-8"))
    assert len(document["meshes"]) == 1
    assert len(document["nodes"]) == 3
    assert all(node["mesh"] == 0 for node in document["nodes"])
    assert document["nodes"][1]["extras"]["mesh_instance"] is True
    # glTF matrices are column-major; the X translation remains visible at
    # index 12 after the internal Z-up conversion.
    assert document["nodes"][2]["matrix"][12] == 8.0
