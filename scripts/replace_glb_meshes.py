"""Replace named Terrain/structural mesh buffers in an existing GLB.

This is used for bounded structural regressions when upstream external
stages are already materialized but a deterministic mesh implementation has
changed.  Node transforms, materials, and all unrelated asset meshes remain
untouched.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

from worldclaw_oss.export import vertex_normals


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--terrain", type=Path, required=True)
    parser.add_argument("--river", type=Path, required=True)
    return parser.parse_args()


def _chunks(blob: bytes) -> tuple[dict, bytearray]:
    if len(blob) < 20 or blob[:4] != b"glTF":
        raise ValueError("source is not a GLB")
    version, declared_length = struct.unpack_from("<II", blob, 4)
    if version != 2 or declared_length != len(blob):
        raise ValueError("invalid GLB header")
    offset = 12
    json_chunk = None
    binary = None
    while offset < len(blob):
        length, kind = struct.unpack_from("<II", blob, offset)
        payload = blob[offset + 8:offset + 8 + length]
        if kind == 0x4E4F534A:
            json_chunk = json.loads(payload.decode("utf-8"))
        elif kind == 0x004E4942:
            binary = bytearray(payload)
        offset += 8 + length
    if json_chunk is None or binary is None:
        raise ValueError("GLB must contain JSON and BIN chunks")
    return json_chunk, binary


def _encode_mesh(document: dict, binary: bytearray, mesh_index: int, vertices: np.ndarray, triangles: np.ndarray) -> None:
    mesh = document["meshes"][mesh_index]
    if len(mesh.get("primitives", [])) != 1:
        raise ValueError(f"mesh {mesh_index} must contain one primitive")
    primitive = mesh["primitives"][0]
    attributes = primitive["attributes"]
    position_accessor = document["accessors"][attributes["POSITION"]]
    position_view = document["bufferViews"][position_accessor["bufferView"]]
    count = int(position_accessor["count"])
    values = np.asarray(vertices, dtype=np.float32)
    if values.shape != (count, 3):
        raise ValueError(f"mesh {mesh_index} vertex count mismatch: {values.shape} != {(count, 3)}")
    # Internal coordinates are Z-up; GLB stores Y-up as (x, z, -y).
    encoded = np.column_stack((values[:, 0], values[:, 2], -values[:, 1])).astype("<f4")
    position_offset = int(position_view.get("byteOffset", 0)) + int(position_accessor.get("byteOffset", 0))
    position_bytes = encoded.tobytes()
    if len(position_bytes) != int(position_view["byteLength"]):
        raise ValueError("position buffer view is not tightly packed")
    binary[position_offset:position_offset + len(position_bytes)] = position_bytes
    position_accessor["min"] = encoded.min(axis=0).tolist()
    position_accessor["max"] = encoded.max(axis=0).tolist()

    if "NORMAL" in attributes:
        normal_accessor = document["accessors"][attributes["NORMAL"]]
        normal_view = document["bufferViews"][normal_accessor["bufferView"]]
        normals = vertex_normals(values, np.asarray(triangles, dtype=np.uint32))
        encoded_normals = np.column_stack((normals[:, 0], normals[:, 2], -normals[:, 1])).astype("<f4")
        normal_offset = int(normal_view.get("byteOffset", 0)) + int(normal_accessor.get("byteOffset", 0))
        normal_bytes = encoded_normals.tobytes()
        if len(normal_bytes) != int(normal_view["byteLength"]):
            raise ValueError("normal buffer view is not tightly packed")
        binary[normal_offset:normal_offset + len(normal_bytes)] = normal_bytes
        normal_accessor["min"] = encoded_normals.min(axis=0).tolist()
        normal_accessor["max"] = encoded_normals.max(axis=0).tolist()


def main() -> None:
    args = _args()
    document, binary = _chunks(args.source.read_bytes())
    nodes = {str(node.get("name")): node for node in document.get("nodes", [])}
    terrain_node = nodes.get("Terrain")
    river_node = nodes.get("river_river_000")
    if terrain_node is None or river_node is None:
        raise ValueError("source GLB must contain Terrain and river_river_000 nodes")
    terrain = np.load(args.terrain)
    river = np.load(args.river)
    _encode_mesh(document, binary, int(terrain_node["mesh"]), terrain["vertices"], terrain["triangles"])
    _encode_mesh(document, binary, int(river_node["mesh"]), river["vertices"], river["triangles"])
    json_payload = json.dumps(document, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    json_payload += b" " * ((4 - len(json_payload) % 4) % 4)
    binary_payload = bytes(binary)
    total = 12 + 8 + len(json_payload) + 8 + len(binary_payload)
    output = struct.pack("<4sII", b"glTF", 2, total)
    output += struct.pack("<II", len(json_payload), 0x4E4F534A) + json_payload
    output += struct.pack("<II", len(binary_payload), 0x004E4942) + binary_payload
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(output)
    print(json.dumps({"status": "ok", "output": str(args.output), "terrain_vertices": len(terrain["vertices"]), "river_vertices": len(river["vertices"])}))


if __name__ == "__main__":
    main()
