#!/usr/bin/env python3
"""Small Open3D bridge used by the Hunyuan texture worker."""
from __future__ import annotations

import argparse
from pathlib import Path

import open3d as o3d


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-faces", type=int, default=40_000)
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.input)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"empty mesh: {args.input}")
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) > args.target_faces:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=args.target_faces
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh, write_triangle_uvs=False):
        raise RuntimeError(f"failed to write mesh: {output}")


if __name__ == "__main__":
    main()
