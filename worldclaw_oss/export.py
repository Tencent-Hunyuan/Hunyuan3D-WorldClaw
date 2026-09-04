from __future__ import annotations

import json
import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np


COLORS={"terrain":(0.22,0.42,0.18,1.0),"castle":(0.5,0.5,0.48,1.0),"house":(0.48,0.25,0.12,1.0),"tree":(0.08,0.32,0.1,1.0),"rock":(0.3,0.31,0.3,1.0),"building":(0.55,0.43,0.25,1.0),"palm":(0.12,0.4,0.14,1.0),"cabin":(0.4,0.2,0.08,1.0)}


def box_mesh(size=(1.0,1.0,1.0)) -> tuple[np.ndarray,np.ndarray]:
    sx,sy,sz=[v/2 for v in size]
    v=np.array([[x,y,z] for z in (-sz,sz) for y in (-sy,sy) for x in (-sx,sx)],np.float32)
    q=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    t=np.array([(a,b,c) for a,b,c,d in q for a,b,c in ((a,b,c),(a,c,d))],np.uint32)
    return v,t


def vertex_normals(vertices: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return area-weighted, unit vertex normals in the internal Z-up frame."""
    vertices = np.asarray(vertices, dtype=np.float32)
    triangles = np.asarray(triangles, dtype=np.uint32)
    normals = np.zeros_like(vertices, dtype=np.float64)
    tri_vertices = vertices[triangles]
    face_normals = np.cross(tri_vertices[:, 1] - tri_vertices[:, 0], tri_vertices[:, 2] - tri_vertices[:, 0])
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    normals[lengths[:, 0] <= 1e-12] = (0.0, 0.0, 1.0)
    return normals.astype(np.float32)


def orient_terrain_triangles_upward(
    vertices: np.ndarray,
    triangles: np.ndarray,
    min_upward_ratio: float = 0.99,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """Return Terrain triangles whose internal Z-up face side is positive.

    Winding is checked before the internal Z-up to glTF Y-up conversion.  A
    globally reversed mesh is corrected once; mixed or degenerate winding is
    rejected so a viewer's backface setting cannot hide an invalid export.
    """
    values = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.uint32)
    if values.ndim != 2 or values.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Terrain vertices/triangles have invalid shapes")
    if len(faces) == 0:
        raise ValueError("Terrain contains no triangles")
    tri = values[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float64)
    lengths = np.linalg.norm(cross, axis=1)
    valid = lengths > 1e-12
    if not np.any(valid):
        raise ValueError("Terrain triangles are all degenerate")
    unit_z = cross[valid, 2] / lengths[valid]
    mean_normal_z = float(np.mean(unit_z))
    flipped = False
    if mean_normal_z < 0.0:
        faces = faces[:, [0, 2, 1]].copy()
        flipped = True
        tri = values[faces]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]).astype(np.float64)
        lengths = np.linalg.norm(cross, axis=1)
        valid = lengths > 1e-12
        unit_z = cross[valid, 2] / lengths[valid]
        mean_normal_z = float(np.mean(unit_z))
    upward = int(np.count_nonzero(unit_z > 1e-6))
    downward = int(np.count_nonzero(unit_z < -1e-6))
    oriented = upward + downward
    upward_ratio = float(upward / oriented) if oriented else 0.0
    stats: dict[str, float | int | bool] = {
        "mean_normal_z": mean_normal_z,
        "upward_face_ratio": upward_ratio,
        "upward_faces": upward,
        "downward_faces": downward,
        "degenerate_faces": int(len(faces) - int(np.count_nonzero(valid))),
        "flipped": flipped,
    }
    if mean_normal_z <= 0.0 or upward_ratio <= float(min_upward_ratio):
        raise ValueError(
            "Terrain winding validation failed: "
            f"mean_normal_z={mean_normal_z:.6f}, upward_face_ratio={upward_ratio:.6f}"
        )
    return faces, stats


def write_glb(path: Path,meshes: list[dict]):
    materials=[]; mat_lookup={}; mesh_lookup={}
    blob=bytearray(); views=[]; accessors=[]; gltf_meshes=[]; nodes=[]
    def align():
        while len(blob)%4: blob.append(0)
    for item in meshes:
        name=item["name"]; category=item.get("category","building")
        if category not in mat_lookup:
            mat_lookup[category]=len(materials); materials.append({"name":category,"pbrMetallicRoughness":{"baseColorFactor":list(COLORS.get(category,COLORS["building"])),"roughnessFactor":0.85,"metallicFactor":0.0}})
        vertices=np.asarray(item["vertices"],np.float32)
        triangles=np.asarray(item["triangles"],"<u4")
        if category == "terrain":
            triangles, _ = orient_terrain_triangles_upward(vertices, triangles)
        # Reuse a mesh whenever geometry and material are identical.  This is
        # the glTF equivalent of a GPU instance: every node keeps its own
        # transform while vertex/index buffers are emitted once.
        digest = hashlib.sha256()
        digest.update(np.ascontiguousarray(vertices).tobytes())
        digest.update(np.ascontiguousarray(triangles).tobytes())
        digest.update(category.encode("utf-8"))
        mesh_key = digest.hexdigest()
        mesh_index = mesh_lookup.get(mesh_key)
        reused = mesh_index is not None
        if mesh_index is None:
            # Encode internal Z-up as glTF Y-up.
            gv=np.column_stack((vertices[:,0],vertices[:,2],-vertices[:,1])).astype("<f4")
            normals=vertex_normals(vertices, triangles)
            # Normals use the same proper rotation as positions (internal Z-up
            # to glTF Y-up), preserving the authored upward direction.
            gn=np.column_stack((normals[:,0],normals[:,2],-normals[:,1])).astype("<f4")
            po=len(blob); raw=gv.tobytes(); blob.extend(raw); align(); pv=len(views); views.append({"buffer":0,"byteOffset":po,"byteLength":len(raw),"target":34962})
            pa=len(accessors); accessors.append({"bufferView":pv,"componentType":5126,"count":len(gv),"type":"VEC3","min":gv.min(0).tolist(),"max":gv.max(0).tolist()})
            no=len(blob); raw=gn.tobytes(); blob.extend(raw); align(); nv=len(views); views.append({"buffer":0,"byteOffset":no,"byteLength":len(raw),"target":34962})
            na=len(accessors); accessors.append({"bufferView":nv,"componentType":5126,"count":len(gn),"type":"VEC3","min":gn.min(0).tolist(),"max":gn.max(0).tolist()})
            io=len(blob); raw=triangles.tobytes(); blob.extend(raw); align(); iv=len(views); views.append({"buffer":0,"byteOffset":io,"byteLength":len(raw),"target":34963})
            ia=len(accessors); accessors.append({"bufferView":iv,"componentType":5125,"count":int(triangles.size),"type":"SCALAR","min":[int(triangles.min())],"max":[int(triangles.max())]})
            mesh_index = len(gltf_meshes)
            gltf_meshes.append({"name":name,"primitives":[{"attributes":{"POSITION":pa,"NORMAL":na},"indices":ia,"material":mat_lookup[category]}]})
            mesh_lookup[mesh_key] = mesh_index
        extras = dict(item.get("extras",{}))
        if reused:
            extras.setdefault("mesh_instance", True)
        node = {"name":name,"mesh":mesh_index,"extras":extras}
        transform = item.get("transform_z_up", item.get("transform"))
        if transform is not None:
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(f"{name}: transform must be a 4x4 matrix")
            # Internal Z-up -> glTF Y-up basis: (x,y,z) -> (x,z,-y).
            basis = np.asarray(((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0)))
            converted = np.eye(4)
            converted[:3, :3] = basis @ matrix[:3, :3] @ basis.T
            converted[:3, 3] = basis @ matrix[:3, 3]
            # glTF stores matrices column-major in JSON.
            node["matrix"] = converted.T.reshape(-1).tolist()
        nodes.append(node)
    doc={
        "asset": {"version": "2.0", "generator": "worldclaw_oss"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": gltf_meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": views,
        "buffers": [{"byteLength": len(blob)}],
        "extras": {
            "node_count": len(nodes),
            "unique_mesh_count": len(gltf_meshes),
            "instanced_node_count": sum(1 for node in nodes if node.get("extras", {}).get("mesh_instance")),
        },
    }
    js=json.dumps(doc,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode(); js+=b" "*((4-len(js)%4)%4)
    binary=bytes(blob); binary+=b"\0"*((4-len(binary)%4)%4)
    body=struct.pack("<I4s",len(js),b"JSON")+js+struct.pack("<I4s",len(binary),b"BIN\0")+binary
    path.write_bytes(struct.pack("<4sII",b"glTF",2,12+len(body))+body)


def write_png(path: Path,rgb: np.ndarray):
    rgb=np.asarray(rgb,dtype=np.uint8); h,w,_=rgb.shape
    def chunk(tag,data): return struct.pack(">I",len(data))+tag+data+struct.pack(">I",zlib.crc32(tag+data)&0xffffffff)
    raw=b"".join(b"\0"+rgb[y].tobytes() for y in range(h))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw,9))+chunk(b"IEND",b""))


def layout_preview(labels: np.ndarray,path: Path):
    palette=np.array([[63,110,64],[190,154,83],[72,120,162],[140,105,85],[100,120,100]],np.uint8)
    write_png(path,palette[labels%len(palette)])
