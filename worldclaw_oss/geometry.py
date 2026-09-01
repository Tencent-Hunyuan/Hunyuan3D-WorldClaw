from __future__ import annotations

import math

import numpy as np


def gltf_vertices_to_z_up(vertices: np.ndarray) -> np.ndarray:
    """Convert glTF Y-up vertex coordinates to the pipeline's internal Z-up.

    glTF uses (X, Y-up, Z), while placement and terrain logic use a right-handed
    (X, Y, Z-up) frame.  The conversion preserves handedness: (x, y, z) maps
    to (x, -z, y).
    """
    values = np.asarray(vertices, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    return values[:, (0, 2, 1)] * np.asarray([1.0, -1.0, 1.0])


def z_up_to_gltf_matrix() -> np.ndarray:
    # Internal (x,y,z) -> glTF (x,z,-y).
    return np.array([[1,0,0,0],[0,0,1,0],[0,-1,0,0],[0,0,0,1]], dtype=float)


def gltf_to_z_up_matrix() -> np.ndarray:
    return np.linalg.inv(z_up_to_gltf_matrix())


def cropped_intrinsics(k: np.ndarray, x0: float, y0: float, crop_w: float, crop_h: float, out_w: float, out_h: float) -> np.ndarray:
    result = np.asarray(k, dtype=float).copy()
    sx, sy = out_w / crop_w, out_h / crop_h
    result[0, 0] *= sx; result[1, 1] *= sy
    result[0, 2] = (result[0, 2] - x0) * sx
    result[1, 2] = (result[1, 2] - y0) * sy
    return result


def pixel_ray(pixel: tuple[float,float], k: np.ndarray, camera_to_world: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    uv1 = np.array([pixel[0], pixel[1], 1.0])
    direction_camera = np.linalg.inv(k) @ uv1
    direction_world = camera_to_world[:3,:3] @ direction_camera
    direction_world /= np.linalg.norm(direction_world)
    return camera_to_world[:3,3].copy(), direction_world


def ray_triangle(origin: np.ndarray, direction: np.ndarray, triangle: np.ndarray, epsilon: float = 1e-8) -> float | None:
    v0,v1,v2 = np.asarray(triangle, dtype=float)
    e1,e2 = v1-v0,v2-v0
    h = np.cross(direction,e2); a = float(e1 @ h)
    if -epsilon < a < epsilon: return None
    f = 1.0/a; s=origin-v0; u=f*float(s@h)
    if u < 0 or u > 1: return None
    q=np.cross(s,e1); v=f*float(direction@q)
    if v < 0 or u+v > 1: return None
    t=f*float(e2@q)
    return t if t > epsilon else None


def ray_mesh(origin: np.ndarray, direction: np.ndarray, vertices: np.ndarray, triangles: np.ndarray) -> tuple[float,np.ndarray] | None:
    best = math.inf
    for indices in triangles:
        hit = ray_triangle(origin, direction, vertices[indices])
        if hit is not None and hit < best: best=hit
    return None if math.isinf(best) else (best, origin + best*direction)


def projected_bbox_scale(target_bbox: tuple[float,float,float,float], current_bbox: tuple[float,float,float,float]) -> float:
    tw=max(target_bbox[2]-target_bbox[0],1e-9); th=max(target_bbox[3]-target_bbox[1],1e-9)
    cw=max(current_bbox[2]-current_bbox[0],1e-9); ch=max(current_bbox[3]-current_bbox[1],1e-9)
    return math.sqrt((tw/cw)*(th/ch))


def bottom_contact_ratio(vertices_world: np.ndarray, terrain_height, tolerance: float, bottom_fraction: float = 0.1) -> float:
    vertices=np.asarray(vertices_world,float); zmin,zmax=float(vertices[:,2].min()),float(vertices[:,2].max())
    cutoff=zmin+max((zmax-zmin)*bottom_fraction,tolerance)
    bottom=vertices[vertices[:,2] <= cutoff]
    if len(bottom)==0: return 0.0
    errors=np.array([abs(v[2]-terrain_height(v[0],v[1])) for v in bottom])
    return float(np.mean(errors <= tolerance))


def search_depth_scale(origin: np.ndarray, direction: np.ndarray, local_vertices: np.ndarray, terrain_height, depths, scales, tolerance: float) -> tuple[float,float,float]:
    best=(-1.0,0.0,1.0)
    for depth in depths:
        anchor=origin+float(depth)*direction
        for scale in scales:
            world=local_vertices*float(scale)+anchor
            ratio=bottom_contact_ratio(world,terrain_height,tolerance)
            if ratio > best[0]: best=(ratio,float(depth),float(scale))
    return best[1],best[2],best[0]
