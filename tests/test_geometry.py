import numpy as np

from worldclaw_oss.geometry import (
    bottom_contact_ratio,
    cropped_intrinsics,
    gltf_to_z_up_matrix,
    pixel_ray,
    projected_bbox_scale,
    ray_mesh,
    z_up_to_gltf_matrix,
)


def test_coordinate_conversion_round_trip():
    transform = z_up_to_gltf_matrix()
    assert np.allclose(gltf_to_z_up_matrix() @ transform, np.eye(4))
    point = np.array([2.0, 3.0, 4.0, 1.0])
    assert np.allclose(transform @ point, [2.0, 4.0, -3.0, 1.0])


def test_crop_intrinsics_and_center_ray():
    k = np.array([[100.0, 0, 50], [0, 100.0, 40], [0, 0, 1]])
    crop = cropped_intrinsics(k, 10, 10, 80, 60, 160, 120)
    assert np.allclose(crop, [[200, 0, 80], [0, 200, 60], [0, 0, 1]])
    origin, direction = pixel_ray((80, 60), crop, np.eye(4))
    assert np.allclose(origin, [0, 0, 0])
    assert np.allclose(direction, [0, 0, 1])


def test_ray_mesh_and_scale():
    vertices = np.array([[-1, -1, 5], [1, -1, 5], [0, 1, 5]], dtype=float)
    hit = ray_mesh(np.zeros(3), np.array([0, 0, 1.0]), vertices, np.array([[0, 1, 2]]))
    assert hit is not None and abs(hit[0] - 5.0) < 1e-8
    assert projected_bbox_scale((0, 0, 20, 10), (0, 0, 10, 5)) == 2.0


def test_contact_ratio():
    vertices = np.array([[-1, -1, 0], [1, -1, 0.01], [1, 1, 1], [-1, 1, 1]])
    ratio = bottom_contact_ratio(vertices, lambda x, y: 0.0, tolerance=0.02)
    assert ratio == 1.0

