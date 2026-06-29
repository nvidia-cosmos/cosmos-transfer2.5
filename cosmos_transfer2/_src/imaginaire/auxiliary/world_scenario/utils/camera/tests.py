# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for WorldScenario camera classes."""

import numpy as np
import pytest
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.bbox_utils import build_cuboid_bounding_box
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.pinhole import PinholeCamera

W, H = 1280, 720
FX = FY = (W / 2) / np.tan(np.radians(120) / 2)  # ~120 deg horizontal FOV
CX, CY = W / 2, H / 2

def make_pinhole(device: str = "cpu") -> PinholeCamera:
    return PinholeCamera(fx=FX, fy=FY, cx=CX, cy=CY, w=W, h=H, device=device)

def naive_project_np(cam: PinholeCamera, rays: np.ndarray) -> np.ndarray:
    """Original (unguarded) pinhole projection: K @ (rays / z)."""
    rays_norm = rays / rays[:, 2:3]
    return np.einsum("ij,nj->ni", cam.intrinsics_matrix_np, rays_norm)[:, :2]

def test_in_front_projection_matches_unguarded_formula():
    cam = make_pinhole()
    rng = np.random.RandomState(0)
    rays = rng.randn(256, 3).astype(np.float32)
    rays[:, 2] = np.abs(rays[:, 2]) + 0.5  # strictly in front (z >> near plane)

    np.testing.assert_allclose(cam.ray2pixel_np(rays), naive_project_np(cam, rays), rtol=0, atol=1e-3)

def test_np_and_torch_agree():
    cam = make_pinhole()
    rays = np.array(
        [[1.0, 0.5, 10.0], [-2.0, 1.0, 3.0], [0.3, -0.4, 0.0], [1.0, 0.0, -5.0]],
        dtype=np.float32,
    )
    out_np = cam.ray2pixel_np(rays)
    out_torch = cam.ray2pixel_torch(torch.from_numpy(rays)).cpu().numpy()
    np.testing.assert_allclose(out_np, out_torch, rtol=1e-5, atol=1e-3)

@pytest.mark.parametrize(
    "ray",
    [
        [1.0, 0.0, 0.0],  # exactly on the principal plane (was inf)
        [0.0, 0.0, 0.0],  # degenerate origin
        [1.0, -1.0, -5.0],  # behind the camera
        [2.0, 3.0, -1e-9],  # infinitesimally behind
    ],
)
def test_no_inf_or_nan_at_or_behind_near_plane(ray):
    """Near plane / behind camera must stay finite."""
    cam = make_pinhole()
    arr = np.array([ray], dtype=np.float32)
    assert np.isfinite(cam.ray2pixel_np(arr)).all()
    assert torch.isfinite(cam.ray2pixel_torch(torch.from_numpy(arr))).all()

@pytest.mark.parametrize("z", [-0.5, -5.0, -50.0])
def test_behind_camera_does_not_sign_flip(z):
    """Behind-camera points must NOT wrap to the opposite side."""
    cam = make_pinhole()
    right = cam.ray2pixel_np(np.array([[3.0, 0.0, z]], dtype=np.float32))[0]
    left = cam.ray2pixel_np(np.array([[-3.0, 0.0, z]], dtype=np.float32))[0]
    down = cam.ray2pixel_np(np.array([[0.0, 3.0, z]], dtype=np.float32))[0]

    assert right[0] >= CX, f"right-of-camera point wrapped to the left (u={right[0]})"
    assert left[0] <= CX, f"left-of-camera point wrapped to the right (u={left[0]})"
    assert down[1] >= CY, f"below-camera point wrapped upward (v={down[1]})"

    # Regression guard: the original unguarded formula DID wrap these.
    assert naive_project_np(cam, np.array([[3.0, 0.0, z]], dtype=np.float32))[0, 0] < CX

def _actor_to_world(center_z: float, lateral_x: float = 2.0) -> np.ndarray:
    """Scenario: an actor cuboid crossing the camera plane.
    Car (length along its local x) rotated so its 4.5 m length lies along world +z.
    """
    rot = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.float64)  # obj_x -> world_z
    tfm = np.eye(4)
    tfm[:3, :3] = rot
    tfm[:3, 3] = [lateral_x, 0.0, center_z]
    return tfm

def test_crossing_cuboid_projects_finite_and_on_correct_side():
    cam = make_pinhole()
    lwh = np.array([4.5, 2.0, 1.8])  # length, width, height (m)

    saw_straddle = False
    for center_z in np.arange(4.0, -2.01, -0.5):
        verts = build_cuboid_bounding_box(*lwh, _actor_to_world(center_z))  # camera == world (identity pose)
        depth = verts[:, 2]
        uv = cam.ray2pixel_np(verts.astype(np.float32))

        assert np.isfinite(uv).all(), f"non-finite projection at center_z={center_z}"

        if (depth < 0).any() and (depth > 0).any():  # cuboid straddles the camera plane
            saw_straddle = True
            # The actor is entirely to the right (lateral_x=+2). After the fix no
            # vertex may wrap to the far left; the original code produced u far below 0.
            assert uv[:, 0].min() > -CX, (
                f"straddling cuboid wrapped to the opposite side at center_z={center_z} "
                f"(u_min={uv[:, 0].min():.1f})"
            )

    assert saw_straddle, "test scenario never produced a straddling cuboid"

def test_renderer_no_opposite_side_wrap_for_crossing_actor():
    """End-to-end check through the real TiledMultiCameraRenderer."""
    pytest.importorskip("moderngl")
    if not torch.cuda.is_available():
        pytest.skip("renderer end-to-end test requires CUDA + EGL")

    from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import EgoPose, SceneData
    from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.tiled_multi_camera_renderer import (
        TiledMultiCameraRenderer,
    )

    cam = make_pinhole(device="cuda")
    lwh = np.array([4.5, 2.0, 1.8])
    center_zs = list(np.arange(4.0, -1.01, -0.5))

    scene = SceneData(scene_id="issue107", duration_seconds=len(center_zs) / 30.0, frame_rate=30)
    for i in range(len(center_zs)):
        scene.ego_poses.append(
            EgoPose(timestamp=i, position=np.zeros(3, np.float32), orientation=np.array([0, 0, 0, 1], np.float32))
        )

    name = "camera_front_wide_120fov"
    try:
        renderer = TiledMultiCameraRenderer(
            camera_models={name: cam}, scene_data=scene, use_persistent_vbos=False, multi_sample=1
        )
    except Exception as exc:  # pragma: no cover - environment-dependent GL context creation
        pytest.skip(f"could not create GL context: {exc}")

    pose = np.eye(4, dtype=np.float32)  # camera-to-world identity; world +z = forward
    try:
        for fr, cz in enumerate(center_zs):
            obj_info = {
                "car0": {
                    "object_type": "Car",
                    "object_to_world": _actor_to_world(cz).tolist(),
                    "object_lwh": lwh.tolist(),
                    "object_is_moving": True,
                }
            }
            frame = renderer.render_all_cameras({name: pose}, fr, object_info=obj_info)[name]
            assert np.isfinite(frame).all()

            nonblack = frame.sum(axis=2) > 10
            if nonblack.any():
                # The actor is entirely to the right of the camera; the buggy wrap
                # dumped large fills into the left quarter. Assert that does not happen.
                left_quarter = nonblack[:, : W // 4].sum()
                total = nonblack.sum()
                assert left_quarter / total < 0.2, (
                    f"frame {fr} (center_z={cz}): {left_quarter}/{total} non-black pixels in the "
                    f"left quarter suggests opposite-side wrap"
                )
    finally:
        renderer.cleanup()
