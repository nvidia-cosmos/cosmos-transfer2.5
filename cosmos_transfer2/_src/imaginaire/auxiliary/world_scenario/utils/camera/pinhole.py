# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
File originally from https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-drive-dreams-toolkits
"""

from typing import Optional

import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.base import CameraBase
from cosmos_transfer2._src.imaginaire.modules.camera import Camera


class PinholeCamera(CameraBase):
    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        w: int | float,
        h: int | float,
        dtype: torch.dtype = torch.float32,
        device: Optional[str | torch.device] = None,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.w = int(w)
        self.h = int(h)
        self.dtype = dtype

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self._intrinsics = np.array([fx, fy, cx, cy, w, h], dtype=np.float32)
        self.cache_torch_and_np_intrinsics()

    def cache_torch_and_np_intrinsics(self) -> None:
        # cache intrinsics matrix
        self.intrinsics_matrix_torch = self.get_intrinsics_matrix()
        self.intrinsics_matrix_inv_torch = self.get_inv_intrinsics_matrix()
        self.intrinsics_matrix_np = self.intrinsics_matrix_torch.cpu().numpy()
        self.intrinsics_matrix_inv_np = self.intrinsics_matrix_inv_torch.cpu().numpy()

    @staticmethod
    def from_tensor(x: torch.Tensor) -> "PinholeCamera":
        return PinholeCamera(
            x[0].item(), x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), device=x.device
        )

    @staticmethod
    def from_numpy(x: np.ndarray, device: Optional[str] = None) -> "PinholeCamera":
        return PinholeCamera(
            x[0].item(), x[1].item(), x[2].item(), x[3].item(), x[4].item(), x[5].item(), device=device
        )

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PinholeCamera":
        if device is None:
            target_device: str | torch.device = self.device
        else:
            target_device = device
        if isinstance(target_device, torch.device):
            device_str = target_device.type
            if target_device.index is not None:
                device_str = f"{device_str}:{target_device.index}"
        else:
            device_str = str(target_device)

        target_dtype = dtype if dtype is not None else self.dtype

        if device_str == self.device and target_dtype == self.dtype:
            return self

        return PinholeCamera(
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            w=self.w,
            h=self.h,
            dtype=target_dtype,
            device=device_str,
        )

    @property
    def width(self) -> int:
        return self.w

    @property
    def height(self) -> int:
        return self.h

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics

    def rescale(self, ratio_h: float, ratio_w: Optional[float] = None) -> None:
        if ratio_w is None:
            ratio_w = ratio_h

        self.w = int(self.w * ratio_w)
        self.h = int(self.h * ratio_h)

        self.fx = self.fx * ratio_w
        self.fy = self.fy * ratio_h
        self.cx = self.cx * ratio_w
        self.cy = self.cy * ratio_h

        self._intrinsics = np.array([self.fx, self.fy, self.cx, self.cy, self.w, self.h], dtype=np.float32)
        self.cache_torch_and_np_intrinsics()

    def get_fovx(self) -> float:
        """
        Returns:
            fovx: float, horizontal field of view in radians
        """
        return 2 * np.arctan(self.w / (2 * self.fx))

    def get_fovy(self) -> float:
        """
        Returns:
            fovy: float, vertical field of view in radians
        """
        return 2 * np.arctan(self.h / (2 * self.fy))

    def get_intrinsics(self) -> torch.Tensor:
        return torch.tensor([self.fx, self.fy, self.cx, self.cy, self.w, self.h], device=self.device, dtype=self.dtype)

    def get_intrinsics_matrix(self) -> torch.Tensor:
        # Use centralized helper to build K from params for consistency
        params = torch.tensor([self.fx, self.fy, self.cx, self.cy], device=self.device, dtype=self.dtype)
        K = Camera.intrinsic_params_to_matrices(params)
        assert isinstance(K, torch.Tensor), "Expected torch.Tensor from Camera helper when given torch input"
        return K

    def get_inv_intrinsics_matrix(self) -> torch.Tensor:
        return torch.inverse(self.get_intrinsics_matrix())

    def _get_rays_impl(self) -> torch.Tensor:
        """
        Returns:
            rays: (H, W, 3), normalized camera rays in opencv convention

          z (front)
         /
        o ------> x (right)
        |
        v y (down)

        Returns:
            rays: (H, W, 3), normalized camera rays camera coordinate
        """
        u = torch.arange(self.w, dtype=torch.int32, device=self.device)
        v = torch.arange(self.h, dtype=torch.int32, device=self.device)
        u, v = torch.meshgrid(u, v, indexing="xy")  # must pass indexing='xy'
        uv_coords = torch.stack([u, v], dim=-1)  # shape (H, W, 2)
        uv_coords_pad = torch.cat([uv_coords, torch.ones_like(uv_coords[..., :1])], dim=-1)  # shape (H, W, 3)
        cam_coords_norm = self.intrinsics_matrix_inv_torch @ uv_coords_pad.view(-1, 3).T.float()
        cam_coords_norm = cam_coords_norm.T.view(self.h, self.w, 3)
        rays = cam_coords_norm / cam_coords_norm.norm(dim=-1, keepdim=True)

        return rays

    def ray2pixel_np(self, rays: np.ndarray) -> np.ndarray:
        """
        Args:
            rays: (M, 3), camera rays in camera coordinate (opencv convention)
        Returns:
            pixel_coords: (M, 2), pixel coordinates, not normalized to (0, 1)
        """
        if len(rays.shape) == 1:
            rays = rays.reshape(1, -1)

        rays_norm = rays / rays[:, 2:3]  # normalize the rays, image plane is z=1
        pixel_coords = np.einsum("ij,nj->ni", self.intrinsics_matrix_np, rays_norm)[:, :2]

        return pixel_coords

    def ray2pixel_torch(self, rays: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rays: (N, 3), camera rays in camera coordinate (opencv convention)
        Returns:
            pixel_coords: (N, 2), pixel coordinates, not normalized to (0, 1)
        """
        if len(rays.shape) == 1:
            rays = rays.unsqueeze(0)

        rays_norm = rays / rays[:, 2:3]  # normalize the rays, image plane is z=1
        pixel_coords = torch.einsum("ij,nj->ni", self.intrinsics_matrix_torch, rays_norm)[:, :2]

        return pixel_coords

    def pixel2ray_np(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        Args:
            pixel_coords: (M, 2), pixel coordinates, not normalized to (0, 1)
        Returns:
            rays: (M, 3), camera rays in camera coordinate (opencv convention)
        """
        if len(pixel_coords.shape) == 1:
            pixel_coords = pixel_coords.reshape(1, -1)

        pixel_coords_pad = np.concatenate([pixel_coords, np.ones_like(pixel_coords[..., :1])], axis=-1)  # shape (M, 3)
        rays = np.einsum("ij,nj->ni", self.intrinsics_matrix_np, pixel_coords_pad)[:, :3]

        return rays

    def pixel2ray_torch(self, pixel_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_coords: (N, 2), pixel coordinates, not normalized to (0, 1)
        Returns:
            rays: (N, 3), camera rays in camera coordinate (opencv convention)
        """
        if len(pixel_coords.shape) == 1:
            pixel_coords = pixel_coords.unsqueeze(0)

        pixel_coords_pad = torch.cat([pixel_coords, torch.ones_like(pixel_coords[..., :1])], dim=-1)  # shape (N, 3)
        rays = torch.einsum("ij,nj->ni", self.intrinsics_matrix_torch, pixel_coords_pad)[:, :3]

        return rays
