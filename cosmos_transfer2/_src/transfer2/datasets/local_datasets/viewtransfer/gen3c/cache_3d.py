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

from pathlib import Path

import torch
from einops import rearrange

from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import atomic_save_npz
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.gen3c.forward_warp_utils_pytorch import (
    forward_warp,
    reliable_depth_mask_range_batch,
    unproject_points,
)


class Cache3D_Base:
    def __init__(
        self,
        input_image,
        input_depth,
        input_w2c,
        input_intrinsics,
        input_mask=None,
        input_format=None,
        input_points=None,
        weight_dtype=torch.float32,
        is_depth=True,
        device="cuda",
        filter_points_threshold=1.0,
        foreground_masking=False,
    ):
        """
        input_image: Tensor with varying dimensions.
        input_format: List of dimension labels corresponding to input_image's dimensions.
                      E.g., ['B', 'C', 'H', 'W'], ['B', 'F', 'C', 'H', 'W'], etc.
        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold
        self.foreground_masking = foreground_masking
        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        # Map dimension names to their indices in input_image
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape
        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask], dim=format_to_indices.get("C"))

        # B (batch size), F (frame count), N dimensions: no aggregation during warping.
        # Only broadcasting over F to match the target w2c.
        # V: aggregate via concatenation or duster
        B = input_shape[format_to_indices.get("B", 0)] if "B" in format_to_indices else 1  # batch
        F = input_shape[format_to_indices.get("F", 0)] if "F" in format_to_indices else 1  # frame
        N = input_shape[format_to_indices.get("N", 0)] if "N" in format_to_indices else 1  # buffer
        V = input_shape[format_to_indices.get("V", 0)] if "V" in format_to_indices else 1  # view
        H = input_shape[format_to_indices.get("H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get("W", 0)] if "W" in format_to_indices else None

        # Desired dimension order
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]

        # Build permute order based on input_format
        permute_order = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            if idx is not None:
                permute_order.append(idx)
            else:
                # Placeholder for dimensions to be added later
                permute_order.append(None)

        # Remove None values for permute operation
        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        # Insert dimensions of size 1 where necessary
        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has the shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :3], input_image[:, :, :, :, 3:]
            self.input_mask = self.input_mask.to("cpu")
        else:
            self.input_mask = None
            self.input_image = input_image
        self.input_image = self.input_image.to(weight_dtype).to("cpu")

        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W, 3).to("cpu")
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)
            self.input_points = (
                self._compute_input_points(
                    input_depth.reshape(-1, 1, H, W),
                    input_w2c.reshape(-1, 4, 4),
                    input_intrinsics.reshape(-1, 3, 3),
                )
                .to(weight_dtype)
                .reshape(B, F, N, V, H, W, 3)
                .to("cpu")
            )
            self.input_depth = input_depth

        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(
                input_depth, ratio_thresh=self.filter_points_threshold
            ).reshape(B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask.to("cpu")
            else:
                self.input_mask = self.input_mask * depth_mask.to(self.input_mask.device)
        self.boundary_mask = None
        if foreground_masking:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth)
            self.boundary_mask = (~depth_mask).reshape(B, F, N, V, 1, H, W).to("cpu")

    def _compute_input_points(self, input_depth, input_w2c, input_intrinsics):
        input_points = unproject_points(
            input_depth,
            input_w2c,
            input_intrinsics,
            is_depth=self.is_depth,
        )
        return input_points

    def update_cache(self):
        raise NotImplementedError

    def input_frame_count(self) -> int:
        return self.input_image.shape[1]

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        bs, F_target, _, _ = target_w2cs.shape

        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B, f"Batch size of target_w2cs ({bs}) must match batch size of input_image ({B}), {bs} != {B}"

        target_w2cs = (
            target_w2cs.reshape(B, F_target, 1, 4, 4).expand(B, F_target, N, 4, 4).reshape(-1, 4, 4).to(self.device)
        )
        target_intrinsics = (
            target_intrinsics.reshape(B, F_target, 1, 3, 3)
            .expand(B, F_target, N, 3, 3)
            .reshape(-1, 3, 3)
            .to(self.device)
        )

        # Keep large tensors on CPU; move only per-chunk slices to GPU inside the loop
        first_images = rearrange(
            self.input_image[:, start_frame_idx : start_frame_idx + F_target].expand(B, F_target, N, V, C, H, W),
            "B F N V C H W-> (B F N) V C H W",
        )
        first_points = rearrange(
            self.input_points[:, start_frame_idx : start_frame_idx + F_target].expand(B, F_target, N, V, H, W, 3),
            "B F N V H W C-> (B F N) V H W C",
        )
        first_masks = (
            rearrange(
                self.input_mask[:, start_frame_idx : start_frame_idx + F_target].expand(B, F_target, N, V, 1, H, W),
                "B F N V C H W-> (B F N) V C H W",
            )
            if self.input_mask is not None
            else None
        )
        boundary_masks = (
            rearrange(self.boundary_mask.expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W")
            if self.boundary_mask is not None
            else None
        )

        if first_images.shape[1] == 1:
            warp_chunk_size = 2
            rendered_warp_images = []
            rendered_warp_masks = []
            rendered_warp_depth = []

            first_images = first_images.squeeze(1)
            first_points = first_points.squeeze(1)
            first_masks = first_masks.squeeze(1) if first_masks is not None else None
            for i in range(0, first_images.shape[0], warp_chunk_size):
                with torch.no_grad():
                    imgs_chunk = first_images[i : i + warp_chunk_size].to(self.device, non_blocking=True)
                    pts_chunk = first_points[i : i + warp_chunk_size].to(self.device, non_blocking=True)
                    masks_chunk = (
                        first_masks[i : i + warp_chunk_size].to(self.device, non_blocking=True)
                        if first_masks is not None
                        else None
                    )
                    bmask_chunk = (
                        boundary_masks[i : i + warp_chunk_size, 0, 0].to(self.device, non_blocking=True)
                        if boundary_masks is not None
                        else None
                    )
                    (
                        rendered_warp_images_chunk,
                        rendered_warp_masks_chunk,
                        rendered_warp_depth_chunk,
                        _,
                    ) = forward_warp(
                        imgs_chunk,
                        mask1=masks_chunk,
                        depth1=None,
                        transformation1=None,
                        transformation2=target_w2cs[i : i + warp_chunk_size],
                        intrinsic1=target_intrinsics[i : i + warp_chunk_size],
                        intrinsic2=target_intrinsics[i : i + warp_chunk_size],
                        render_depth=render_depth,
                        world_points1=pts_chunk,
                        foreground_masking=self.foreground_masking,
                        boundary_mask=bmask_chunk,
                    )
                    rendered_warp_images.append(rendered_warp_images_chunk.to("cpu"))
                    rendered_warp_masks.append(rendered_warp_masks_chunk.to("cpu"))
                    if render_depth:
                        rendered_warp_depth.append(rendered_warp_depth_chunk.to("cpu"))
                    del imgs_chunk, pts_chunk, masks_chunk, bmask_chunk
                    del rendered_warp_images_chunk, rendered_warp_masks_chunk
                    if render_depth:
                        del rendered_warp_depth_chunk
                    torch.cuda.empty_cache()
            rendered_warp_images = torch.cat(rendered_warp_images, dim=0)
            rendered_warp_masks = torch.cat(rendered_warp_masks, dim=0)
            if render_depth:
                rendered_warp_depth = torch.cat(rendered_warp_depth, dim=0)

        else:
            raise NotImplementedError

        pixels = rearrange(rendered_warp_images, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        masks = rearrange(rendered_warp_masks, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        if render_depth:
            pixels = rearrange(rendered_warp_depth, "(b f n) h w -> b f n h w", b=bs, f=F_target, n=N)
        return pixels.to(self.device), masks.to(self.device)

    def export_point_cloud(self, output_path: str | Path):
        assert self.input_points is not None, "Input points were not precomputed. Cannot export point cloud."
        original_shape = tuple(self.input_points.shape)
        input_points = rearrange(self.input_points.cpu().clone(), "B F N V H W C-> (B F N V) H W C")  # N H W 3
        atomic_save_npz(output_path, points=input_points, shape=original_shape)


class Cache4D(Cache3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        rendered_warp_images, rendered_warp_masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx
        )
        return rendered_warp_images, rendered_warp_masks
