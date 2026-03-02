# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agibot single-view dataset for view-transfer post-training."""

from __future__ import annotations

import argparse
import subprocess
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch
from moge.model.v2 import MoGeModel
from torch.utils.data import Dataset
from torchcodec import FrameBatch
from torchcodec.decoders import VideoDecoder

from cosmos_transfer2._src.imaginaire.lazy_config import instantiate
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import get_view_transfer_video_augmentor
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import (
    atomic_save_npz,
    get_video_fps,
    load_depth_mkv_ffv1_mm_u16,
    load_full_video_frames,
    load_mask_mkv_ffv1,
    save_depth_mkv_ffv1_mm_u16,
    save_mask_mkv_ffv1,
    save_render_mp4_h264,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.camera_parameters import (
    _generate_camera_parameters_for_episode,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.depth_backends import (
    generate_depth,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.gen3c.cache_3d import Cache4D
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import (
    CacheCategory,
    camera_parameters_cache_path,
    depth_cache_path,
    point_cloud_cache_path,
    raw_video_path,
    render_cache_paths,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.sample_index import (
    AnchorDirection,
    PairMode,
    ViewTransferPairSample,
    build_samples,
)
from cosmos_transfer2._src.transfer2.utils.input_handling import detect_aspect_ratio


# Mock URL object for augmentor compatibility
class MockUrlMeta:
    """Mock metadata object for WebDataset compatibility."""

    def __init__(self):
        self.opts = {}


class MockUrl:
    """Mock URL object that augmentors expect from WebDataset."""

    def __init__(self, url: str):
        self._url = url
        self.meta = MockUrlMeta()

    def __str__(self) -> str:
        return self._url

    def __repr__(self) -> str:
        return f"MockUrl({self._url})"


class AgibotViewTransferDataset(Dataset):
    """Agibot dataset with cache-first view-transfer conditioning."""

    def __init__(
        self,
        dataset_dir: str,
        num_frames: int,
        resolution: str = "720",
        is_train: bool = True,
        caption_type: str = "t2w_qwen2p5_7b",
        cache_root: str = ".agibot_cache",
        cache_categories: tuple[str | CacheCategory, ...] | None = None,
        clip_names: tuple[str, ...] = ("head", "hand_left", "hand_right"),
        depth_estimator: str = "agibot",
        pair_mode: str | PairMode = "all_ordered_nonself",
        anchor_clip: str | None = None,
        anchor_direction: str | AnchorDirection | None = "source_to_others",
        explicit_pairs: tuple[tuple[str, str], ...] | None = None,
        urdf_path: str = "",
        usd_path: str = "",
        camera_prims: dict[str, str] | None = None,
        base_frame: str = "base_link",
        decoder_device: str = "cpu",
        moge_device: str = "cuda",
        moge_batch_size: int = 8,
        render_device: str = "cuda",
        lock_timeout_sec: float = 600.0,
        lock_poll_sec: float = 0.25,
        **kwargs: object,
    ) -> None:
        super().__init__()
        del kwargs

        self.dataset_dir = dataset_dir
        self.sequence_length = int(num_frames)
        self.resolution = resolution
        self.is_train = is_train
        self.caption_type = caption_type

        self.cache_root = Path(cache_root).resolve()
        self.cache_categories = {CacheCategory(value) for value in cache_categories} if cache_categories else set()
        self.clip_names = tuple(clip_names)
        self.depth_estimator = depth_estimator
        self.pair_mode = PairMode(pair_mode)
        self.anchor_clip = anchor_clip
        self.anchor_direction = AnchorDirection(anchor_direction) if anchor_direction else None
        self.explicit_pairs = explicit_pairs

        self.urdf_path = urdf_path
        self.usd_path = usd_path
        self.camera_prims = camera_prims
        self.base_frame = base_frame

        self.decoder_device = decoder_device
        self.moge_device = moge_device
        self.moge_batch_size = int(moge_batch_size)
        self.lock_timeout_sec = float(lock_timeout_sec)
        self.lock_poll_sec = float(lock_poll_sec)
        self.render_device = render_device

        log.info(f"Initializing AgibotViewTransferDataset with dataset_dir={dataset_dir}")
        self.samples = build_samples(
            dataset_dir=dataset_dir,
            clip_names=self.clip_names,
            pair_mode=self.pair_mode,
            anchor_clip=self.anchor_clip,
            anchor_direction=self.anchor_direction,
            explicit_pairs=self.explicit_pairs,
        )
        if not self.samples:
            raise RuntimeError(
                "AgibotViewTransferDataset found no samples after scanning dataset root. "
                f"dataset_dir={dataset_dir}, clip_names={self.clip_names}, pair_mode={self.pair_mode}"
            )

        # If moge depth estimation is chosen, load model here once:
        self.moge_model = None
        if self.depth_estimator == "moge":
            self.moge_model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl").to(self.moge_device)

        self.num_failed = 0
        self.num_failed_loads = 0
        self.bad_sample_indices = set()  # Track samples that fail

        # Use proper augmentor pipeline for training quality
        self.source_video_key = "source_video"
        self.target_video_key = "target_video"
        self.inpaint_video_key = "rendered_video"
        self.inpaint_mask_key = "rendered_video_mask"
        self.source_intrinsics_key = "source_intrinsics"
        self.source_w2c_key = "source_w2c"
        self.target_intrinsics_key = "target_intrinsics"
        self.target_w2c_key = "target_w2c"
        augmentor_config = get_view_transfer_video_augmentor(
            source_video_key=self.source_video_key,
            target_video_key=self.target_video_key,
            inpaint_video_key=self.inpaint_video_key,
            resolution=resolution,
        )

        # Instantiate augmentors
        self.augmentor = {k: instantiate(v) for k, v in augmentor_config.items()}

        # Double-check text_transform is not present
        if "text_transform" in self.augmentor:
            raise RuntimeError("text_transform should have been filtered out but is still present!")

        log.info(f"Initialized AgibotViewTransferDataset with {len(self.samples)} videos")
        log.info(f"Dataset dir: {self.dataset_dir}")
        log.info(f"Resolution: {resolution}")
        log.info(f"Required frames: {self.sequence_length}")

    def _should_cache(self, category: CacheCategory) -> bool:
        return category in self.cache_categories

    def __len__(self) -> int:
        return len(self.samples)

    def __str__(self) -> str:
        return f"AgibotViewTransferDataset: {len(self.samples)} samples from {self.dataset_dir}"

    def __getitem__(self, index: int) -> dict[str, Any]:
        max_retries = 1  # Try up to 10 different samples
        original_index = index

        for retry in range(max_retries):
            # Skip known bad videos
            if index in self.bad_sample_indices:
                index = (index + 1) % len(self.samples)
                continue

            sample = self.samples[index]
            log.debug(f"Loading sample {sample} (index {index}, attempt {retry + 1}/{max_retries})")
            try:
                source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
                target_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.target_clip)

                (
                    source_frames_nchw_uint8,
                    target_frames_nchw_uint8,
                    fps,
                    frame_ids,
                ) = self._load_source_target_frames(
                    source_video_path=source_video_path,
                    target_video_path=target_video_path,
                )  # N C H W uint8, N C H W uint8, float, list[int]

                (
                    render_frames_nchw_uint8,
                    render_mask_frames_n1hw_uint8,
                ) = self._get_render_frames(
                    sample=sample,
                    frame_ids=frame_ids,
                    fps=fps,
                )  # N C H W uint8, N 1 H W uint8

                # Permute to (C, T, H, W) format expected by augmentors
                source_video = source_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                target_video = target_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                control = render_frames_nchw_uint8.permute(1, 0, 2, 3)  # C T H W
                mask = render_mask_frames_n1hw_uint8.permute(1, 0, 2, 3)  # 1 T H W

                target_h, target_w = target_video.shape[-2:]
                aspect_ratio = detect_aspect_ratio((target_w, target_h))

                # Build data dictionary
                data = {
                    self.source_video_key: source_video,
                    self.target_video_key: target_video,
                    self.inpaint_video_key: control,
                    self.inpaint_mask_key: mask,
                    "aspect_ratio": aspect_ratio,
                    "fps": fps,
                    "frame_start": frame_ids[0],
                    "frame_end": frame_ids[-1] + 1,
                    "num_frames": self.sequence_length,
                    "chunk_index": 0,
                    "frame_indices": frame_ids,
                    "n_orig_video_frames": len(frame_ids),
                }

                # Add camera parameters
                source_intrinsics_n33, source_w2c_n44 = self._get_camera_parameters(
                    sample=sample, clip_name=sample.source_clip
                )
                target_intrinsics_n33, target_w2c_n44 = self._get_camera_parameters(
                    sample=sample, clip_name=sample.target_clip
                )

                data["source_intrinsics"] = source_intrinsics_n33
                data["source_w2c"] = source_w2c_n44
                data["target_intrinsics"] = target_intrinsics_n33
                data["target_w2c"] = target_w2c_n44

                # Load caption placeholder
                caption = "a robot video"
                data[self.caption_type] = caption

                # Pass raw caption for on-the-fly encoding by model's text encoder
                data["ai_caption"] = caption

                # Add URL and key for logging (used by augmentors and training)
                # Use MockUrl object for augmentor compatibility (augmentors expect __url__.meta.opts)
                data["__url__"] = MockUrl(str(self.dataset_dir))
                data["__key__"] = f"Agibot_{sample}"

                # Apply augmentation pipeline
                # This includes: resizing, padding, and control input generation
                for aug_name, aug_fn in self.augmentor.items():
                    result = aug_fn(data)  # pyright: ignore
                    # Check if augmentor returned None (e.g., filtering)
                    if result is None:
                        raise ValueError(f"Augmentor {aug_name} filtered out the sample")
                    data = result

                # Convert MockUrl back to string for DataLoader collate compatibility
                # (PyTorch's collate function can't handle custom objects)
                if isinstance(data.get("__url__"), MockUrl):
                    data["__url__"] = str(data["__url__"])

                # Add final metadata (after augmentation)
                c, t, h, w = data[self.target_video_key].shape
                if "image_size" not in data:
                    data["image_size"] = torch.tensor([h, w, h, w])
                if "padding_mask" not in data:
                    data["padding_mask"] = torch.ones(1, h, w)  # All valid (no padding)

                # Validate output format after augmentation
                for key in [
                    self.source_video_key,
                    self.target_video_key,
                    self.inpaint_video_key,
                    self.inpaint_mask_key,
                ]:
                    assert key in data, f"Augmentor output missing expected key: {key}"
                    assert isinstance(data[key], torch.Tensor), f"Expected tensor for {key}, got {type(data[key])}"
                    assert data[key].dtype == torch.uint8, f"Expected uint8 dtype for {key}, got {data[key].dtype}"
                    assert data[key].ndim == 4, f"Expected 4D tensor for {key}, got {data[key].ndim}D"
                    assert data[key].shape[1] == self.sequence_length, (
                        f"Expected {self.sequence_length} frames for {key}, got {data[key].shape[1]}"
                    )

                return data

            except Exception as e:
                self.num_failed_loads += 1
                self.bad_sample_indices.add(index)

                tb_str = traceback.format_exc()

                log.warning(
                    f"Failed to load sample {self.samples[index]} (index {index}):\n"
                    f"{tb_str}\n"
                    f"Marking as bad and trying next sample. "
                    f"(attempt {retry + 1}/{max_retries}, "
                    f"total bad samples: {len(self.bad_sample_indices)})",
                    rank0_only=False,
                )

                if retry == max_retries - 1:
                    log.error(
                        f"Failed to load data after {max_retries} attempts starting from index {original_index}. "
                        f"Total bad samples: {len(self.bad_sample_indices)}/{len(self.samples)}"
                    )
                    raise RuntimeError(
                        f"Failed to load data after {max_retries} attempts. "
                        f"Original index: {original_index}, last tried: {self.samples[index]}"
                    )

                # Try the next sample in sequence (wraps around at end)
                index = (index + 1) % len(self.samples)

        raise RuntimeError("Should not reach here")

    def _load_source_target_frames(
        self,
        *,
        source_video_path: Path,
        target_video_path: Path,
    ) -> tuple[torch.Tensor, torch.Tensor, float, list[int]]:
        source_decoder = VideoDecoder(str(source_video_path), device=self.decoder_device)
        target_decoder = VideoDecoder(str(target_video_path), device=self.decoder_device)

        min_frames = min(len(source_decoder), len(target_decoder))
        if min_frames < self.sequence_length:
            raise ValueError(
                f"Not enough frames for sample. source={len(source_decoder)}, target={len(target_decoder)}, "
                f"required={self.sequence_length}"
            )

        if self.is_train:
            start = np.random.randint(0, min_frames - self.sequence_length + 1)
        else:
            start = 0
        frame_ids = list(range(start, start + self.sequence_length))

        source_frames: FrameBatch = source_decoder.get_frames_at(frame_ids)  # N C H W, uint8
        target_frames: FrameBatch = target_decoder.get_frames_at(frame_ids)  # N C H W, uint8

        fps = self._get_fps_from_decoder(target_decoder)
        source_fps = self._get_fps_from_decoder(source_decoder)
        if abs(source_fps - fps) > 1e-3:
            log.warning(
                f"Source/target FPS mismatch for sample: source_fps={source_fps}, target_fps={fps}. Using target FPS."
            )

        return (
            source_frames.data,
            target_frames.data,
            fps,
            frame_ids,
        )

    @staticmethod
    def _depth_m_to_mm_u16(depth_m: np.ndarray) -> np.ndarray:
        depth_m = np.asarray(depth_m)
        valid = np.isfinite(depth_m) & (depth_m > 0)
        depth_mm = np.zeros(depth_m.shape, dtype=np.uint16)
        depth_mm[valid] = np.clip(np.rint(depth_m[valid] * 1000.0), 0, 65535).astype(np.uint16)
        return depth_mm

    def _get_camera_parameters(
        self,
        *,
        sample: ViewTransferPairSample,
        clip_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cam_path = camera_parameters_cache_path(
            cache_root=self.cache_root,
            task=sample.task,
            episode=sample.episode,
            clip_name=clip_name,
        )
        if cam_path.exists():
            with np.load(cam_path) as cam_params:
                intrinsics = torch.from_numpy(cam_params["intrinsics"]).float()
                w2c = torch.from_numpy(cam_params["w2c"]).float()
            return intrinsics, w2c

        log.debug(f"Computing {cam_path}...")
        params_by_clip = _generate_camera_parameters_for_episode(
            dataset_dir=self.dataset_dir,
            task=sample.task,
            episode=sample.episode,
            clip_names=self.clip_names,
            urdf_path=self.urdf_path,
            usd_path=self.usd_path,
            camera_prims=self.camera_prims,
            base_frame=self.base_frame,
        )
        if clip_name not in params_by_clip:
            raise ValueError(
                f"Camera parameters for clip {clip_name!r} not found. Available clips: {tuple(params_by_clip.keys())}"
            )

        if self._should_cache(CacheCategory.CAMERA_PARAMETERS):
            log.debug(f"Caching {cam_path}...")
            for cache_clip_name, params in params_by_clip.items():
                cache_path = camera_parameters_cache_path(
                    cache_root=self.cache_root,
                    task=sample.task,
                    episode=sample.episode,
                    clip_name=cache_clip_name,
                )
                atomic_save_npz(cache_path, intrinsics=params["intrinsics"], w2c=params["w2c"])

        params = params_by_clip[clip_name]
        intrinsics = torch.from_numpy(params["intrinsics"]).float()
        w2c = torch.from_numpy(params["w2c"]).float()
        return intrinsics, w2c

    def _get_depth(
        self,
        *,
        sample: ViewTransferPairSample,
        source_video_path: Path,
        frame_count: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        d_path = depth_cache_path(
            cache_root=self.cache_root,
            task=sample.task,
            episode=sample.episode,
            source_clip=sample.source_clip,
            depth_estimator=self.depth_estimator,
        )
        if d_path.exists():
            depth_nhw_m_float64_np = load_depth_mkv_ffv1_mm_u16(
                path=d_path,
                width=width,
                height=height,
                return_float64=True,
            )
        else:
            log.debug(f"Computing {d_path}...")
            depth_nhw_m_float64_np = generate_depth(
                dataset_dir=self.dataset_dir,
                task=sample.task,
                episode=sample.episode,
                source_clip=sample.source_clip,
                depth_estimator=self.depth_estimator,
                moge_model=self.moge_model,
                moge_device=self.moge_device,
                moge_batch_size=self.moge_batch_size,
            )
            if depth_nhw_m_float64_np.ndim != 3:
                raise ValueError(f"Expected depth with shape [F,H,W], got {depth_nhw_m_float64_np.shape}")
            if tuple(depth_nhw_m_float64_np.shape[1:]) != (height, width):
                raise ValueError(
                    f"Depth resolution mismatch. depth={depth_nhw_m_float64_np.shape[1:]}, expected={(height, width)}"
                )
            if self._should_cache(CacheCategory.DEPTH):
                log.debug(f"Caching {d_path}...")
                depth_mm_u16 = self._depth_m_to_mm_u16(depth_nhw_m_float64_np)
                source_fps = get_video_fps(source_video_path)
                save_depth_mkv_ffv1_mm_u16(depth_mm=depth_mm_u16, path=d_path, fps=source_fps)

        if depth_nhw_m_float64_np.shape[0] != frame_count:
            raise ValueError(
                f"Depth frame count mismatch. depth={depth_nhw_m_float64_np.shape[0]}, source={frame_count}"
            )
        return torch.from_numpy(depth_nhw_m_float64_np).unsqueeze(1)  # [F, 1, H, W], float64

    def _get_point_cloud(self, *, sample: ViewTransferPairSample) -> Cache4D:
        source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
        input_frames_nchw_uint8 = load_full_video_frames(source_video_path, decoder_device=self.decoder_device)
        image_tensor = (input_frames_nchw_uint8.float() / 127.5) - 1.0  # [0,255] -> [-1,1]

        pcd_path = point_cloud_cache_path(
            self.cache_root,
            sample.task,
            sample.episode,
            sample.source_clip,
            sample.target_clip,
            self.depth_estimator,
        )
        if pcd_path.exists():
            points = torch.from_numpy(np.load(pcd_path)["points"])  # F, H, W, 3
            return Cache4D(
                input_image=image_tensor,  # [F, C, H, W]
                input_depth=None,
                input_w2c=None,
                input_intrinsics=None,
                input_format=["F", "C", "H", "W"],
                input_points=points,  # [F, H, W, 3]
                device=self.render_device,
            )

        log.debug(f"Computing {pcd_path}...")
        _, _, height, width = image_tensor.shape
        depth_n1hw_m_float64 = self._get_depth(
            sample=sample,
            source_video_path=source_video_path,
            frame_count=image_tensor.shape[0],
            height=height,
            width=width,
        )
        intrinsics_n33, initial_w2c_n44 = self._get_camera_parameters(sample=sample, clip_name=sample.source_clip)

        cache_4d = Cache4D(
            input_image=image_tensor,  # [F, C, H, W]
            input_depth=depth_n1hw_m_float64,  # [F, 1, H, W]
            input_w2c=initial_w2c_n44,  # [F, 4, 4]
            input_intrinsics=intrinsics_n33,  # [F, 3, 3]
            input_format=["F", "C", "H", "W"],
            device=self.render_device,
        )

        if self._should_cache(CacheCategory.POINT_CLOUD):
            log.debug(f"Caching {pcd_path}...")
            cache_4d.export_point_cloud(pcd_path)

        return cache_4d

    def _get_render_frames(
        self,
        *,
        sample: ViewTransferPairSample,
        frame_ids: list[int],
        fps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        render_path, mask_path = render_cache_paths(
            self.cache_root,
            sample.task,
            sample.episode,
            sample.source_clip,
            sample.target_clip,
            self.depth_estimator,
        )
        if render_path.exists() and mask_path.exists():
            render_decoder = VideoDecoder(str(render_path), device=self.decoder_device)
            render_fps = self._get_fps_from_decoder(render_decoder)
            if abs(render_fps - fps) > 1e-3:
                log.warning(
                    f"Render/target FPS mismatch for sample: render_fps={render_fps}, target_fps={fps}. "
                    "Using target FPS."
                )
            width, height = self._get_shape_from_decoder(render_decoder)
            mask_frames = load_mask_mkv_ffv1(path=mask_path, width=width, height=height)  # T H W, uint8
            if len(render_decoder) != len(mask_frames):
                raise ValueError(f"Render and mask lengths mismatch: {len(render_decoder)} vs {len(mask_frames)}")
            if frame_ids[-1] >= len(render_decoder):
                raise ValueError(
                    f"Render cache shorter than requested frame ids. "
                    f"max_requested={frame_ids[-1]}, render={len(render_decoder)}, mask={len(mask_frames)}"
                )
            render_frames: FrameBatch = render_decoder.get_frames_at(frame_ids)  # N C H W, uint8
            rendered_masks_n1hw_uint8 = torch.from_numpy(mask_frames[frame_ids]).unsqueeze(1)  # N 1 H W, uint8
            rendered_images_nchw_uint8 = render_frames.data

        else:
            log.debug(f"Computing {render_path} and {mask_path}...")
            cache_4d = self._get_point_cloud(sample=sample)
            target_intrinsics_n33, target_w2c_n44 = self._get_camera_parameters(
                sample=sample, clip_name=sample.target_clip
            )
            rendered_warp_images, rendered_warp_masks = cache_4d.render_cache(
                target_w2cs=target_w2c_n44.unsqueeze(0),  # [1, F, 4, 4]
                target_intrinsics=target_intrinsics_n33.unsqueeze(0),  # [1, F, 3, 3]
            )  # [B, F, N, C, H, W]

            rendered_images_nchw_uint8 = (
                ((rendered_warp_images.squeeze(0).squeeze(1).cpu() + 1.0) * 127.5).clamp(0, 255).byte()
            )
            rendered_masks_n1hw_uint8 = (rendered_warp_masks.squeeze(0).squeeze(1).cpu() > 0).byte() * 255
            if frame_ids[-1] >= rendered_images_nchw_uint8.shape[0]:
                raise ValueError(
                    f"On-the-fly render shorter than requested frame ids. "
                    f"max_requested={frame_ids[-1]}, rendered={rendered_images_nchw_uint8.shape[0]}"
                )

            if self._should_cache(CacheCategory.RENDER):
                log.debug(f"Caching {render_path} and {mask_path}...")
                source_video_path = raw_video_path(self.dataset_dir, sample.task, sample.episode, sample.source_clip)
                source_fps = get_video_fps(source_video_path)
                save_render_mp4_h264(
                    frames=rendered_images_nchw_uint8.permute(0, 2, 3, 1).numpy(),
                    path=render_path,
                    fps=source_fps,
                )
                save_mask_mkv_ffv1(masks=rendered_masks_n1hw_uint8.squeeze(1).numpy(), path=mask_path, fps=source_fps)

            rendered_images_nchw_uint8 = rendered_images_nchw_uint8[frame_ids]
            rendered_masks_n1hw_uint8 = rendered_masks_n1hw_uint8[frame_ids]

        return rendered_images_nchw_uint8, rendered_masks_n1hw_uint8

    def _get_fps_from_decoder(self, decoder: VideoDecoder) -> float:
        fps = decoder.metadata.average_fps
        assert fps is not None, "Decord returned None for average_fps"
        fps = float(fps)
        if fps <= 0:
            raise RuntimeError("Failed to obtain a valid FPS")
        return fps

    def _get_shape_from_decoder(self, decoder: VideoDecoder) -> tuple[int, int]:
        width = decoder.metadata.width
        height = decoder.metadata.height
        if width is None or height is None:
            raise RuntimeError("Failed to obtain valid video dimensions from decoder metadata")
        return width, height


def _summarize_sample(sample: dict[str, Any]) -> None:
    print("Sample keys:")
    for key in sorted(sample.keys()):
        value = sample[key]
        if isinstance(value, torch.Tensor):
            print(f"  {key}: Tensor shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, (str, int, float, bool)):
            print(f"  {key}: {value!r}")
        elif isinstance(value, (list, tuple)):
            if len(value) <= 10:
                print(f"  {key}: {type(value).__name__} len={len(value)} value={value}")
            else:
                print(f"  {key}: {type(value).__name__} len={len(value)}")
        else:
            print(f"  {key}: {type(value).__name__}")


def visualize_sample_dict(sample: dict[str, Any], *, output_dir: str | Path, sample_name: str = "sample") -> Path:
    """
    Save tensor videos from a sample and create a side-by-side mp4 using ffmpeg.

    Returns:
        Path to the side-by-side mp4.
    """
    out_dir = Path(output_dir)
    sample_dir = out_dir / sample_name
    sample_dir.mkdir(parents=True, exist_ok=True)

    fps = sample["fps"]

    video_keys = ["source_video", "target_video", "rendered_video", "rendered_video_mask"]

    video_paths: list[Path] = []
    for key in video_keys:
        frames_thwc = sample[key].cpu().permute(1, 2, 3, 0).numpy()  # T H W C
        path = sample_dir / f"{key}.mp4"
        path.unlink(missing_ok=True)
        save_render_mp4_h264(path=path, frames=frames_thwc, fps=fps)
        video_paths.append(path)

    if len(video_paths) == 1:
        return video_paths[0]

    side_by_side_path = sample_dir / "side_by_side.mp4"
    side_by_side_path.unlink(missing_ok=True)
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for path in video_paths:
        cmd += ["-i", str(path)]
    filter_graph = "".join([f"[{i}:v]" for i in range(len(video_paths))]) + f"hstack=inputs={len(video_paths)}[v]"
    cmd += ["-filter_complex", filter_graph, "-map", "[v]", "-an", str(side_by_side_path)]

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required for visualization but was not found in PATH.") from exc

    return side_by_side_path


def _parse_args():
    parser = argparse.ArgumentParser(description="Debug AgibotViewTransferDataset by loading one sample.")
    parser.add_argument("--depth-estimator", type=str, default="agibot", choices=["agibot", "moge"])
    parser.add_argument("--moge-device", type=str, default="cuda")
    parser.add_argument("--moge-batch-size", type=int, default=8)
    parser.add_argument("--is-train", action="store_true", help="Use training sampling (random start frame).")
    parser.add_argument("--output-dir", type=str, default="output/debug_viewtransfer")
    parser.add_argument("--skip-visualize", action="store_true", help="Only print sample summary.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    dataset = AgibotViewTransferDataset(
        dataset_dir="/mnt/central_storage/data_pool_raw/AgiBotWorld-Alpha",
        cache_root="/raid/andrew.mitri/agibot_cache",
        cache_categories=("camera", "render"),
        num_frames=93,
        is_train=args.is_train,
        depth_estimator=args.depth_estimator,
        pair_mode=PairMode.ONE_DIRECTION_FROM_ANCHOR,
        anchor_clip="head",
        anchor_direction=AnchorDirection.SOURCE_TO_OTHERS,
        urdf_path="/raid/andrew.mitri/geniesim_assets/G1_120s/G1_120s.urdf",
        usd_path="/raid/andrew.mitri/geniesim_assets/G1_120s/G1_120s.usda",
        moge_device=args.moge_device,
        moge_batch_size=args.moge_batch_size,
        camera_prims={
            "head": "/G1/head_link2/Head_Camera",
            "hand_right": "/G1/gripper_r_base_link/Right_Camera",
            "hand_left": "/G1/gripper_l_base_link/Left_Camera",
        },
    )

    print(dataset)
    samples = [dataset[idx] for idx in range(20)]
    _summarize_sample(samples[0])

    if not args.skip_visualize:
        for idx, sample in enumerate(samples):
            out_path = visualize_sample_dict(
                sample,
                output_dir=args.output_dir,
                sample_name=f"sample_{idx}",
            )
            print(f"Visualization written to: {out_path}")


if __name__ == "__main__":
    main()
