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

"""Plenoptic Multiview Camera inference module."""

import os
from pathlib import Path
from typing import Any

import imageio
import numpy as np
import torch
import torchvision
from einops import rearrange
from loguru import logger
from PIL import Image
from torchvision import io, transforms

from cosmos_transfer2._src.imaginaire.modules.camera import Camera
from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

# pyrefly: ignore[missing-import]
from cosmos_transfer2._src.predict2.camera.inference.multiview_camera_ar_video2world import Video2WorldInference
from cosmos_transfer2.config import MODEL_CHECKPOINTS, load_callable
from cosmos_transfer2.plenoptic_config import (
    CAMERA_MOTION_TYPES,
    CameraLoadFn,
    PlenopticInferenceArguments,
    PlenopticSetupArguments,
)


def _finalize_camera_metadata(
    data: dict[str, Any],
    extrinsics: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    height: int,
    width: int,
) -> dict[str, Any]:
    """Normalize camera tensors for downstream consumption."""
    extrinsics_flat = Camera.invert_pose(extrinsics[:, :3, :]).reshape(-1, 3, 4)  # pyrefly: ignore[missing-attribute]
    intrinsics_flat = intrinsics.reshape(-1, 4)
    image_size = torch.tensor([height, width, height, width], device=extrinsics_flat.device)

    data["extrinsics"] = extrinsics_flat
    data["intrinsics"] = intrinsics_flat
    data["image_size"] = image_size
    return data


def _load_single_camera_extrinsics(
    cam_type: str,
    base_path: str,
    latent_frames: int,
    extrinsic_scale: float,
    scale_fn,  # scale_rotation_angle_and_translation function
) -> torch.Tensor:
    """Load camera extrinsics from .pt or .txt file with scaling."""
    extrinsics_path = os.path.join(base_path, "cameras", cam_type + ".pt")
    if os.path.exists(extrinsics_path):
        extrinsics = torch.load(extrinsics_path).to(torch.float32)
    else:
        # Fallback to txt format - apply extrinsic scaling as in compute_camera_similarity.py
        txt_path = os.path.join(base_path, "cameras", cam_type + ".txt")
        extrinsics_tgt = torch.tensor(np.loadtxt(txt_path), dtype=torch.float32)
        extrinsics_tgt = extrinsics_tgt[:latent_frames]
        extrinsics = torch.cat(
            (
                extrinsics_tgt,
                torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float32).unsqueeze(0).expand(latent_frames, -1),
            ),
            dim=1,
        ).reshape(-1, 4, 4)
        # Apply extrinsic scale to rotation and translation
        extrinsics = scale_fn(extrinsics, angle_factor=extrinsic_scale, trans_factor=extrinsic_scale)
    return extrinsics


def _remove_indices(lst: list, indices: list[int]) -> list:
    """Remove elements at specified indices from list."""
    lst = lst.copy()
    for idx in sorted(indices, reverse=True):
        if 0 <= idx < len(lst):
            lst.pop(idx)
    return lst


def _generate_camera_combinations_with_memory_retrieval(
    camera_list: list[str],
    cam_data_dict: dict[str, torch.Tensor],
    hfov_deg: float,
    vfov_deg: float,
    max_retrieval_range: int = 6,
    retrieval_num: int = 4,
    compute_similarity_fn=None,
    combine_sequences_fn=None,
) -> list[list[str]]:
    """Generate camera combinations using memory retrieval algorithm.

    This implements the same algorithm as compute_camera_similarity.py,
    iterating progressively through the camera list (like the original bash script).
    For each camera index i from 1 to len-1, it generates ONE combination:
    - conditioning cameras (up to retrieval_num) + target camera (camera_list[i])

    Args:
        camera_list: List of camera names in sequence (first is typically "static")
        cam_data_dict: Dictionary mapping camera names to extrinsics tensors
        hfov_deg: Horizontal field of view in degrees
        vfov_deg: Vertical field of view in degrees
        max_retrieval_range: Maximum number of cameras to consider for retrieval
        retrieval_num: Number of conditioning cameras to select (typically 4)
        compute_similarity_fn: Function to compute similarity matrix
        combine_sequences_fn: Function to combine camera sequences

    Returns:
        List of camera combinations, each combination has retrieval_num+1 cameras
        (retrieval_num conditioning + 1 target)
    """
    output_combinations = []
    merge_idx = 0

    # Iterate progressively through the camera list, generating one combination per target camera
    # This matches the original bash script behavior: for ((i=1; i<${#cam_list[@]}; i++))
    for i in range(1, len(camera_list)):
        # Current sub-list of cameras up to and including the target
        current_cameras = camera_list[: i + 1]
        target_camera = current_cameras[-1]

        if len(current_cameras) < max_retrieval_range:
            # Simple case: pad with static cameras
            # We need exactly retrieval_num + 1 cameras (4 conditioning + 1 target)
            total_needed = retrieval_num + 1  # 5 cameras
            if len(current_cameras) < total_needed:
                # Pad with static to reach total_needed
                num_static_needed = total_needed - len(current_cameras)
                combination = ["static"] * num_static_needed + list(current_cameras)
            else:
                # Use the last total_needed cameras
                combination = list(current_cameras[-total_needed:])
            output_combinations.append(combination)
        else:
            # Complex case: use similarity-based selection
            cam_type_list = list(current_cameras)
            pose_list = [cam_data_dict[name] for name in cam_type_list]

            # Compute similarity and prune to max_retrieval_range cameras
            # pyrefly: ignore[not-callable]
            sim = compute_similarity_fn(pose_list, hfov_deg=hfov_deg, vfov_deg=vfov_deg, match="index")
            mask = torch.tril(torch.ones_like(sim))
            sim = sim * mask

            if len(cam_type_list) >= (max_retrieval_range + 1):
                # Keep only max_retrieval_range most similar cameras to target
                rm_indices = torch.argsort(sim[-1][:-1])[:-max_retrieval_range].tolist()
                pose_list = _remove_indices(list(pose_list), rm_indices)
                cam_type_list = _remove_indices(list(cam_type_list), rm_indices)

            # Now iteratively merge until we have retrieval_num + 1 cameras
            while len(cam_type_list) > retrieval_num + 1:
                # pyrefly: ignore[not-callable]
                sim = compute_similarity_fn(pose_list, hfov_deg=hfov_deg, vfov_deg=vfov_deg, match="index")
                mask = torch.tril(torch.ones_like(sim))
                sim = sim * mask

                # pyrefly: ignore[bad-argument-type]
                len_indices = min(len(cam_type_list) - retrieval_num, retrieval_num)
                min_indices = torch.argsort(sim[-1][:-1])[:len_indices].tolist()

                # Merge the least similar cameras
                merged_pose = combine_sequences_fn([pose_list[idx] for idx in min_indices])  # pyrefly: ignore
                merge_name = f"merge_{merge_idx}"
                cam_data_dict[merge_name] = merged_pose
                merge_idx += 1

                # Update lists: remove merged cameras, add merge at front
                cam_type_list = _remove_indices(list(cam_type_list), min_indices)
                cam_type_list.insert(0, merge_name)
                pose_list = _remove_indices(list(pose_list), min_indices)
                pose_list.insert(0, merged_pose)

            # Final combination: should now have retrieval_num + 1 cameras
            # Pad with static if needed
            if len(cam_type_list) < retrieval_num + 1:
                num_static_needed = retrieval_num + 1 - len(cam_type_list)
                cam_type_list = ["static"] * num_static_needed + cam_type_list

            output_combinations.append(cam_type_list)

    return output_combinations


def load_plenoptic_camera_fn() -> CameraLoadFn:
    """Create a camera loading function for plenoptic multiview inference.

    This function integrates the memory retrieval algorithm from compute_camera_similarity.py
    to select the most informative conditioning cameras when there are more cameras than
    the model can condition on (num_input_video, typically 4).
    """
    # Import here to avoid circular imports
    # pyrefly: ignore[missing-import]
    from cosmos_transfer2._src.predict2.camera.datasets.camera_conditioned.dataset_utils import (
        combine_camera_sequences,
        compute_similarity_matrix,
        intrinsic_to_fov,
        scale_rotation_angle_and_translation,
    )

    def load_fn(
        text: str,
        video: torch.Tensor,
        path: str,
        base_path: str,
        latent_frames: int,
        width: int,
        height: int,
        input_video_res: str,
        focal_length: int,
        camera_list: list[str],
        extrinsic_scale: float,
    ) -> list[dict[str, Any]]:
        # Load intrinsics and compute FOV for similarity calculation
        K = np.loadtxt(os.path.join(base_path, "cameras", f"intrinsics_focal{focal_length}.txt"))
        intrinsics_matrix = torch.tensor([[K[0], 0.0, K[2]], [0.0, K[1], K[3]], [0.0, 0.0, 1.0]], dtype=torch.float)
        hfov_deg, vfov_deg = intrinsic_to_fov(intrinsics_matrix, width, height)

        # Load all camera extrinsics - always include "static" as base camera
        cam_data_dict: dict[str, torch.Tensor] = {}
        cameras_to_load = set(camera_list)
        cameras_to_load.add("static")  # Always load static as it's used for padding
        for cam_type in cameras_to_load:
            cam_data_dict[cam_type] = _load_single_camera_extrinsics(
                cam_type, base_path, latent_frames, extrinsic_scale, scale_rotation_angle_and_translation
            )

        # Generate camera combinations using memory retrieval algorithm
        # This selects the most informative conditioning cameras based on FOV similarity
        camera_combinations = _generate_camera_combinations_with_memory_retrieval(
            camera_list=camera_list,
            cam_data_dict=cam_data_dict,
            hfov_deg=hfov_deg,
            vfov_deg=vfov_deg,
            max_retrieval_range=6,
            retrieval_num=4,
            compute_similarity_fn=compute_similarity_matrix,
            combine_sequences_fn=combine_camera_sequences,
        )

        # Build result for each camera combination
        result: list[dict[str, Any]] = []

        for cam_combo in camera_combinations:
            data: dict[str, Any] = {"text": text, "video": video.clone(), "path": path}

            # Collect extrinsics for this combination
            extrinsics_list = []
            for cam_type in cam_combo:
                extrinsics_list.append(cam_data_dict[cam_type].to(torch.bfloat16))
            extrinsics = torch.cat(extrinsics_list, dim=0)

            # Build intrinsics
            intrinsics = torch.tensor(K).to(torch.bfloat16)
            intrinsics = intrinsics.unsqueeze(0).expand(extrinsics.shape[0], -1).clone()

            # Store camera combination for later use
            data["camera_data_list"] = cam_combo

            result.append(
                _finalize_camera_metadata(
                    data=data,
                    extrinsics=extrinsics,
                    intrinsics=intrinsics,
                    height=height,
                    width=width,
                )
            )

        return result

    return load_fn


class TextVideoCameraDataset(torch.utils.data.Dataset):
    """Dataset for loading video and camera data for plenoptic inference."""

    def __init__(
        self,
        base_path: str,
        args: PlenopticSetupArguments,
        inference_args: list[PlenopticInferenceArguments],
        num_frames: int,
        max_num_frames: int = 93,
        frame_interval: int = 1,
        patch_spatial: int = 16,
        camera_load_fn: CameraLoadFn | None = None,
    ):
        assert camera_load_fn is not None, "Camera load function must be provided"
        self.camera_load_fn = camera_load_fn
        self.base_path = base_path
        self.data = inference_args

        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.latent_frames = num_frames // 4 + 1
        self.patch_spatial = patch_spatial
        self.input_video_res = args.input_video_res
        self.num_input_video = args.num_input_video

        # Only 480p resolution is supported
        self.height, self.width = 432, 768

        self.frame_process = transforms.Compose(
            [
                transforms.CenterCrop(size=(self.height, self.width)),
                transforms.Resize(size=(self.height, self.width), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def crop_and_resize(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(  # pyrefly: ignore
            image,  # pyrefly: ignore
            (round(height * scale), round(width * scale)),  # pyrefly: ignore
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
        )
        return image

    def load_frames_using_imageio(
        self, file_path: str, max_num_frames: int, start_frame_id: int, interval: int, num_frames: int
    ) -> torch.Tensor | None:
        reader = imageio.get_reader(file_path)
        if (
            reader.count_frames() < max_num_frames  # pyrefly: ignore
            or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval  # pyrefly: ignore
        ):
            reader.close()
            return None

        frames = []
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        return frames

    def load_video(self, file_path: str) -> torch.Tensor | None:
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(
            file_path,
            self.max_num_frames,
            start_frame_id,  # pyrefly: ignore
            self.frame_interval,
            self.num_frames,
        )
        return frames

    def __getitem__(self, data_id: int) -> list[dict[str, Any]]:  # pyrefly: ignore[bad-param-name-override]
        inference_args = self.data[data_id]
        path = str(inference_args.input_path)
        text = inference_args.prompt or ""

        video = self.load_video(path)
        if video is None:
            raise ValueError(f"{path} is not a valid video.")
        video = video.repeat(1, self.num_input_video, 1, 1)

        result = self.camera_load_fn(
            text=text,
            video=video,
            path=path,
            base_path=self.base_path,
            latent_frames=self.latent_frames,
            width=self.width,
            height=self.height,
            input_video_res=self.input_video_res,
            focal_length=inference_args.focal_length,
            camera_list=inference_args.camera_sequence,
            extrinsic_scale=inference_args.extrinsic_scale,
        )

        for x in result:
            # camera_data_list is set by the memory retrieval algorithm
            # It contains the actual camera combination used (may include merged cameras)
            x.update(
                {
                    "name": inference_args.name,
                    "seed": inference_args.seed,
                    "guidance": inference_args.guidance,
                    "negative_prompt": inference_args.negative_prompt,
                    "original_camera_sequence": inference_args.camera_sequence,
                    "fps": inference_args.fps,
                }
            )

        return result

    def __len__(self) -> int:
        return len(self.data)


def inference(
    setup_args: PlenopticSetupArguments,
    all_inference_args: list[PlenopticInferenceArguments],
) -> list[str]:
    """Run plenoptic multiview inference.

    Args:
        setup_args: Setup arguments for model and environment.
        all_inference_args: List of inference arguments for each sample.

    Returns:
        List of output video paths.
    """
    assert len(all_inference_args) > 0

    create_camera_load_fn = load_callable(setup_args.camera_load_create_fn)
    dataset = TextVideoCameraDataset(
        base_path=str(setup_args.base_path),
        args=setup_args,
        inference_args=all_inference_args,
        num_frames=setup_args.num_output_frames,
        camera_load_fn=create_camera_load_fn(),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=setup_args.dataloader_num_workers,
    )

    checkpoint = MODEL_CHECKPOINTS[setup_args.model_key]
    experiment = setup_args.experiment or checkpoint.experiment
    checkpoint_path = setup_args.checkpoint_path or checkpoint.s3.uri

    vid2vid_cli = Video2WorldInference(
        experiment_name=experiment,  # pyrefly: ignore
        ckpt_path=checkpoint_path,
        s3_credential_path="",
        context_parallel_size=setup_args.context_parallel_size,  # pyrefly: ignore
    )

    mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    # Only process files on rank 0 if using distributed processing
    rank0 = True
    if setup_args.context_parallel_size > 1:  # pyrefly: ignore[unsupported-operation]
        rank0 = distributed.get_rank() == 0

    output_paths: list[str] = []

    # Process each sample
    for batch_idx, batch in enumerate(dataloader):
        for video_idx in range(len(batch)):
            ex = batch[video_idx]
            tgt_text = ex["text"][0] if isinstance(ex["text"], list) else ex["text"]
            src_video = ex["video"]
            fps = ex["fps"].item() if isinstance(ex["fps"], torch.Tensor) else ex["fps"]
            sample_name = ex["name"][0] if isinstance(ex["name"], list) else ex["name"]

            # camera_data_list contains the camera combination from memory retrieval
            # Format: [cond1, cond2, cond3, cond4, target] where some may be merged cameras
            camera_data_list = ex["camera_data_list"]
            # Handle DataLoader collation (strings become lists)
            if isinstance(camera_data_list[0], (list, tuple)):
                camera_data_list = [c[0] for c in camera_data_list]

            # Extract group name from sample name for output directory
            # E.g., "demo_0_rot_left" -> group "demo_0" to enable autoregressive generation
            # across multiple samples that share the same input video
            group_name = sample_name or f"video_{batch_idx}"
            # Strip the target camera suffix if name ends with _<camera_type>
            for cam_type in CAMERA_MOTION_TYPES:
                suffix = f"_{cam_type}"
                if group_name.endswith(suffix):
                    group_name = group_name[: -len(suffix)]
                    break
            save_root = Path(setup_args.output_dir) / experiment / group_name  # pyrefly: ignore
            save_root.mkdir(parents=True, exist_ok=True)

            # Clean up any old merge files
            for f in save_root.glob("merge*.mp4"):
                f.unlink()

            image_size = ex["image_size"] if "image_size" in ex else None

            # Get conditioning and target camera types from memory retrieval result
            camera_cond_type_list = camera_data_list[:-1]
            camera_tgt_type = camera_data_list[-1]

            # Use previously generated videos as conditioning inputs (autoregressive)
            src_video_lists = list(torch.chunk(src_video, len(camera_cond_type_list), dim=2))
            for camera_idx, camera_cond_type in enumerate(camera_cond_type_list):
                if camera_cond_type != "static":
                    cond_video_path = save_root / f"{camera_cond_type}.mp4"
                    if cond_video_path.exists():
                        cond_video, _, _ = io.read_video(str(cond_video_path), pts_unit="sec")
                        cond_video = (cond_video.to(src_video) / 127.5 - 1.0).permute(3, 0, 1, 2).unsqueeze(0)
                        if cond_video.shape == src_video_lists[camera_idx].shape:
                            src_video_lists[camera_idx] = cond_video
            src_video = torch.cat(src_video_lists, dim=2)

            video = vid2vid_cli.generate_vid2world(
                prompt=tgt_text,
                input_video=src_video,
                extrinsics=ex["extrinsics"],
                intrinsics=ex["intrinsics"],
                image_size=image_size,  # pyrefly: ignore
                num_input_video=setup_args.num_input_video,
                num_output_video=setup_args.num_output_video,
                num_latent_conditional_frames=setup_args.num_input_frames,
                seed=ex["seed"].item() if isinstance(ex["seed"], torch.Tensor) else ex["seed"],  # pyrefly: ignore
                guidance=ex["guidance"].item()  # pyrefly: ignore
                if isinstance(ex["guidance"], torch.Tensor)
                else ex["guidance"],
                negative_prompt=ex["negative_prompt"][0]
                if isinstance(ex["negative_prompt"], list)
                else ex["negative_prompt"],
            )

            if rank0:
                # Extract string from tuple/list (DataLoader collation wraps strings)
                output_name = camera_tgt_type[0] if isinstance(camera_tgt_type, (list, tuple)) else camera_tgt_type
                output_path = save_root / output_name
                save_img_or_video((1.0 + video[0]) / 2, str(output_path), fps=fps)  # pyrefly: ignore[bad-argument-type]
                logger.info(f"Saved video to {output_path}.mp4")
                output_paths.append(f"{output_path}.mp4")

    # Synchronize all processes before cleanup
    if setup_args.context_parallel_size > 1:  # pyrefly: ignore[unsupported-operation]
        torch.distributed.barrier()

    # Clean up distributed resources
    vid2vid_cli.cleanup()

    return output_paths
