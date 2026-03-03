# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from pathlib import Path

import torch
from loguru import logger
from torchvision import io

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from cosmos_transfer2._src.imaginaire.utils import distributed
from cosmos_transfer2._src.imaginaire.visualize.video import save_img_or_video

# pyrefly: ignore[missing-import]
from cosmos_transfer2._src.predict2.camera.inference.multiview_camera_ar_video2world import Video2WorldInference
from cosmos_transfer2.config import MODEL_CHECKPOINTS, load_callable
from cosmos_transfer2.plenoptic import TextVideoCameraDataset
from cosmos_transfer2.plenoptic_config import (
    CAMERA_MOTION_TYPES,
    PlenopticInferenceArguments,
    PlenopticSetupArguments,
)


class MultiviewManyCamera_Worker:
    def __init__(
        self,
        num_gpus: int,
        disable_guardrails: bool = False,
        base_path: str | None = None,
    ):
        if base_path is None:
            base_path = os.environ.get("MULTIVIEW_MANY_CAMERA_BASE_PATH")
        if base_path is None:
            asset_dir = os.environ.get("ASSET_DIR")
            if asset_dir:
                base_path = os.path.join(asset_dir, "plenoptic_example")
        if base_path is None:
            raise ValueError(
                "base_path must be provided either as argument, via MULTIVIEW_MANY_CAMERA_BASE_PATH, "
                "or via ASSET_DIR environment variable"
            )

        self.setup_args = PlenopticSetupArguments(
            model="robot/multiview-many-camera",
            context_parallel_size=num_gpus,
            output_dir=Path("outputs"),
            keep_going=True,
            disable_guardrails=disable_guardrails,
            base_path=Path(base_path),
        )

        checkpoint = MODEL_CHECKPOINTS[self.setup_args.model_key]
        self.experiment = self.setup_args.experiment or checkpoint.experiment
        checkpoint_path = self.setup_args.checkpoint_path or checkpoint.s3.uri

        self.vid2vid_cli = Video2WorldInference(
            experiment_name=self.experiment,  # pyrefly: ignore
            ckpt_path=checkpoint_path,
            s3_credential_path="",
            context_parallel_size=self.setup_args.context_parallel_size,  # pyrefly: ignore
        )

        mem_bytes = torch.cuda.memory_allocated(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"GPU memory usage after model load: {mem_bytes / (1024**3):.2f} GB")

    def __del__(self):
        if hasattr(self, "vid2vid_cli") and self.vid2vid_cli is not None:
            self.vid2vid_cli.cleanup()

    def infer(self, args: dict):
        output_dir = args.pop("output_dir", "outputs")
        setup_args = self.setup_args.model_copy(update={"output_dir": Path(output_dir)})

        inference_args = PlenopticInferenceArguments(**args)

        create_camera_load_fn = load_callable(setup_args.camera_load_create_fn)
        dataset = TextVideoCameraDataset(
            base_path=str(setup_args.base_path),
            args=setup_args,
            inference_args=[inference_args],
            num_frames=setup_args.num_output_frames,
            camera_load_fn=create_camera_load_fn(),
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            shuffle=False,
            batch_size=1,
            num_workers=setup_args.dataloader_num_workers,
        )

        rank0 = True
        if setup_args.context_parallel_size > 1:  # pyrefly: ignore[unsupported-operation]
            rank0 = distributed.get_rank() == 0

        output_paths: list[str] = []

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
                group_name = sample_name or f"video_{batch_idx}"
                # Strip the target camera suffix if name ends with _<camera_type>
                for cam_type in CAMERA_MOTION_TYPES:
                    suffix = f"_{cam_type}"
                    if group_name.endswith(suffix):
                        group_name = group_name[: -len(suffix)]
                        break
                save_root = Path(setup_args.output_dir) / self.experiment / group_name  # pyrefly: ignore
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

                video = self.vid2vid_cli.generate_vid2world(
                    prompt=tgt_text,
                    input_video=src_video,
                    extrinsics=ex["extrinsics"],
                    intrinsics=ex["intrinsics"],
                    image_size=image_size,  # pyrefly: ignore
                    num_input_video=setup_args.num_input_video,
                    num_output_video=setup_args.num_output_video,
                    num_latent_conditional_frames=setup_args.num_input_frames,
                    seed=ex["seed"].item() if isinstance(ex["seed"], torch.Tensor) else ex["seed"],  # pyrefly: ignore
                    guidance=(
                        ex["guidance"].item()  # pyrefly: ignore[bad-argument-type]
                        if isinstance(ex["guidance"], torch.Tensor)
                        else ex["guidance"]
                    ),
                    negative_prompt=ex["negative_prompt"][0]
                    if isinstance(ex["negative_prompt"], list)
                    else ex["negative_prompt"],
                )

                if rank0:
                    output_name = camera_tgt_type[0] if isinstance(camera_tgt_type, (list, tuple)) else camera_tgt_type
                    output_path = save_root / output_name
                    save_img_or_video(
                        (1.0 + video[0]) / 2,
                        str(output_path),
                        fps=fps,  # pyrefly: ignore[bad-argument-type]
                    )
                    logger.info(f"Saved video to {output_path}.mp4")
                    output_paths.append(f"{output_path}.mp4")

        if setup_args.context_parallel_size > 1:  # pyrefly: ignore[unsupported-operation]
            torch.distributed.barrier()

        # NOTE: cleanup() is called in __del__ when the worker is deleted on server shutdown

        return {
            "videos": output_paths,
        }
