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

import ctypes
import gc
import io
import random
from typing import Optional

import av
import numpy as np
import torch

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.augmentors.video_parsing import VideoParsing


class SingleViewVideoParsing(VideoParsing):
    """
    Merge multiple views of the same video into a single video.
    """

    def __init__(self, input_keys: list, output_keys: Optional[list] = None, args: Optional[dict] = None) -> None:
        super().__init__(input_keys, output_keys, args)
        self.driving_dataloader_config = args["driving_dataloader_config"]
        self.num_video_frames_per_view = self.driving_dataloader_config.num_video_frames_per_view
        self.num_video_frames_loaded_per_view = self.driving_dataloader_config.num_video_frames_loaded_per_view
        self.num_video_frames_per_view = self.driving_dataloader_config.num_video_frames_per_view
        mult = self.num_video_frames_loaded_per_view - 1
        div = self.num_video_frames_per_view - 1
        assert mult % div == 0, (
            f"num_video_frames_loaded_per_view ({self.num_video_frames_loaded_per_view}) - 1 must be divisible by num_video_frames_per_view ({self.num_video_frames_per_view}) - 1. Got {mult} % {div} = {mult % div}"
        )
        self.temporal_subsampling_factor = mult // div
        # minimum start index. From alpamayo dataverse
        # NOTE V2 data might have issue in the first few frames, set a minimum to exclude them.
        self.minimum_start_index = self.driving_dataloader_config.minimum_start_index
        self.H = self.driving_dataloader_config.H
        self.W = self.driving_dataloader_config.W

        self.batch_counter = 0
        self.gc_every_n = 100

    def probe_video_length(self, video_bytes: bytes) -> tuple[int, float]:
        """
        Return (num_frames, fps) for a video contained in `video_bytes`.
        This opens the container, looks at the stream header, and closes it
        — no frame decoding is performed.

        Args:
            video_bytes: Binary contents of an .mp4/.mkv/etc. file.

        Returns:
            n_frames (int): total number of frames in the stream.
            fps      (float): average frame rate reported by the container.
        """
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            n_frames = stream.frames  # populated from header
            fps = float(stream.average_rate)
            if not n_frames or n_frames <= 0:  # extremely rare fallback
                n_frames = int(stream.duration * stream.average_rate)
            if fps < self.min_fps:
                log.warning(f"Video FPS {fps} is less than min_fps {self.min_fps}", rank0_only=False)
                return None
            if fps > self.max_fps:
                log.warning(f"Video FPS {fps} is greater than max_fps {self.max_fps}", rank0_only=False)
                return None
            return n_frames, fps

    def _decode_view(
        self,
        cam_key: str,
        video_bytes: bytes,
        start_idx: int,
        end_idx: int,
        H: int,
        W: int,
    ) -> torch.Tensor:
        n_frames = end_idx - start_idx
        if n_frames <= 0:
            raise ValueError("end_idx must be > start_idx")

        video_buffer = io.BytesIO(video_bytes)
        frames_np = np.empty((self.num_video_frames_per_view, H, W, 3), dtype=np.uint8)
        log.debug(f"SingleViewVideoParsing: Temporal subsampling factor: {self.temporal_subsampling_factor}")
        with av.open(video_buffer, options={"threads": str(self.video_decode_num_threads)}) as container:
            stream = container.streams.video[0]
            frame_pos = 0
            for frame_idx, frame in enumerate(container.decode(stream)):
                if frame_idx < start_idx:
                    continue
                if (frame_idx - start_idx) % self.temporal_subsampling_factor != 0:
                    continue
                if frame_idx >= end_idx:
                    break

                # Resize + Color convert in C (libswscale)
                frame_resized = frame.reformat(width=W, height=H, format="rgb24")
                # Directly write the C-contiguous ndarray into pre-allocated buffer.
                frames_np[frame_pos] = frame_resized.to_ndarray()
                frame_pos += 1

        if frame_pos != self.num_video_frames_per_view:
            raise RuntimeError(
                f"Decoded {frame_pos} frames for {cam_key}, expected {self.num_video_frames_per_view} "
                f"({start_idx=}, {end_idx=}, {self.temporal_subsampling_factor=})"
            )

        # Convert once to torch (shares memory) and permute to (C, T, H, W).
        view_video_frames = torch.from_numpy(frames_np).permute(3, 0, 1, 2)
        return view_video_frames

    def __call__(self, data_dict: dict) -> Optional[dict]:
        try:
            if self.batch_counter % self.gc_every_n == 0:
                gc.collect()
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            self.batch_counter = (self.batch_counter + 1) % (self.gc_every_n + 1)
            sample_key = data_dict["__key__"]

            n_orig_video_frames, fps = self.probe_video_length(data_dict[self.video_key])
            meta_dict = {
                "nb_frames": n_orig_video_frames,
                "framerate": fps,
                self.key_for_caption: [
                    {
                        "start_frame": 0,
                        "end_frame": 128,
                    }
                ],
            }

            if isinstance(meta_dict, list):
                meta_dict = meta_dict[0]
            options: list = list((i, item) for i, item in enumerate(meta_dict[self.key_for_caption]))
            assert len(options) == 1, f"Expected 1 option for MADS, got {len(options)}"
            random.shuffle(options)
            for chunk_index, option in options:
                start_frame = option["start_frame"]
                end_frame = option["end_frame"]
                assert start_frame == 0, f"MADS start_frame must be 0, got {start_frame}"
                assert end_frame in (100, 128), f"MADS end_frame should be 100 or 128, got {end_frame}"
                start_frame = self.minimum_start_index
                end_frame = self.minimum_start_index + self.num_video_frames_loaded_per_view

                frame_indices = np.arange(start_frame, end_frame).tolist()
                video = self._decode_view(
                    cam_key=self.video_key,
                    video_bytes=data_dict[self.video_key],
                    start_idx=start_frame,
                    end_idx=end_frame,
                    H=self.H,
                    W=self.W,
                )
                log.debug(f"SingleViewVideoParsing: Video shape: {video.shape}")
                data_dict[self.video_key] = {
                    "video": video,
                    "fps": (
                        meta_dict["framerate"]
                        if self.driving_dataloader_config.override_original_fps is None
                        else self.driving_dataloader_config.override_original_fps
                    ),
                    "n_orig_video_frames": meta_dict["nb_frames"] if "nb_frames" in meta_dict else None,
                    "chunk_index": chunk_index,
                    "frame_indices": frame_indices,
                    "frame_start": start_frame,
                    "frame_end": end_frame,
                    "num_frames": end_frame - start_frame,
                    "temporal_subsampling_factor": self.temporal_subsampling_factor,
                }
                return data_dict
        except Exception as e:
            log.error(f"Error in SingleViewVideoParsing: {e}", rank0_only=False)
            log.error(f"Sample keys: {data_dict.keys()}", rank0_only=False)
            log.error(f"Sample key: {sample_key}", rank0_only=False)
            if "__url__" in data_dict:
                log.error(f"__url__: {data_dict['__url__']}", rank0_only=False)
            if "__key__" in data_dict:
                log.error(f"__key__: {data_dict['__key__']}", rank0_only=False)
            return None
