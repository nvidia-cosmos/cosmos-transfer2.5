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


import json
from collections import defaultdict
from pathlib import Path

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2_multiview.datasets.local import LocalMultiViewDataset
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import AugmentationConfig, collate_fn


class MultiviewTransferDataset(LocalMultiViewDataset):
    def __init__(
        self,
        dataset_dir: str,
        augmentation_config: AugmentationConfig,
        folder_to_camera_key: dict[str, str],
    ) -> None:
        self.dataset_dir = dataset_dir
        self.augmentation_config = augmentation_config
        self.folder_to_camera_key = folder_to_camera_key
        self.camera_key_to_folder = {v: k for k, v in folder_to_camera_key.items()}
        assert len(self.camera_key_to_folder) == len(self.folder_to_camera_key), (
            "camera_key_to_folder and folder_to_camera_key must have the same length!"
        )

        dataset_path = Path(dataset_dir)
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist!")

        control_path = dataset_path / "control_input_hdmap_bbox"
        if not control_path.exists():
            raise FileNotFoundError(f"Control input directory {control_path} does not exist!")

        video_path = dataset_path / "videos"
        if not video_path.exists():
            raise FileNotFoundError(f"Video directory {video_path} does not exist!")

        caption_path = dataset_path / "captions"
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption directory {caption_path} does not exist!")

        captions_files = [f for f in caption_path.glob("**/*.json")]
        unique_names = [f.stem for f in captions_files]

        captions_dict, video_files_dict, control_files_dict = defaultdict(dict), defaultdict(dict), defaultdict(dict)

        for name in unique_names:
            for folder, camera_key in self.folder_to_camera_key.items():
                caption_file = caption_path / folder / f"{name}.json"
                if caption_file.exists():
                    caption_json = json.load(open(caption_file, "r"))
                    captions_dict[name][camera_key] = caption_json["caption"]

                video_file = video_path / folder / f"{name}.mp4"
                if not video_file.exists():
                    raise FileNotFoundError(f"Expected video file {video_file} to exist!")

                control_file = control_path / folder / f"{name}.mp4"
                if not control_file.exists():
                    raise FileNotFoundError(f"Expected control file {control_file} to exist!")

                video_files_dict[name][camera_key] = video_file
                control_files_dict[name][camera_key] = control_file

        if len(captions_dict) != len(video_files_dict) != len(control_files_dict):
            raise ValueError("Number of captions, video files, and control files must be the same!")

        self.video_file_dicts = [video_files_dict[name] for name in unique_names]
        self.control_file_dicts = [control_files_dict[name] for name in unique_names]
        self.prompts = [
            captions_dict[name][self.augmentation_config.single_caption_camera_name] for name in unique_names
        ]

        super().__init__(
            video_file_dicts=self.video_file_dicts,
            prompts=self.prompts,
            augmentation_config=self.augmentation_config,
            control_file_dicts=self.control_file_dicts,
        )


#  NOTE 1: For customized post train: add your dataloader registration here.
def register_dataloader_local() -> None:
    from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
        DEFAULT_CAMERA_VIEW_MAPPING,
        DEFAULT_CAMERAS,
        DEFAULT_CAPTION_KEY_MAPPING,
        DEFAULT_CAPTION_PREFIXES,
        DEFAULT_VIDEO_KEY_MAPPING,
    )

    cs = ConfigStore()

    augmentation_config = L(AugmentationConfig)(
        resolution_hw=(720, 1280),
        fps_downsample_factor=1,
        num_video_frames=29,
        camera_keys=DEFAULT_CAMERAS if not SMOKE else DEFAULT_CAMERAS[:1],
        camera_view_mapping=DEFAULT_CAMERA_VIEW_MAPPING,
        camera_video_key_mapping=DEFAULT_VIDEO_KEY_MAPPING,
        camera_caption_key_mapping=DEFAULT_CAPTION_KEY_MAPPING,
        caption_probability={"dummy": 1.0},
        single_caption_camera_name="camera_front_wide_120fov",
        add_view_prefix_to_caption=True,
        camera_prefix_mapping=DEFAULT_CAPTION_PREFIXES,
        camera_control_key_mapping={camera_name: f"world_scenario_{camera_name}" for camera_name in DEFAULT_CAMERAS},
    )

    dataset = L(MultiviewTransferDataset)(
        dataset_dir="assets/multiview_hdmap_posttrain_dataset",
        augmentation_config=augmentation_config,
        folder_to_camera_key={f"ftheta_{camera_name}": camera_name for camera_name in DEFAULT_CAMERAS},
    )

    cs.store(
        group="data_train",
        package="dataloader_train",
        name=f"example_multiview_train_data_control_input_hdmap",
        node=L(get_generic_dataloader)(
            dataset=dataset,
            sampler=L(get_sampler)(dataset=dataset) if dist.is_initialized() else None,
            collate_fn=collate_fn,
            batch_size=1,
            drop_last=True,
            num_workers=4,
            pin_memory=True,
        ),
    )
