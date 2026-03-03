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
from typing import Any

import torch.distributed as dist
from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.predict2_multiview.datasets.local import LocalMultiViewDataset
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
    AGIBOT_CAPTION_KEY_MAPPING,
    AGIBOT_CAPTION_PREFIXES,
    AGIBOT_CONTROL_KEY_MAPPING,
    AGIBOT_VIDEO_KEY_MAPPING,
    AGIBOT_VIEW_MAPPING,
    AGIBOT_VIEWS,
    AugmentationConfig,
    collate_fn,
)
from cosmos_transfer2._src.transfer2.datasets.augmentors.control_input import AddControlInputBlur, AddControlInputEdge


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


# -----------------------------------------------------------------------------
# Agibot multiview: 3 views (head_color, hand_left, hand_right) + control (edge/depth/seg/vis).
# Same parent as MultiviewTransferDataset; Layout: videos/{view}/*.mp4,
# captions/{view}/*.json; for depth/seg add control_input_depth/ or control_input_seg/ with same structure.
# -----------------------------------------------------------------------------


def _resolve_agibot_dataset_dir(dataset_dir: str) -> str:
    """Resolve relative dataset_dir from cwd so paths like datasets/your_dataset work."""
    path = Path(dataset_dir)
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path)


class AgibotMultiviewLocalDataset(LocalMultiViewDataset):
    """3-view Agibot local dataset: videos + captions + optional control (edge/depth/seg/vis). Inherits from LocalMultiViewDataset like MultiviewTransferDataset but no camera/HDMap; only 3 views + control."""

    def __init__(
        self,
        dataset_dir: str,
        control_input_type: str,
        augmentation_config: AugmentationConfig,
    ) -> None:
        self.control_input_type = control_input_type
        dataset_dir = _resolve_agibot_dataset_dir(dataset_dir)
        root = Path(dataset_dir)
        if not root.exists():
            raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist!")
        video_path = root / "videos"
        caption_path = root / "captions"
        if not video_path.exists():
            raise FileNotFoundError(f"Video directory {video_path} does not exist!")
        if not caption_path.exists():
            raise FileNotFoundError(f"Caption directory {caption_path} does not exist!")
        for view in AGIBOT_VIEWS:
            if not (video_path / view).exists():
                raise FileNotFoundError(f"Expected videos/{view}/ under {dataset_dir}")

        sample_ids = sorted({f.stem for f in (video_path / "head_color").glob("*.mp4")})
        if not sample_ids:
            raise FileNotFoundError(f"No .mp4 files under {video_path / 'head_color'}")

        need_control_files = control_input_type in ("depth", "seg")
        if need_control_files:
            control_path = root / f"control_input_{control_input_type}"
            if not control_path.exists():
                raise FileNotFoundError(f"For control type {control_input_type} expect directory {control_path}")
        else:
            control_path = None

        video_file_dicts: list[dict[str, Path]] = []
        control_file_dicts: list[dict[str, Path]] | None = [] if control_path else None
        prompts: list[str] = []

        for sample_id in sample_ids:
            v_dict: dict[str, Path] = {}
            c_dict: dict[str, Path] = {} if control_path else {}
            prompt_caption = ""

            for view in AGIBOT_VIEWS:
                v_file = video_path / view / f"{sample_id}.mp4"
                if not v_file.exists():
                    raise FileNotFoundError(f"Expected video {v_file}")
                v_dict[view] = v_file

                cap_file = caption_path / view / f"{sample_id}.json"
                if cap_file.exists():
                    with open(cap_file) as f:
                        cap = json.load(f).get("caption", "")
                    if view == "head_color":
                        prompt_caption = cap

                if control_path is not None:
                    c_file = control_path / view / f"{sample_id}.mp4"
                    if not c_file.exists():
                        raise FileNotFoundError(f"Expected control {c_file}")
                    c_dict[view] = c_file

            video_file_dicts.append(v_dict)
            if control_file_dicts is not None:
                control_file_dicts.append(c_dict)
            prompts.append(prompt_caption)

        super().__init__(
            video_file_dicts=video_file_dicts,
            prompts=prompts,
            augmentation_config=augmentation_config,
            control_file_dicts=control_file_dicts,
        )

        if control_input_type in ("edge", "vis"):
            self._add_control = (
                AddControlInputEdge(input_keys=["video"], output_keys=["control_input_edge"], use_random=True)
                if control_input_type == "edge"
                else AddControlInputBlur(input_keys=["video"], output_keys=["control_input_vis"], use_random=True)
            )
        else:
            self._add_control = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        data = super().__getitem__(index)
        if self.control_input_type == "edge":
            data["control_input_edge"] = self._add_control({"video": data["video"]})["control_input_edge"]
        elif self.control_input_type == "vis":
            data["control_input_vis"] = self._add_control({"video": data["video"]})["control_input_vis"]
        elif self.control_input_type == "depth":
            # Parent puts control_head_color/etc. in data; ExtractFramesAndCaptions builds control_input_hdmap_bbox.
            data["control_input_depth"] = data.pop("control_input_hdmap_bbox")
        elif self.control_input_type == "seg":
            data["control_input_seg"] = data.pop("control_input_hdmap_bbox")
        return data


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

    # Agibot 3-view multicontrol (edge, depth, seg, vis)
    # num_video_frames must match model: tokenizer.get_pixel_num_frames(state_t) = (24-1)*4+1 = 93
    for ctrl_type in ("edge", "depth", "seg", "vis"):
        agibot_aug = L(AugmentationConfig)(
            resolution_hw=(432, 768),
            fps_downsample_factor=1,
            num_video_frames=93,
            camera_keys=AGIBOT_VIEWS,
            camera_view_mapping=AGIBOT_VIEW_MAPPING,
            camera_video_key_mapping=AGIBOT_VIDEO_KEY_MAPPING,
            camera_caption_key_mapping=AGIBOT_CAPTION_KEY_MAPPING,
            caption_probability={"dummy": 1.0},
            single_caption_camera_name="head_color",
            add_view_prefix_to_caption=True,
            camera_prefix_mapping=AGIBOT_CAPTION_PREFIXES,
            camera_control_key_mapping=AGIBOT_CONTROL_KEY_MAPPING if ctrl_type in ("depth", "seg") else None,
        )
        agibot_ds = L(AgibotMultiviewLocalDataset)(
            dataset_dir="assets/agibot_posttrain",
            control_input_type=ctrl_type,
            augmentation_config=agibot_aug,
        )
        cs.store(
            group="data_train",
            package="dataloader_train",
            name=f"example_agibot_multiview_train_data_{ctrl_type}",
            node=L(get_generic_dataloader)(
                dataset=agibot_ds,
                sampler=L(get_sampler)(dataset=agibot_ds) if dist.is_initialized() else None,
                collate_fn=collate_fn,
                batch_size=1,
                drop_last=True,
                num_workers=4,
                pin_memory=True,
            ),
        )
