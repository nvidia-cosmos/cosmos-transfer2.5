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


from typing import Final

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.configs.common.mock_data import (
    MOCK_DATA_VIDEO_ONLY_CONFIG,
)
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.dataloader import (
    DEFAULT_CAMERA_VIEW_CONFIGS,
)
from cosmos_transfer2._src.predict2_multiview.datasets.multiview import (
    AugmentationConfig,
    get_multiview_video_loader,
)
from cosmos_transfer2._src.predict2_multiview.datasets.wdinfo_utils import DEFAULT_CATALOG

INDEX_TO_CAMERA_MAPPING: Final = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_right_70fov",
    5: "camera_rear_tele_30fov",
    6: "camera_front_tele_30fov",
}

DEFAULT_VIDEO_KEY_MAPPING: Final = {camera_name: f"video_{i}" for i, camera_name in INDEX_TO_CAMERA_MAPPING.items()}
DEFAULT_CONTROL_KEY_MAPPING: Final = {
    camera_name: f"world_scenario_{i}" for i, camera_name in INDEX_TO_CAMERA_MAPPING.items()
}
# this is unpacked from metas dict
DEFAULT_CAMERA_CAPTION_KEY_MAPPING: Final = {
    camera_name: f"metas_{camera_name}" for _, camera_name in INDEX_TO_CAMERA_MAPPING.items()
}


def register_dataloaders() -> None:
    cs = ConfigStore.instance()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_VIDEO_ONLY_CONFIG)

    for object_store in ["s3", "gcs"]:
        for dataset_name in ["mads_multiview_0823"]:
            assert dataset_name in DEFAULT_CATALOG, (
                f"Dataset {dataset_name} not found in catalog [{DEFAULT_CATALOG.keys()}]"
            )
            for resolution_str, resolution_hw in [("480p", (480, 832)), ("720p", (720, 1280))]:
                for fps_str, fps_downsample_factor in [
                    ("10fps", 1)
                ]:  # N.B. mads_multiview_0823 dataset has videos at 10Hz, not 30Hz!
                    for num_video_frames_str, num_video_frames in [
                        ("29frames", 29),
                        ("61frames", 61),
                        ("93frames", 93),
                    ]:
                        for views_str, camera_keys in DEFAULT_CAMERA_VIEW_CONFIGS.items():
                            name = f"video_control_{dataset_name}_{object_store}_{resolution_str}_{fps_str}_{num_video_frames_str}_{views_str}"
                            cs.store(
                                group=f"data_train",
                                package=f"dataloader_train",
                                name=name,
                                node=L(get_multiview_video_loader)(
                                    is_train=True,
                                    dataset_name=dataset_name,
                                    object_store=object_store,
                                    augmentation_config=L(AugmentationConfig)(
                                        resolution_hw=resolution_hw,
                                        fps_downsample_factor=fps_downsample_factor,
                                        num_video_frames=num_video_frames,
                                        camera_keys=camera_keys,
                                        camera_video_key_mapping=DEFAULT_VIDEO_KEY_MAPPING,
                                        camera_caption_key_mapping=DEFAULT_CAMERA_CAPTION_KEY_MAPPING,
                                        camera_control_key_mapping=DEFAULT_CONTROL_KEY_MAPPING,
                                        position_to_camera_mapping=INDEX_TO_CAMERA_MAPPING,
                                    ),
                                ),
                            )
