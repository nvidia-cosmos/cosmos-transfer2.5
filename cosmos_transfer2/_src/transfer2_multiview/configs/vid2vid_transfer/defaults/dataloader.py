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

from copy import deepcopy

from hydra.core.config_store import ConfigStore

# DO NOT REMOVE THIS LINE. Ensures that AV dataset is registered before creating the dataloader. During training, cosmos dataset registration occurs via imports, not direct function calls
import cosmos_transfer2._src.transfer2_multiview.datasets.data_sources.data_registration  # noqa: F401
from cosmos_transfer2._src.predict2.configs.common.mock_data import (
    MOCK_DATA_IMAGE_ONLY_CONFIG,
    MOCK_DATA_INTERLEAVE_CONFIG,
    MOCK_DATA_VIDEO_ONLY_CONFIG,
)
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.dataloader import get_video_dataloader_multiview
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)


def register_training_and_val_data():
    cs = ConfigStore()
    cs.store(group="data_train", package="dataloader_train", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_image", node=MOCK_DATA_IMAGE_ONLY_CONFIG)
    cs.store(group="data_train", package="dataloader_train", name="mock_video", node=MOCK_DATA_VIDEO_ONLY_CONFIG)
    cs.store(group="data_val", package="dataloader_val", name="mock", node=MOCK_DATA_INTERLEAVE_CONFIG)
    for training_type in ["train", "val"]:  # register datasets for both training and validation
        for object_store in ["s3", "swiftstack"]:
            for video_version_by_date in ["transfer2_av_mads_mv_20250710", "transfer2_av_mads_mv_20250823"]:
                for resolution in ["720", "720p", "480p"]:
                    for num_video_frames_loaded_per_view in [121, 85, 61, 29]:
                        for num_video_frames_per_view in [29, 61]:
                            mads_driving_dataloader_config = deepcopy(
                                MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[resolution]
                            )
                            mads_driving_dataloader_config.num_video_frames_loaded_per_view = (
                                num_video_frames_loaded_per_view
                            )
                            if video_version_by_date == "transfer2_av_mads_mv_20250823":
                                mads_driving_dataloader_config.override_original_fps = 30.0
                            mads_driving_dataloader_config.num_video_frames_per_view = num_video_frames_per_view
                            resolution_str = "" if resolution == "720" else f"_{resolution}"
                            if num_video_frames_loaded_per_view == 85 and num_video_frames_per_view == 29:
                                frames_str = ""  # default 10fps dataset
                            elif num_video_frames_loaded_per_view == num_video_frames_per_view:
                                frames_str = f"_{num_video_frames_per_view}frames"
                            else:
                                frames_str = f"_{num_video_frames_loaded_per_view}framesto{num_video_frames_per_view}"
                            cs.store(
                                group=f"data_{training_type}",
                                package=f"dataloader_{training_type}",
                                name=f"video_only_cosmos_{video_version_by_date}{resolution_str}{frames_str}_{object_store}",
                                node=get_video_dataloader_multiview(
                                    dataset_name=f"cosmos_{video_version_by_date}_video_whole",
                                    object_store=object_store,
                                    resolution=resolution,
                                    driving_dataloader_config=mads_driving_dataloader_config,
                                ),
                            )
