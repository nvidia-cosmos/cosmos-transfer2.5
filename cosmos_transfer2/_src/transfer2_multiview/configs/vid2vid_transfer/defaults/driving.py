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


from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION as MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION,
)
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.driving import (
    MADSDrivingVideoDataloaderConfig as MADSDrivingVideoDataloaderConfig,
)


def setup_config(
    resolution: str,
    enable_autoregressive: bool,
    num_video_frames_per_view: int,
    minimum_start_index: int,
    num_video_frames_loaded_per_view: int,
    n_views: int,
):
    config = MADS_DRIVING_DATALOADER_CONFIG_PER_RESOLUTION[resolution]
    if enable_autoregressive:
        config.num_video_frames_per_view = num_video_frames_per_view
        config.minimum_start_index = minimum_start_index
        config.num_video_frames_loaded_per_view = num_video_frames_loaded_per_view
        config.n_views = n_views
        return config
    else:
        config.n_views = n_views
        return config
