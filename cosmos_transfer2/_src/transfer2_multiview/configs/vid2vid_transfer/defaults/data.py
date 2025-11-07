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


from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import SMOKE
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.datasets.local_datasets.dataset_video import get_generic_dataloader, get_sampler
from cosmos_transfer2._src.transfer2.datasets.local_datasets.multiview_dataset import MultiviewTransferDataset


def get_hdmap_multiview_dataset(is_train=True):
    camera_keys = [
        "ftheta_camera_front_wide_120fov",
        "ftheta_camera_cross_left_120fov",
        "ftheta_camera_cross_right_120fov",
        "ftheta_camera_rear_left_70fov",
        "ftheta_camera_rear_right_70fov",
        "ftheta_camera_rear_tele_30fov",
        "ftheta_camera_front_tele_30fov",
    ]
    if SMOKE:
        camera_keys = camera_keys[:1]
    camera_to_view_id = {
        "ftheta_camera_front_wide_120fov": 0,
        "ftheta_camera_cross_left_120fov": 1,
        "ftheta_camera_cross_right_120fov": 2,
        "ftheta_camera_rear_left_70fov": 3,
        "ftheta_camera_rear_right_70fov": 4,
        "ftheta_camera_rear_tele_30fov": 5,
        "ftheta_camera_front_tele_30fov": 6,
    }

    dataset = L(MultiviewTransferDataset)(
        dataset_dir="assets/multiview_hdmap_posttrain_dataset",
        hint_key="control_input_hdmap_bbox",
        resolution="720",
        state_t=8,
        num_frames=29,
        sequence_interval=1,
        camera_keys=camera_keys,
        video_size=(704, 1280),
        front_camera_key="ftheta_camera_front_wide_120fov",
        camera_to_view_id=camera_to_view_id,
        front_view_caption_only=True,
        is_train=True,
    )
    return L(get_generic_dataloader)(
        dataset=dataset,
        sampler=L(get_sampler)(dataset=dataset),
        batch_size=1,
        drop_last=True,
        num_workers=8,
        prefetch_factor=2,
        pin_memory=True,
    )


#  NOTE 1: For customized post train: add your dataloader registration here.
def register_data_ctrlnet():
    cs = ConfigStore()
    cs.store(
        group="data_train",
        package="dataloader_train",
        name=f"example_multiview_train_data_control_input_hdmap",
        node=get_hdmap_multiview_dataset(is_train=True),
    )
