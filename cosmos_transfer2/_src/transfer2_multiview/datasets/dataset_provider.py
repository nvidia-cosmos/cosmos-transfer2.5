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

import omegaconf

from cosmos_transfer2._src.imaginaire.datasets.webdataset.config.schema import DatasetConfig
from cosmos_transfer2._src.imaginaire.utils import log

try:
    from megatron.core import parallel_state

    USE_MEGATRON = True
except ImportError:
    USE_MEGATRON = False
from typing import Optional

from webdataset.handlers import warn_and_continue

import cosmos_transfer2._src.imaginaire.datasets.webdataset.decoders.pickle as pickle_decoders
import cosmos_transfer2._src.imaginaire.datasets.webdataset.distributors as distributors
import cosmos_transfer2._src.predict2.datasets.decoders.video_decoder as video_decoder  # noqa: F401
import cosmos_transfer2._src.predict2.datasets.distributor.parallel_sync_multi_aspect_ratio as parallel_sync_multi_aspect_ratio
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import AUGMENTOR_OPTIONS
from cosmos_transfer2._src.transfer2.datasets.data_sources.data_registration import DATASET_OPTIONS
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.driving import (
    MADSDrivingVideoDataloaderConfig,
)
from cosmos_transfer2._src.transfer2_multiview.datasets.mads_webdataset import MadsWebdataset

MULTI_VIEW_LOADING_KEYS = [
    "video_0",
    "video_1",
    "video_2",
    "video_3",
    "video_4",
    "video_5",
    "video_6",
    "hdmap_bbox_0",
    "hdmap_bbox_1",
    "hdmap_bbox_2",
    "hdmap_bbox_3",
    "hdmap_bbox_4",
    "hdmap_bbox_5",
    "hdmap_bbox_6",
    "t5_xxl_0",
    "t5_xxl_1",
    "t5_xxl_2",
    "t5_xxl_3",
    "t5_xxl_4",
    "t5_xxl_5",
    "t5_xxl_6",
    "metas",
]


def get_transfer2_multiview_dataset(
    *args,
    dataset_name: str,
    video_decoder_name: str,
    resolution: str,
    driving_dataloader_config: Optional[MADSDrivingVideoDataloaderConfig] = None,
    is_train: bool = True,
    num_video_frames: int = 121,
    chunk_size: int = 0,
    min_fps_thres: int = 10,
    max_fps_thres: int = 60,
    dataset_resolution_type: str = "all",
    augmentor_name: str = "video_basic_augmentor_v1",
    object_store: str = "s3",
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    detshuffle: bool = False,
    long_caption_ratio: int = 100,
    medium_caption_ratio: int = 0,
    short_caption_ratio: int = 0,
    user_caption_ratio: int = 0,
    use_native_fps: bool = False,
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    select_views: list[str] | None = None,
    # Not passed to get_video_dataset, but used in MadsWebdataset
    dataset_loading_keys=MULTI_VIEW_LOADING_KEYS,
    **kwargs,
) -> omegaconf.dictconfig.DictConfig:
    # Copied from transfer2 to add support for new augmentor, dataset_loading_keys, and URL level edits to mads dataset
    assert resolution in VIDEO_RES_SIZE_INFO.keys(), "The provided resolution cannot be found in VIDEO_RES_SIZE_INFO."
    assert object_store in ["s3", "swiftstack"], "We support s3 and swiftstack only."
    basic_augmentor_names = [
        "video_basic_augmentor_v2",
        "video_basic_augmentor_v2_with_control",
        "video_basic_augmentor_v2_with_control_and_image_context",
        "video_basic_augmentor_v2_multiview_with_control",
        "video_basic_augmentor_v3_multiview_with_control",
    ]
    if video_decoder_name == "video_naive_bytes":
        assert augmentor_name in basic_augmentor_names, (
            f"We can only use augmentors {basic_augmentor_names} with video_naive_bytes decoder. Got {augmentor_name}"
        )
    if augmentor_name in basic_augmentor_names:
        assert video_decoder_name == "video_naive_bytes", (
            f"{augmentor_name} can only be used with video_naive_bytes decoder. Got {video_decoder_name}"
        )

    assert dataset_resolution_type in [
        "all",
        "gt720p",
        "gt1080p",
    ], f"The provided dataset resolution type {dataset_resolution_type} is not supported."
    # dataset_resolution_type
    # -- all - uses all dataset resolutions
    # -- gt720p - Uses only resolutions >= 720p
    # -- gt1080p - Uses only resolutions >= 1080p
    dataset_info_fn = DATASET_OPTIONS[dataset_name]
    dataset_info = dataset_info_fn(object_store, caption_type, embedding_type, dataset_resolution_type)
    if dataset_loading_keys:
        for dset_info in dataset_info:
            dset_info.per_dataset_keys = dataset_loading_keys
    augmentor = AUGMENTOR_OPTIONS[augmentor_name](
        resolution=resolution,
        driving_dataloader_config=driving_dataloader_config,
        caption_type=caption_type,
        embedding_type=embedding_type,
        min_fps=min_fps_thres,
        max_fps=max_fps_thres,
        long_caption_ratio=long_caption_ratio,
        medium_caption_ratio=medium_caption_ratio,
        short_caption_ratio=short_caption_ratio,
        user_caption_ratio=user_caption_ratio,
        num_video_frames=num_video_frames,
        use_native_fps=use_native_fps,
        use_control_mask_prob=use_control_mask_prob,
        num_control_inputs_prob=num_control_inputs_prob,
        select_views=select_views,
    )

    if parallel_state.is_initialized() and (
        parallel_state.get_context_parallel_world_size() > 1
        or parallel_state.get_tensor_model_parallel_world_size() > 1
    ):
        log.critical(
            f"Using parallelism size CP :{parallel_state.get_context_parallel_world_size()}, TP :{parallel_state.get_tensor_model_parallel_world_size()} for video dataset, switch to ShardlistMultiAspectRatioParallelSync distributor"
        )
        distributor = parallel_sync_multi_aspect_ratio.ShardlistMultiAspectRatioParallelSync(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=True,
            is_infinite_loader=is_train,
        )
        detshuffle = True  # overwrite detshuffle.
    else:
        distributor = distributors.ShardlistMultiAspectRatio(
            shuffle=True,
            split_by_node=True,
            split_by_worker=True,
            resume_flag=True,
            verbose=False,
            is_infinite_loader=is_train,
        )

    video_data_config = DatasetConfig(
        keys=[],  # use the per_dataset_keys in DatasetInfo instead
        buffer_size=1,
        streaming_download=True,
        dataset_info=dataset_info,
        distributor=distributor,
        decoders=[
            video_decoder.construct_video_decoder(
                video_decoder_name=video_decoder_name,
                sequence_length=-1,
                chunk_size=chunk_size,
                min_fps_thres=min_fps_thres,
                max_fps_thres=max_fps_thres,
            ),
            pickle_decoders.pkl_decoder,
        ],
        augmentation=augmentor,
        remove_extension_from_keys=True,
        sample_keys_full_list_path=None,
    )
    return MadsWebdataset(
        config=video_data_config, decoder_handler=warn_and_continue, detshuffle=detshuffle, is_train=is_train
    )
