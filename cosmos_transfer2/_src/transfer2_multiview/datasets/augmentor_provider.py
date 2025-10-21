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

from typing import Optional

import cosmos_transfer2._src.imaginaire.datasets.webdataset.augmentors.image.padding as padding
import cosmos_transfer2._src.imaginaire.datasets.webdataset.augmentors.image.resize as resize
import cosmos_transfer2._src.transfer2_multiview.datasets.augmentors.control_input as control_input
import cosmos_transfer2._src.transfer2_multiview.datasets.augmentors.merge_datadict_multiview_with_control as merge_datadict_multiview_with_control
import cosmos_transfer2._src.transfer2_multiview.datasets.augmentors.singleview_video_parsing as singleview_video_parsing
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.predict2.datasets.utils import VIDEO_RES_SIZE_INFO
from cosmos_transfer2._src.transfer2.datasets.augmentor_provider import augmentor_register
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.driving import (
    MADSDrivingVideoDataloaderConfig,
)
from cosmos_transfer2._src.transfer2_multiview.datasets.augmentors.text_transforms_for_multiview_video_with_control import (
    TextTransformForVideoCustomizedKey,
)


def get_multi_video_text_transform(
    caption_type: str,
    embedding_type: str,
    long_caption_ratio: int = 7,
    medium_caption_ratio: int = 2,
    short_caption_ratio: int = 1,
    user_caption_ratio: int = 90,
    driving_dataloader_config: Optional[MADSDrivingVideoDataloaderConfig] = None,
    num_videos: int = 7,
):
    """Create a multi-video text transform by composing the existing get_video_text_transform function."""

    # Get the base text transform
    augmentation = {}
    for id in range(num_videos):
        if caption_type == "vila_caption":
            video_text_transform = L(TextTransformForVideoCustomizedKey)(
                input_keys=[],
                args={
                    "captions_key": "metas",
                    "embeddings_key": embedding_type + f"_{id}" if embedding_type is not None else None,
                    "original_embeddings_key": f"t5_xxl_{id}",
                    "caption_windows_key": "windows",
                    "caption_type": "vila_caption",
                    "embedding_caption_type": "vila_caption",
                    "t5_tokens": {"num": 512},
                    "is_mask_all_ones": True,
                    "driving_dataloader_config": driving_dataloader_config,
                },
                return_embedding_key=f"t5_text_embeddings_{id}",
            )
        elif caption_type == "t2w_qwen2p5_7b":
            log.info(
                f"caption_type: {caption_type}, long_caption_ratio: {long_caption_ratio}, medium_caption_ratio: {medium_caption_ratio}, short_caption_ratio: {short_caption_ratio}, user_caption_ratio: {user_caption_ratio}"
            )
            video_text_transform = L(TextTransformForVideoCustomizedKey)(
                input_keys=[],
                args={
                    "captions_key": "metas",
                    "embeddings_key": embedding_type + f"_{id}" if embedding_type is not None else None,
                    "original_embeddings_key": f"t5_xxl_{id}",
                    "caption_windows_key": "t2w_windows",
                    "caption_type": "qwen2p5_7b_caption",
                    "embedding_caption_type": "t2w_qwen2p5_7b",
                    "t5_tokens": {"num": 512},
                    "is_mask_all_ones": True,
                    "caption_probs": {
                        "long": long_caption_ratio,
                        "medium": medium_caption_ratio,
                        "short": short_caption_ratio,
                        "user": user_caption_ratio,
                    },
                    "driving_dataloader_config": driving_dataloader_config,
                },
                return_embedding_key=f"t5_text_embeddings_{id}",
            )
        elif caption_type == "i2w_qwen2p5_7b_later_frames":
            video_text_transform = L(TextTransformForVideoCustomizedKey)(
                input_keys=[],
                args={
                    "captions_key": "metas",
                    "embeddings_key": embedding_type + f"_{id}" if embedding_type is not None else None,
                    "original_embeddings_key": f"t5_xxl_{id}",
                    "caption_windows_key": "i2w_windows_later_frames",
                    "caption_type": "qwen2p5_7b_caption",
                    "embedding_caption_type": "i2w_qwen2p5_7b_later_frames",
                    "t5_tokens": {"num": 512},
                    "is_mask_all_ones": True,
                    "caption_probs": {
                        "long": long_caption_ratio,
                        "medium": medium_caption_ratio,
                        "short": short_caption_ratio,
                        "user": user_caption_ratio,
                    },
                    "driving_dataloader_config": driving_dataloader_config,
                },
                return_embedding_key=f"t5_text_embeddings_{id}",
            )
        else:
            raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

        augmentation[f"text_transform_{id}"] = video_text_transform

    return augmentation


@augmentor_register("video_basic_augmentor_v2_multiview_with_control")
def get_video_augmentor_v2_multiview_with_control(
    resolution: str,
    caption_type: str = "t2w_qwen2p5_7b",
    embedding_type: str = "t5_xxl",
    min_fps: int = 10,
    max_fps: int = 60,
    driving_dataloader_config: Optional[MADSDrivingVideoDataloaderConfig] = None,
    long_caption_ratio: int = 100,
    medium_caption_ratio: int = 0,
    short_caption_ratio: int = 0,
    user_caption_ratio: int = 0,
    num_video_frames: int = -1,
    use_native_fps: bool = False,
    use_control_mask_prob: float = 0.0,
    num_control_inputs_prob: list[float] = [1.0, 0.0, 0.0, 0.0],
    **kwargs,
):
    """Multi-video augmentor that processes video_0 through video_6, t5_xxl_0 through t5_xxl_6,
    and hdmap_bbox_0 through hdmap_bbox_6, then concatenates them together.

    This augmentor handles:
    - Multiple video inputs (video_0 to video_6)
    - Multiple t5_xxl embeddings (t5_xxl_0 to t5_xxl_6)
    - Multiple hdmap_bbox inputs (hdmap_bbox_0 to hdmap_bbox_6)
    - Concatenation of all processed inputs
    """
    n_views = driving_dataloader_config.n_views
    # Use custom multi-video text transform that handles multiple metas and t5_xxl embeddings
    assert medium_caption_ratio == 0, "medium_caption_ratio not supported for multiview dataset"
    assert short_caption_ratio == 0, "short_caption_ratio not supported for multiview dataset"
    assert user_caption_ratio == 0, "user_caption_ratio not supported for multiview dataset"

    if caption_type == "t2w_qwen2p5_7b":
        key_for_caption = "t2w_windows"
    elif caption_type == "i2w_qwen2p5_7b_later_frames":
        key_for_caption = "i2w_windows_later_frames"
    else:
        raise ValueError(f"Unsupported caption type ({caption_type}) for video data")

    assert embedding_type in (
        "t5_xxl",
        "umt5_xxl",
        None,
    ), f"Unsupported embeddings type ({embedding_type}) for video data"

    # Create the augmentor configuration
    augmentor_config = {}

    # Optionally rename world_scenario to hdmap_bbox and umt5_xxl to t5_xxl
    augmentor_config["v2_to_v3_compatibility"] = L(merge_datadict_multiview_with_control.OptionalKeyRenamer)(
        input_keys=[f"world_scenario_{i}" for i in range(n_views)] + [f"umt5_xxl_{i}" for i in range(n_views)],
        output_keys=[f"hdmap_bbox_{i}" for i in range(n_views)] + [f"t5_xxl_{i}" for i in range(n_views)],
    )

    # Map correct camera keys to indices
    camera_to_view_id = driving_dataloader_config.camera_to_view_id
    video_id_to_camera_key = driving_dataloader_config.video_id_to_camera_key
    input_ids = video_id_to_camera_key.keys()
    output_ids = [str(camera_to_view_id[video_id_to_camera_key[i]]) for i in input_ids]
    augmentor_config["video_rename"] = L(merge_datadict_multiview_with_control.OptionalKeyRenamer)(
        input_keys=[f"video_{i}" for i in input_ids]
        + [f"hdmap_bbox_{i}" for i in input_ids]
        + [f"t5_xxl_{i}" for i in input_ids],
        output_keys=[f"video_{i}" for i in output_ids]
        + [f"hdmap_bbox_{i}" for i in output_ids]
        + [f"t5_xxl_{i}" for i in output_ids],
    )

    # Add video parsing for each video_i
    for i in range(n_views):  # video_0 to video_6
        augmentor_config[f"video_parsing_{i}"] = L(singleview_video_parsing.SingleViewVideoParsing)(
            input_keys=["metas", f"video_{i}"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 8,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "driving_dataloader_config": driving_dataloader_config,
            },
        )

    # Add hdmap_bbox parsing for each hdmap_bbox_i
    for i in range(n_views):  # hdmap_bbox_0 to hdmap_bbox_6
        augmentor_config[f"hdmap_parsing_{i}"] = L(singleview_video_parsing.SingleViewVideoParsing)(
            input_keys=["metas", f"hdmap_bbox_{i}"],
            args={
                "key_for_caption": key_for_caption,
                "min_duration": 4.0,
                "min_fps": min_fps,
                "max_fps": max_fps,
                "video_decode_num_threads": 8,
                "num_video_frames": num_video_frames,
                "use_native_fps": use_native_fps,
                "driving_dataloader_config": driving_dataloader_config,
            },
        )

    # Concatenate video and hdmap_bbox data (t5_xxl will be handled by text transform)

    video_keys = [f"video_{i}" for i in range(n_views)]
    hdmap_keys = [f"hdmap_bbox_{i}" for i in range(n_views)]

    # Text transform for embeddings - handles multiple metas and t5_xxl embeddings

    augmentor_config["concat_datadict"] = L(merge_datadict_multiview_with_control.DataDictConcatenator)(
        input_keys=video_keys + hdmap_keys,
        output_keys=[
            "video",  # Concatenated videos
            "hdmap_bbox",  # Concatenated hdmap_bbox
            # "t5_text_embeddings",
        ],
        args={
            "concat_dim": 1,  # Concatenate along temporal dimension
        },
    )

    # Resize and padding for concatenated videos
    augmentor_config["resize_largest_side_aspect_ratio_preserving"] = L(resize.ResizeLargestSideAspectPreserving)(
        input_keys=["video"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    augmentor_config["reflection_padding"] = L(padding.ReflectionPadding)(
        input_keys=["video"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    # Resize and padding for concatenated hdmap_bbox
    augmentor_config["resize_largest_side_aspect_ratio_preserving_hdmap"] = L(resize.ResizeLargestSideAspectPreserving)(
        input_keys=["hdmap_bbox"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    augmentor_config["reflection_padding_hdmap"] = L(padding.ReflectionPadding)(
        input_keys=["hdmap_bbox"],
        args={"size": VIDEO_RES_SIZE_INFO[resolution]},
    )

    augmentor_config["add_control_input"] = L(control_input.AddControlInputHdmapBbox)(
        input_keys=["hdmap_bbox"], output_keys=["control_input_hdmap_bbox"], use_random=False
    )

    augmentor_config["repackage_multiview"] = L(merge_datadict_multiview_with_control.AVMultiviewAdapter)(
        input_keys=[],
        args={
            "driving_dataloader_config": driving_dataloader_config,
            "embedding_type": None,
        },
    )

    return augmentor_config
