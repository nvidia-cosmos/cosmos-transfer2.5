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

import math

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.common.types.high_sigma_strategy import HighSigmaStrategy
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_edge_control_layer3_firstn
"""

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


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_edge_control_layer3_firstn():
    sample_n_views = 7
    return dict(
        defaults=[
            {
                "override /data_train": "alpamayo_v2_7cameras_tar_sample7views_85frames_res720_noviewprefix_1cap_norepeat"
            },
            {"override /model": "fsdp_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "log_sigma_loss",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_edge_control_layer3_firstn",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_from_object_store=dict(
                enabled=True,
            ),
            save_to_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
        ),
        optimizer=dict(
            lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[0.25],
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[40_000],
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=2,
                condition_locations=["first_random_n"],
                state_t=8,
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    vace_has_mask=False,  # just the latent of the control input, no inactive/active/mask stuff as in VACE paper (for now)
                    use_input_hint_block=True,
                    hint_nf=[16, 16, 32, 32, 96, 96, 256],
                    condition_strategy="first_n",
                    vace_block_every_n=9,  # 28 // 9 = 3 (first 3 blocks when condition_strategy="first_n")
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012-0/checkpoints/iter_000025000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                # Inherited from predict2 + transfer2 (also used in predict2_multiview)
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),
                high_sigma_ratio=0.05,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="720",
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                    compile_encode=False,
                ),
            ),
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            callbacks=dict(
                compile_tokenizer=dict(
                    enabled=False,
                ),
                iter_speed=dict(
                    hit_thres=50,
                    every_n=100,
                ),
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    is_x0=False,
                    is_ema=False,
                    num_sampling_step=35,
                    guidance=[7],
                    fps=10,
                    sample_n_views=sample_n_views,
                    dataset_name=None,
                    ctrl_hint_keys=["control_input_edge"],
                    control_weights=[0.0, 1.0],
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    is_x0=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[7],
                    fps=10,
                    sample_n_views=sample_n_views,
                    dataset_name=None,
                    ctrl_hint_keys=["control_input_edge"],
                    control_weights=[0.0, 1.0],
                ),
            ),
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            dataloaders=dict(
                image_data=dict(
                    ratio=0,
                ),
                video_data=dict(
                    dataloader=dict(),
                    ratio=1,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp
"""


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp():
    sample_n_views = 7
    return dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_s3"},
            {"override /model": "fsdp_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "log_sigma_loss",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_from_object_store=dict(
                enabled=True,
            ),
            save_to_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
        ),
        optimizer=dict(
            # lr=2 ** (-14),  # 2**(-14) = 6.103515625e-05 (32nodes)
            # lr=2 ** (-14.5),  # 2**(-14.5) = 3.0517578125e-05
            lr=2 ** (-13.5),
            weight_decay=0.1,
        ),
        scheduler=dict(
            f_max=[1.0],
            f_min=[0.4],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                hint_keys="hdmap_bbox",
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=2,
                condition_locations=["first_random_n"],
                state_t=8,
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    vace_has_mask=False,  # just the latent of the control input, no inactive/active/mask stuff as in VACE paper (for now)
                    use_input_hint_block=True,
                    condition_strategy="spaced",
                    vace_block_every_n=4,
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPre32k_alpamayo2tar_2p83s_noviewprefix_1cap_cond012-0/checkpoints/iter_000025000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                sde=dict(
                    p_mean=math.log(4.0),
                    p_std=1.2,
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                high_sigma_ratio=0.0,
                loss_scale=10.0,
                adjust_video_noise=True,
                scaling="rectified_flow",
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="720",
                resize_online=True,
                text_encoder_class="T5",
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                    compile_encode=False,
                ),
            ),
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=100,
            callbacks=dict(
                compile_tokenizer=dict(
                    enabled=False,
                ),
                iter_speed=dict(
                    hit_thres=50,
                    every_n=100,
                ),
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    is_x0=False,
                    is_ema=False,
                    num_sampling_step=35,
                    guidance=[7],
                    fps=10,
                    sample_n_views=sample_n_views,
                    dataset_name=None,
                    ctrl_hint_keys=["control_input_hdmap_bbox"],
                    control_weights=[0.0, 1.0],
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    is_x0=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[7],
                    fps=10,
                    sample_n_views=sample_n_views,
                    dataset_name=None,
                    ctrl_hint_keys=["control_input_hdmap_bbox"],
                    control_weights=[0.0, 1.0],
                ),
            ),
            straggler_detection=dict(
                enabled=False,
            ),
        ),
        dataloader_train=dict(
            batch_size=1,
            dataset=dict(
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS,
                augmentor_name="video_basic_augmentor_v2_multiview_with_control",
                video_decoder_name="video_naive_bytes",
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2_fmax0p25
"""


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2_fmax0p25():
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2_fmax0p25",
        ),
        scheduler=dict(
            f_max=[0.25],  # 2^3
            f_min=[0.1],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
        checkpoint=dict(
            save_iter=250,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2
"""


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2():
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2",
        ),
        scheduler=dict(
            f_max=[1.0],  # 2^5
            f_min=[0.4],
            warm_up_steps=[2_000],
            cycle_lengths=[400_000],
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
        checkpoint=dict(
            save_iter=250,
        ),
    )


experiments = [
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_edge_control_layer3_firstn(),
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp(),
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2_fmax0p25(),
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred25k_hdmap_bbox_oldsde_spaced_layer7_mlp_g2(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
