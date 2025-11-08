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

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred2madsreason7brffixdistmatch22k_cond02_hdmapbbox_highsigma_spaced_layer4_mlp
"""

MULTI_VIEW_LOADING_KEYS_V2 = [
    "video_0",
    "video_1",
    "video_2",
    "video_3",
    "video_4",
    "video_5",
    "video_6",
    "world_scenario_0",
    "world_scenario_1",
    "world_scenario_2",
    "world_scenario_3",
    "world_scenario_4",
    "world_scenario_5",
    "world_scenario_6",
    "t5_xxl_0",
    "t5_xxl_1",
    "t5_xxl_2",
    "t5_xxl_3",
    "t5_xxl_4",
    "t5_xxl_5",
    "t5_xxl_6",
    "metas",
]


def buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout():
    sample_n_views = 4
    state_t = 8
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
            {"override /data_train": f"video_only_cosmos_transfer2_av_mads_mv_20250710_480p_s3"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "load_base_model_callbacks",
                ]
            },
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
        ),
        dataloader_train=dict(
            dataset=dict(
                select_views=[
                    "camera_front_wide_120fov",
                    "camera_cross_left_120fov",
                    "camera_cross_right_120fov",
                    "camera_rear_tele_30fov",
                ],
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V2,
            ),
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t8_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform-0/checkpoints/iter_000031000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    view_condition_dim=7,
                    n_cameras_emb=7,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                    rope_enable_fps_modulation=False,
                ),
                resolution="480p",
                train_time_weight="uniform",
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k-0/checkpoints/iter_000010500",
            load_from_object_store=dict(
                enabled=True,
            ),
            save_iter=500,
        ),
        trainer=dict(
            straggler_detection=dict(
                enabled=False,
            ),
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                    sample_n_views=sample_n_views,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                    sample_n_views=sample_n_views,
                ),
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout0_debug():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout0_debug",
        ),
        model=dict(
            config=dict(
                conditioner=dict(
                    text=dict(
                        dropout_rate=0.0,
                        use_empty_string=False,
                    ),
                ),
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=100,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=100,
                ),
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_121framesto61_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase31k_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t8_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform-0/checkpoints/iter_000031000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_61frames_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t8_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform-0/checkpoints/iter_000031000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase4k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_61frames_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase4k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform_textdrop0_4viewdropout-0/checkpoints/iter_000004500",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout-0/checkpoints/iter_000004500",
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase10kselect4views6ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_61frames_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase10kselect4views6ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform_textdrop0_4viewdropout-0/checkpoints/iter_000010250",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1500,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase4k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout-0/checkpoints/iter_000006000",
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase7k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_121framesto61_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase7k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform_textdrop0_4viewdropout-0/checkpoints/iter_000007000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase31k_mads480pmulticaps61frames_world_scenario_4view_dropout-0/checkpoints/iter_000005000",
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase10kselect4views4p5ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout():
    state_t = 16
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_121framesto61_s3"},
        ],
        job=dict(
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase10kselect4views4p5ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout",
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from7kuniform7views_alpamayo1capviewprefix_allcapsviewprefix_61frames_nofps_uniform_textdrop0_4viewdropout-0/checkpoints/iter_000010250",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                state_t=state_t,
                net=dict(
                    state_t=state_t,
                    rope_t_extrapolation_ratio=float(state_t) / 24.0,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase7k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout-0/checkpoints/iter_000004500",
        ),
    )


experiments = [
    buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_frombase5ktdrop0_mads480pmulticaps29frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase5ktdrop0_mads480pmulticaps61frames_world_scenario_4view_dropout0_debug(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase4k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps30_t16_frombase10kselect4views6ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase7k5select4views_mads480pmulticaps61frames_world_scenario_4view_dropout(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t16_frombase10kselect4views4p5ktransfer_mads480pmulticaps61frames_world_scenario_4view_dropout(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
