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

from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2.models.video2world_model import HighSigmaStrategy
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)
from cosmos_transfer2._src.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from cosmos_transfer2._src.reason1.models.vlm_qwen_omni import QwenVLBaseModel
from cosmos_transfer2._src.reason1.tokenizer.processor import build_tokenizer
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import (
    TextAttrEmptyStringDropout,
)

"""

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred2madsreason7brffixdistmatch22k_cond02_hdmapbbox_highsigma_spaced_layer4_mlp
"""


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred2madsreason7brffixdistmatch22k_cond02_hdmapbbox_highsigma_spaced_layer4_mlp():
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
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred2madsreason7brffixdistmatch22k_cond02_hdmapbbox_highsigma_spaced_layer4_mlp",
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
            lr=2 ** (-14.5),
            weight_decay=0.001,  # updated from 0.1
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[2_000],
            cycle_lengths=[100_000],
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                hint_keys="hdmap_bbox",
                adjust_video_noise=False,  # updated from True
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=2,  # i2w or v2v
                condition_locations=["first_random_n"],
                state_t=8,
                sde=dict(
                    p_mean=math.log(5.0),  # updated from 4.0
                    p_std=1.0,  # updated from 1.2
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                sigma_data=1.0,
                high_sigma_strategy=str(HighSigmaStrategy.LOGUNIFORM200_100000),  # Updated
                high_sigma_ratio=0.05,  # Updated
                loss_scale=10.0,
                scaling="rectified_flow",
                rectified_flow_loss_weight_uniform=False,  # this has no impact
                net=dict(
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    vace_has_mask=False,
                    use_input_hint_block=True,
                    condition_strategy="spaced",
                    vace_block_every_n=7,  # 4 layers
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                ),
                text_encoder_class="reason1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp500/sft_exp510_qwen7b_w_critique_n32/checkpoints/iter_000008000/model/",
                    model_config=L(QwenVLBaseModel)(
                        model_config=L(QwenModelConfig)(
                            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
                            name_or_path="Qwen/Qwen2.5-VL-7B-Instruct",
                            hidden_size=3584,
                            intermediate_size=18944,
                            max_window_layers=28,
                            num_attention_heads=28,
                            num_hidden_layers=28,
                            num_key_value_heads=4,
                            tie_word_embeddings=False,
                            vocab_size=152064,
                            vision_config=L(QwenVisionConfig)(out_hidden_size=3584),
                            output_hidden_states=True,
                        ),
                        tokenizer=L(build_tokenizer)(
                            tokenizer_type="Qwen/Qwen2.5-VL-7B-Instruct",
                        ),
                    ),
                ),
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromv2base22p5k_mads_reason7b_noviewprefix_1cap_cond02_rffix_distmatch-0/checkpoints/iter_000022000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                fsdp_shard_size=8,
                resolution="720",
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=L(TextAttrEmptyStringDropout)(
                        use_empty_string=False,  # set to False
                        input_key="t5_text_embeddings",
                        pos_input_key="text_embeddings",
                        dropout_input_key="dropout_text_embeddings",
                        dropout_rate=0.2,
                    ),
                    control_input_hdmap_bbox=dict(
                        dropout_rate=0.0,
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
            dataset=dict(
                embedding_type=None,
            ),
        ),
        upload_reproducible_setup=True,
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox
"""


def buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox():
    sample_n_views = 7
    text_encoder_ckpt_path = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/"
    base_load_path = "bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames-0/checkpoints/iter_000028000"
    base_load_credentials = "credentials/s3_checkpoint.secret"

    return dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_720p_s3"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    # "log_sigma_loss",  #check
                    # "load_base_model_callbacks",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
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
            lr=8.63e-5,  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[100],
            cycle_lengths=[100_000],
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                hint_keys="hdmap_bbox",
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=2,  # i2w or v2v
                condition_locations=["first_random_n"],
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                state_t=8,
                online_text_embeddings_as_dict=False,  # For backward compatibility with old experiments
                fsdp_shard_size=8,
                resolution="720p",  # Updated from 720 to get resolution 720 x 1280 instead of 704 x 1280
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    timestep_scale=0.001,
                    use_wan_fp32_strategy=True,
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    vace_has_mask=False,
                    use_input_hint_block=True,
                    condition_strategy="spaced",
                    vace_block_every_n=7,  # 4 layers
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,  # (TODO: hanzim): check
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                    compile_encode=False,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path=text_encoder_ckpt_path,
                ),
                base_load_from=dict(
                    load_path=base_load_path,
                    credentials=base_load_credentials,
                ),
            )
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
                grad_clip=dict(
                    clip_norm=0.1,
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
            dataset=dict(
                embedding_type=None,
            ),
        ),
        upload_reproducible_setup=True,
    )


def buttercup_transfer2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox():
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
            {"override /data_train": "mock_video"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    fps=30,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    fps=30,
                ),
            ),
        ),
    )


MULTI_VIEW_LOADING_KEYS_V3 = [
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
    "umt5_xxl_0",
    "umt5_xxl_1",
    "umt5_xxl_2",
    "umt5_xxl_3",
    "umt5_xxl_4",
    "umt5_xxl_5",
    "umt5_xxl_6",
    "metas",
]


def buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario():
    # N.B. the MADS 20250823 dataset contains videos that are at 10FPS, so we cannot use it for 30FPS experiments.
    # Therefore, if we select 29 for both num_video_frames_per_view and num_video_frames_loaded_per_view,
    # this will be effectively a 10FPS dataset.
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250823_720p_29frames_s3"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario",
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V3,
            ),
        ),
    )


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


def buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase48k_mads720pmulticaps29frames_world_scenario_resumefrom21k():
    # N.B the MADS 20250714 dataset contains raw and control videos at 30FPS
    # So we can use the base 10fps experiment setup but change the control keys to `world_scenario_0`, ... `world_scenario_6`
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
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
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase48k_mads720pmulticaps29frames_world_scenario_resumefrom21k",
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V2,
            ),
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res720p_fps30_t8_from16kfps10mv_jointalpamayov2mads720pmulticaps29frames-0/checkpoints/iter_000048000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario-0/checkpoints/iter_000021000",
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k():
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox",
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
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k",
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V2,
            ),
        ),
        model=dict(
            config=dict(
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_from48kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_29frames_nofps-0/checkpoints/iter_000005000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
            ),
        ),
        checkpoint=dict(
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario-0/checkpoints/iter_000021000",
            load_from_object_store=dict(
                enabled=True,
            ),
        ),
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase2p5_mads420pmulticaps121to61frames_world_scenario():
    sample_n_views = 7
    return dict(
        defaults=[
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250710_480p_121framesto61_s3"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase2p5_mads420pmulticaps121to61frames_world_scenario",
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
            lr=8.63e-5,  # 2**(-14.5) = 3.0517578125e-05
            weight_decay=1e-3,
            betas=[0.9, 0.999],
        ),
        scheduler=dict(
            f_max=[0.5],
            f_min=[0.2],
            warm_up_steps=[100],
            cycle_lengths=[100_000],
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                hint_keys="hdmap_bbox",
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=2,  # i2w or v2v
                condition_locations=["first_random_n"],
                conditional_frames_probs={0: 0.5, 1: 0.25, 2: 0.25},
                state_t=16,
                online_text_embeddings_as_dict=False,  # For backward compatibility with old experiments
                fsdp_shard_size=8,
                resolution="480p",
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="reweighting",
                train_time_distribution="logitnormal",
                net=dict(
                    timestep_scale=0.001,
                    use_wan_fp32_strategy=True,
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=16,
                    n_cameras_emb=7,
                    vace_has_mask=False,
                    use_input_hint_block=True,
                    condition_strategy="spaced",
                    vace_block_every_n=7,  # 4 layers
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=16.0 / 24.0,
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                ),
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,  # (TODO: hanzim): check
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
                    compile_encode=False,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_mv_7views_res480p_fps30_t16_from16kfps10mv_jointalpamayov2mads480pmulticaps61frames-0/checkpoints/iter_000041000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            )
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
                grad_clip=dict(
                    clip_norm=0.1,
                ),
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1_000,
                    is_x0=False,
                    is_ema=False,
                    num_sampling_step=35,
                    guidance=[7],
                    fps=15,
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
                    fps=15,
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
            dataset=dict(
                embedding_type=None,
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V2,
            ),
        ),
        upload_reproducible_setup=True,
    )


def buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_from5knofps_mads420pmulticaps61frames_world_scenario():
    # N.B. the MADS 20250823 dataset contains videos that are at 10FPS, so we cannot use it for 30FPS experiments.
    # Therefore, if we select 61 for both num_video_frames_per_view and num_video_frames_loaded_per_view,
    # this will be effectively a 10FPS dataset.
    sample_n_views = 7
    return dict(
        defaults=[
            "/experiment/buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase2p5_mads420pmulticaps121to61frames_world_scenario",
            {"override /data_train": "video_only_cosmos_transfer2_av_mads_mv_20250823_480p_61frames_s3"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "adamw"},
            {
                "override /callbacks": [
                    "basic",
                    "viz_online_sampling",
                    "wandb",
                    "cluster_speed",
                    "load_base_model_callbacks",
                ]
            },
            {"override /checkpoint": "s3"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_from5knofps_mads420pmulticaps61frames_world_scenario",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="cosmos_transfer2_multiview/cosmos2_mv/buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase2p5_mads420pmulticaps121to61frames_world_scenario-0/checkpoints/iter_000021500",
            load_from_object_store=dict(
                enabled=True,
            ),
            save_to_object_store=dict(
                enabled=True,
            ),
            load_training_state=False,
            strict_resume=False,
        ),
        model=dict(
            config=dict(
                net=dict(
                    rope_enable_fps_modulation=False,
                ),
                base_load_from=dict(
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res480p_fps30_t16_from41kfps30mv_condprobs0442_joint_alpamayo1capnoviewprefix_allcapsviewprefix_61frames_nofps-0/checkpoints/iter_000009000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
            )
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=L(EveryNDrawSampleMultiviewVideo)(
                    fps=10,
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    fps=10,
                ),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                dataset_loading_keys=MULTI_VIEW_LOADING_KEYS_V3,
            ),
        ),
    )


experiments = [
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_frompred2madsreason7brffixdistmatch22k_cond02_hdmapbbox_highsigma_spaced_layer4_mlp(),
    buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox(),
    buttercup_transfer2p5_2b_mv_7views_res720p_fps30_t8_frombase2p5_mads720pmulticaps29frames_hdmapbbox(),
    buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase2p5_mads720pmulticaps29frames_world_scenario(),
    buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase48k_mads720pmulticaps29frames_world_scenario_resumefrom21k(),
    buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_frombase5knofps_mads720pmulticaps29frames_world_scenario_resumefrom21k(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps15_t8_frombase2p5_mads420pmulticaps121to61frames_world_scenario(),
    buttercup_transfer2p5_2b_mv_7views_res480p_fps10_t8_from5knofps_mads420pmulticaps61frames_world_scenario(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
