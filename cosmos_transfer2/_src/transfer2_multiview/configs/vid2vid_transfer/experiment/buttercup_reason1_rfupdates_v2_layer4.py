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
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py --dryrun -- experiment=buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform
"""


def buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform() -> (
    dict
):
    sample_n_views = 7
    text_encoder_ckpt_path = "s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/"
    base_load_path = "bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2p5_2b_7views_res720p_fps30_t8_joint_alpamayo1capviewprefix_allcapsviewprefix_29frames_nofps_uniform_dropoutt0-0/checkpoints/iter_000012000/"
    base_load_credentials = "credentials/s3_checkpoint.secret"

    return dict(
        defaults=[
            {"override /data_train": "video_control_mads_multiview_0823_s3_720p_10fps_29frames_7views"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_control"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
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
            name="buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform",
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
            warm_up_steps=[1000],
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
                online_text_embeddings_as_dict=False,
                fsdp_shard_size=8,
                resolution="720p",
                shift=5,
                use_dynamic_shift=False,
                train_time_weight="uniform",
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
                    rope_enable_fps_modulation=False,
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
                        use_empty_string=False,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
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
                    every_n=2_000,
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
                    every_n=2_000,
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
            augmentation_config=dict(
                single_caption_camera_name="camera_front_wide_120fov",
                add_view_prefix_to_caption=True,
            ),
        ),
        upload_reproducible_setup=True,
    )


experiments = [
    buttercup_transfer2p5_2b_mv_7views_res720p_fps10_t8_fromfinetuned12knofpsuniform_mads720pmulticaps29frames_world_scenario_nofps_uniform(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
