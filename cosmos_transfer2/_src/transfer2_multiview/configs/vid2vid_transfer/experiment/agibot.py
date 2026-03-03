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
from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

BASE_CKPT_720_T24_CR1PT1_2B_AGIBOT_MULTIVIEW = dict(
    load_path="bucket/cosmos_predict2_multiview/cosmos2p5_mv/predict2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2vfromtni2v18k5_allviews_lr3e5-0/checkpoints/iter_000044000",
    credentials="credentials/s3_checkpoint.secret",
)

_TRAINER_DEBUG_CONFIG = dict(
    max_iter=5000,
    logging_iter=100,
    callbacks=dict(
        every_n_sample_reg=dict(
            every_n=10,
            guidance=[3],
            num_cond_frames=[0, 1],
            fps=10,
            n_viz_sample=1,
            num_sampling_step=35,
            run_at_start=True,
        ),
        every_n_sample_ema=dict(
            every_n=10,
            guidance=[3],
            num_cond_frames=[0, 1],
            fps=10,
            n_viz_sample=1,
            num_sampling_step=35,
            run_at_start=False,
        ),
    ),
)


def build_debug_runs(job):
    w_resume = dict(
        defaults=[
            f"/experiment/{job['job']['name']}",
            "_self_",
        ],
        job=dict(group=job["job"]["group"] + "_debug", name=f"{job['job']['name']}_w_resume"),
        trainer=_TRAINER_DEBUG_CONFIG,
    )
    return [w_resume]


"""
PYTHONPATH=. torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py --dryrun -- experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth_debug"

PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth_debug" job.wandb_mode=disabled
"""
control_input_type = "depth"
transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /data_train": "s3_multiview_train_agibot_720p_nocamera"},
            {"override /conditioner": "video_prediction_multiview_control_conditioner_per_view_dropout"},
            {"override /model": "fsdp_rectified_flow_multiview_control"},
            {"override /net": "cosmos_v1_2B_multiview_crossview_control"},
            {"override /ckpt_type": "dcp"},
            {"override /callbacks": ["basic", "wandb", "cluster_speed", "viz_online_sampling"]},
            "_self_",
        ],
        job=dict(
            group="transfer2p5_mv",
            name=f"transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_{control_input_type}",
        ),
        optimizer=dict(
            lr=5e-5,
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[1000],
            cycle_lengths=[400000],
        ),
        checkpoint=dict(
            save_iter=1000,
            load_from_object_store=dict(
                enabled=True,
            ),
            save_to_object_store=dict(
                enabled=True,
            ),
        ),
        model_parallel=dict(
            context_parallel_size=8,
        ),
        model=dict(
            config=dict(
                hint_keys=control_input_type,
                resolution="720p",
                min_num_conditional_frames=0,
                max_num_conditional_frames=1,
                min_num_conditional_frames_per_view=0,
                max_num_conditional_frames_per_view=1,
                condition_locations=["first_random_n"],
                conditional_frames_probs={0: 0.5, 1: 0.5, 2: 0.0},
                state_t=24,
                train_time_weight="uniform",
                online_text_embeddings_as_dict=False,
                train_sample_views_range=[3, 3],
                base_load_from=BASE_CKPT_720_T24_CR1PT1_2B_AGIBOT_MULTIVIEW,
                net=dict(
                    crossattn_emb_channels=1024,
                    crossattn_proj_in_channels=100352,
                    concat_view_embedding=False,
                    adaln_view_embedding=True,
                    state_t="${model.config.state_t}",
                    n_cameras_emb=3,
                    rope_enable_fps_modulation=False,
                    use_crossattn_projection=True,
                    rope_h_extrapolation_ratio=3.0,  # 720p
                    rope_w_extrapolation_ratio=3.0,  # 720p
                    rope_t_extrapolation_ratio=1.0,
                    timestep_scale=0.001,  # important for rectified flow
                    sac_config=dict(
                        mode="predict2_2b_720_aggressive",
                    ),
                    enable_cross_view_attn=True,
                    cross_view_attn_map_str={
                        "head_color": ["hand_left", "hand_right"],
                        "hand_left": ["head_color", "hand_right"],
                        "hand_right": ["head_color", "hand_left"],
                    },
                    camera_to_view_id={
                        "head_color": 0,
                        "hand_left": 1,
                        "hand_right": 2,
                    },
                    use_wan_fp32_strategy=True,
                    # transfer related configs
                    use_input_hint_block=False,
                    vace_has_mask=False,
                    condition_strategy="spaced",
                    vace_block_every_n=7,
                ),
                tokenizer=dict(
                    temporal_window=16,
                ),
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                ),
                fsdp_shard_size=8,
            ),
        ),
        trainer=dict(
            max_iter=100000,
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
                    every_n=1000,
                    do_x0_prediction=False,
                    is_ema=False,
                    num_sampling_step=35,
                    guidance=[0, 3, 5],
                    num_cond_frames=[0, 1],
                    fps=10,
                    ctrl_hint_keys=[f"control_input_{control_input_type}"],
                ),
                every_n_sample_ema=L(EveryNDrawSampleMultiviewVideo)(
                    every_n=1000,
                    do_x0_prediction=False,
                    is_ema=True,
                    num_sampling_step=35,
                    guidance=[0, 3, 5],
                    num_cond_frames=[0, 1],
                    fps=10,
                    ctrl_hint_keys=[f"control_input_{control_input_type}"],
                ),
            ),
            straggler_detection=dict(enabled=True),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type=control_input_type,
            ),
        ),
        upload_reproducible_setup=True,
    ),
    flags={"allow_objects": True},
)


"""
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment="transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg_debug" job.wandb_mode=disabled
"""
control_input_type = "seg"
transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth",
            "_self_",
        ],
        job=dict(
            group="transfer2p5_mv",
            name=f"transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_{control_input_type}",
        ),
        model=dict(
            config=dict(
                hint_keys=control_input_type,
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
                every_n_sample_ema=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type=control_input_type,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

control_input_type = "edge"
transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth",
            "_self_",
        ],
        job=dict(
            group="transfer2p5_mv",
            name=f"transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_{control_input_type}",
        ),
        model=dict(
            config=dict(
                hint_keys=control_input_type,
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
                every_n_sample_ema=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type=control_input_type,
            ),
        ),
    ),
    flags={"allow_objects": True},
)

control_input_type = "vis"
transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis: LazyDict = LazyDict(
    dict(
        defaults=[
            "/experiment/transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth",
            "_self_",
        ],
        job=dict(
            group="transfer2p5_mv",
            name=f"transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_{control_input_type}",
        ),
        model=dict(
            config=dict(
                hint_keys=control_input_type,
            ),
        ),
        trainer=dict(
            callbacks=dict(
                every_n_sample_reg=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
                every_n_sample_ema=dict(ctrl_hint_keys=[f"control_input_{control_input_type}"]),
            ),
        ),
        dataloader_train=dict(
            dataset=dict(
                control_input_type=control_input_type,
            ),
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()
for _item, _item_w_resume in [
    [
        transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth,
        *build_debug_runs(transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_depth),
    ],
    [
        transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg,
        *build_debug_runs(transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_seg),
    ],
    [
        transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge,
        *build_debug_runs(transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_edge),
    ],
    [
        transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis,
        *build_debug_runs(transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_vis),
    ],
]:
    cs.store(group="experiment", package="_global_", name=_item["job"]["name"], node=_item)
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}_debug", node=_item_w_resume)
