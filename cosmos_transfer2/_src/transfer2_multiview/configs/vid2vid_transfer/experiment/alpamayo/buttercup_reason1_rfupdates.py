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

from cosmos_transfer2._src.common.types.embedding_concat_strategy import EmbeddingConcatStrategy
from cosmos_transfer2._src.imaginaire.lazy_config import LazyCall as L
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.conditioner import TextAttrEmptyStringDropout
from cosmos_transfer2._src.reason1.configs.default.model_config_qwen import QwenModelConfig, QwenVisionConfig
from cosmos_transfer2._src.reason1.models.vlm_qwen_omni import QwenVLBaseModel
from cosmos_transfer2._src.reason1.tokenizer.processor import build_tokenizer

"""

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=buttercup_transfer2_2b_mv_7views_res720_fps10_t8_basefromcond01rffix1n13k_mads_reason7b_emptystr_cond01_hdmap_bboxdrop0p2_oldsde_spaced_layer7_mlp_rffix_distmatch_fp32
"""


def buttercup_transfer2_2b_mv_7views_res720_fps10_t8_basefromcond01rffix1n13k_mads_reason7b_emptystr_cond01_hdmap_bboxdrop0p2_oldsde_spaced_layer7_mlp_rffix_distmatch_fp32():
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
            name="buttercup_transfer2_2b_mv_7views_res720_fps10_t8_basefromcond01rffix1n13k_mads_reason7b_emptystr_cond01_hdmap_bboxdrop0p2_oldsde_spaced_layer7_mlp_rffix_distmatch_fp32",
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
            weight_decay=0.001,  # updated from 0.1
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
                use_wan_fp32_strategy=True,  # updated from False
                min_num_conditional_frames_per_view=0,  # t2w
                max_num_conditional_frames_per_view=1,  # i2w
                condition_locations=["first_random_n"],
                state_t=8,
                net=dict(
                    use_wan_fp32_strategy=True,  # updated from False
                    concat_view_embedding=True,
                    view_condition_dim=7,
                    state_t=8,
                    n_cameras_emb=7,
                    vace_has_mask=False,
                    use_input_hint_block=True,
                    condition_strategy="spaced",
                    vace_block_every_n=4,
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
                    load_path="bucket/cosmos_predict2_multiview/cosmos2_mv/buttercup_predict2_2b_vid2vid_mv_7views_res720_fps10_t8_fromPart2Alpamayo2tar25k_alpamayov2_reason7b_emptystr_noviewprefix_1cap_cond01_rffix/checkpoints/iter_000013000",
                    credentials="credentials/s3_checkpoint.secret",
                ),
                sde=dict(
                    p_mean=math.log(5.0),  # updated from 4.0
                    p_std=1.0,  # updated from 1.2
                    sigma_max=200,
                    sigma_min=0.01,
                ),
                high_sigma_ratio=0.0,
                loss_scale=10.0,
                adjust_video_noise=False,  # updated from True
                scaling="rectified_flow",
                rectified_flow_loss_weight_uniform=False,  # this has no impact
                sigma_data=1.0,
                fsdp_shard_size=8,
                resolution="720",
                resize_online=True,
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=L(TextAttrEmptyStringDropout)(
                        input_key="t5_text_embeddings",
                        pos_input_key="text_embeddings",
                        dropout_input_key="dropout_text_embeddings",
                        dropout_rate=0.2,
                    ),
                    control_input_hdmap_bbox=dict(
                        dropout_rate=0.2,
                    ),
                ),
                tokenizer=dict(
                    temporal_window=16,
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


experiments = [
    buttercup_transfer2_2b_mv_7views_res720_fps10_t8_basefromcond01rffix1n13k_mads_reason7b_emptystr_cond01_hdmap_bboxdrop0p2_oldsde_spaced_layer7_mlp_rffix_distmatch_fp32(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
