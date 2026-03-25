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

"""
Configs for DMD2 distillation of HDMap-conditioned I2V models (Closed Loop).

This implements distillation for the hdmap-conditioned I2V bidirectional model
trained on MADS-large dataset.
"""

import math

from hydra.core.config_store import ConfigStore  # type: ignore[import]

from cosmos_transfer2._src.imaginaire.lazy_config import LazyDict
from cosmos_transfer2._src.interactive.configs.registry_defaults.teacher_model_paths import (
    HDMAP_CONDITIONED_TEACHER_CKPT_2B_RELEASE,
)
from cosmos_transfer2._src.interactive.utils.config_helper import deep_update_config_dict
from cosmos_transfer2._src.predict2.text_encoders.text_encoder import EmbeddingConcatStrategy


def make_experiment(
    name: str,
    model: str = "fsdp_dmd2_model_trigflow",
    net: str = "cosmos_v1_2B_hdmap",
    net_teacher: str = "cosmos_v1_2B_hdmap_teacher",
    net_fake_score: str = "cosmos_v1_2B_hdmap_fake_score",
    net_discriminator_head: str | None = None,
    conditioner: str = "video_prediction_multiview_causal_conditioner_per_view_dropout_hdmap",
    condition_postprocessor: str | None = "hdmap_i2v_condition_postprocessor",
    resolution: str = "480",
    cp_size: int = 1,
    fsdp_size: int = 8,
    overrides: dict | None = None,
) -> LazyDict:
    """
    Create a DMD2 distillation experiment for HDMap-conditioned I2V model.

    The default net architecture is consistent with the teacher_cosmos2_2B_res480p_10fps_hdmap_av
    as defined in projects/cosmos/sil/causal_multiview/configs/causal_cosmos2/experiment/hdmap.py

    Here we add the distillation-related configs to that teacher model config.
    """
    defaults = [
        {"/net_teacher@model.config.net_teacher": net_teacher},
        {"/net_fake_score@model.config.net_fake_score": net_fake_score},
        {"/net_discriminator_head@model.config.net_discriminator_head": net_discriminator_head},
        {"override /data_train": "video_mads_1m_0131_pdx_480p_10fps_93frames_7views"},
        {"override /conditioner": conditioner},
        {"override /condition_postprocessor": condition_postprocessor},
        {"override /ckpt_type": "dcp_distill"},
        {"override /checkpoint": "s3"},
        {"override /tokenizer": "wan2pt1_tokenizer"},
        {"override /optimizer": "fusedadamw"},
        {
            "override /callbacks": [
                "basic",
                "wandb",
                "cluster_speed",
                "viz_online_sampling_distilled",
            ]
        },
        {"override /model": model},
        {"override /net": net},
        "_self_",
    ]
    node = dict(
        defaults=defaults,
        job=dict(group="cosmos3_interactive", name=name),
        model_parallel=dict(context_parallel_size=cp_size),
        checkpoint=dict(
            save_iter=500,
            save_to_object_store=dict(
                enabled=True,
            ),
            load_from_object_store=dict(
                enabled=True,
            ),
            load_path="",
            load_training_state=False,
            strict_resume=True,
        ),
        optimizer=dict(
            lr=1e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[100],
            cycle_lengths=[400_000],
        ),
        trainer=dict(
            max_iter=30_000,
            logging_iter=10,
            callbacks=dict(
                iter_speed=dict(hit_thres=100),
                grad_clip=dict(
                    clip_norm=1.0,
                ),
                every_n_sample_reg=dict(
                    every_n=250,
                    is_image=False,
                    num_samples_per_prompt=3,
                ),
                every_n_sample_ema=dict(
                    every_n=250,
                    is_image=False,
                    num_samples_per_prompt=3,
                ),
            ),
        ),
        model=dict(
            config=dict(
                use_neg_prompt_str=True,
                neg_prompt_str="The video captures a series of frames showing ugly scenes, static with no motion, motion blur, over-saturation, shaky footage, low resolution, grainy texture, pixelated images, poorly lit areas, underexposed and overexposed scenes, poor color balance, washed out colors, choppy sequences, jerky movements, low frame rate, artifacting, color banding, unnatural transitions, outdated special effects, fake elements, unconvincing visuals, poorly edited content, jump cuts, visual noise, and flickering. Overall, the video is of poor quality.",
                multiply_noise_by_video_len=True,
                use_clean_cond_timesteps=False,
                ema=dict(
                    enabled=False,
                ),
                conditional_frames_probs={0: 0.0, 1: 1.0},  # Always use 1 conditional frame for I2V
                conditioner=dict(
                    use_video_condition=dict(
                        dropout_rate=0.0,
                    ),
                    text=dict(
                        dropout_rate=0.2,
                        use_empty_string=False,
                    ),
                ),
                condition_postprocessor=dict(
                    preset_hint_keys=["control_input_hdmap_bbox"],
                    hdmap_process_method="vae_encoding",
                    hdmap_selection_mode="all",
                ),
                fsdp_shard_size=fsdp_size,
                grad_clip=True,
                load_teacher_weights=True,
                intermediate_feature_ids=None,
                loss_scale_GAN_discriminator=1.0,
                loss_scale_GAN_generator=1.0,
                loss_scale_fake_score=1.0,
                loss_scale_sid=1.0,  # dmd2 sid loss
                max_num_conditional_frames=1,
                min_num_conditional_frames=1,
                net=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="block_wise",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    use_wan_fp32_strategy=True,
                    additional_concat_ch=16,
                    additional_init_method="random_init",
                ),
                net_fake_score=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="block_wise",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    additional_concat_ch=16,
                    additional_init_method="random_init",
                ),
                net_teacher=dict(
                    rope_enable_fps_modulation=False,
                    rope_h_extrapolation_ratio=2.0,
                    rope_w_extrapolation_ratio=2.0,
                    rope_t_extrapolation_ratio=24.0 / 24,
                    sac_config=dict(
                        mode="block_wise",
                    ),
                    use_crossattn_projection=True,
                    crossattn_proj_in_channels=100352,
                    crossattn_emb_channels=1024,
                    use_wan_fp32_strategy=True,
                    additional_concat_ch=16,
                    additional_init_method="random_init",
                ),
                optimizer_discriminator_config=dict(
                    lr=2e-7,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                optimizer_fake_score_config=dict(
                    lr=2e-7,
                    weight_decay=0.01,
                    betas=(0.9, 0.999),
                ),
                rectified_flow_loss_weight_uniform=False,
                resolution=resolution,
                scaling="rectified_flow",
                sde=dict(
                    p_mean=-0.8,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                sde_D=dict(
                    p_mean=0.0,
                    p_std=1.6,
                    sigma_max=80,
                    sigma_min=0.0002,
                ),
                selected_sampling_time=[math.pi / 2, math.atan(15), math.atan(5), math.atan(5 / 3)],
                sigma_conditional=0.0001,
                sigma_data=1.0,
                state_t=24,
                student_update_freq=5,
                warmup_steps=1,
                teacher_load_from=HDMAP_CONDITIONED_TEACHER_CKPT_2B_RELEASE,
                teacher_guidance=4,
                text_encoder_class="reason1p1_7B",
                text_encoder_config=dict(
                    ckpt_path="s3://bucket/cosmos_reasoning1/sft_exp700/sft_exp721-1_qwen7b_tl_721_5vs5_s3_balanced_n32_resume_16k/checkpoints/iter_000016000/model/",
                    embedding_concat_strategy=str(EmbeddingConcatStrategy.FULL_CONCAT),
                    compute_online=True,
                    s3_credential_path="credentials/s3_checkpoint.secret",
                ),
                timestep_shift=5,
                # HDMap-specific settings
                # noise_scheme="consistent_noise",
                # num_frame_per_block=24,  # 24 frames per block means only 1 block in total
                # max_latent_frames_per_gpu=24,
                # denoise_replace_gt_frames=True,
                # shift=5,
                # use_dynamic_shift=False,
                # train_time_weight="uniform",
                # split_cp_in_model=False,
            ),
        ),
        dataloader_train=dict(
            num_workers=4,
            augmentation_config=dict(
                # resolution_hw=(480, 832),  # Will be set from dataset config
                # fps_downsample_factor=1,  # 30fps
                # num_video_frames=93,
                camera_keys=[
                    "camera_front_wide_120fov",
                ],
            ),
        ),
        upload_reproducible_setup=True,
    )
    if overrides:
        deep_update_config_dict(node, overrides)
    return LazyDict(node, flags={"allow_objects": True})


####################################
# Create and register experiments #
####################################
dmd2_trigflow_distill_cosmos_cl_2B_hdmap_i2v = make_experiment(
    name="dmd2_trigflow_distill_cosmos_cl_2B_hdmap_i2v",
    overrides=dict(
        model=dict(
            config=dict(
                teacher_load_from=HDMAP_CONDITIONED_TEACHER_CKPT_2B_RELEASE,
            ),
        ),
        trainer=dict(
            callbacks=dict(
                # Disable validation prompt sampling for multiview models
                # The callback's get_sample_batch() doesn't provide multiview-specific fields
                every_n_sample_reg=dict(do_sample_val_prompts=False),
                every_n_sample_ema=dict(do_sample_val_prompts=False),
            ),
        ),
    ),
)


cs = ConfigStore.instance()
"""
Example training command:
PYTHONPATH=. torchrun --nproc_per_node=8 --master_port=12340 -m scripts.train \
  --config=cosmos_transfer2/_src/interactive/configs/registry_cl.py -- \
  experiment=dmd2_trigflow_distill_cosmos_cl_2B_hdmap_i2v \
  job.group=cosmos2_interactive \
  job.name=amirzaei_dmd2_trigflow_distill_cl_2B_hdmap_i2v \
  model.config.use_neg_prompt_str=True
"""
for _item in [
    dmd2_trigflow_distill_cosmos_cl_2B_hdmap_i2v,
]:
    cs.store(group="experiment", package="_global_", name=f"{_item['job']['name']}", node=_item)
