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
from cosmos_transfer2._src.predict2_multiview.callbacks.every_n_draw_sample_multiviewvideo import (
    EveryNDrawSampleMultiviewVideo,
)

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


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_baseline_layer14_sacpred2_b1
"""


def perf_buttercup_baseline_layer14_sacpred2_b1():
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
            name="perf_buttercup_baseline_layer14_sacpred2_b1",
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
                    vace_block_every_n=2,
                    base_n_dense_blocks=-1,
                    control_n_dense_blocks=-1,
                    base_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    base_gna_parameters=None,
                    control_gna_parameters=None,
                    rope_enable_fps_modulation=True,
                    rope_h_extrapolation_ratio=3.0,
                    rope_w_extrapolation_ratio=3.0,
                    rope_t_extrapolation_ratio=8.0 / 24.0,
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
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer4_sacpred2_b1
"""


def perf_buttercup_layer4_sacpred2_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer4_sacpred2_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=7,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_sacpred2_b1
"""


def perf_buttercup_layer1_sacpred2_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_sacpred2_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_mmonlybase_sacpred2control_b1
"""


def perf_buttercup_layer1_mmonlybase_sacpred2control_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_mmonlybase_sacpred2control_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                    base_sac_config=dict(
                        mode="mm_only",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_allsacbase_sacpred2control_b1
"""


def perf_buttercup_layer1_allsacbase_sacpred2control_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_allsacbase_sacpred2control_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                    base_sac_config=dict(
                        mode="all",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_allsacbase_sacpred2control_b1_gradacc2
"""


def perf_buttercup_layer1_allsacbase_sacpred2control_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_allsacbase_sacpred2control_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                    base_sac_config=dict(
                        mode="all",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_allsacbase_sacpred2control_b2
"""


def perf_buttercup_layer1_allsacbase_sacpred2control_b2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_allsacbase_sacpred2control_b2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                    base_sac_config=dict(
                        mode="all",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                ),
            ),
        ),
        dataloader_train=dict(
            batch_size=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_allsacbase_allsaccontrol_b1
"""


def perf_buttercup_layer1_allsacbase_allsaccontrol_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_allsacbase_allsaccontrol_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                    base_sac_config=dict(
                        mode="all",
                    ),
                    control_sac_config=dict(
                        mode="all",
                    ),
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer4_allsacbase_sacpred2control_gnacontrol_b1
"""


def perf_buttercup_layer4_allsacbase_sacpred2control_gnacontrol_b1():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer4_allsacbase_sacpred2control_gnacontrol_b1",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=7,
                    base_sac_config=dict(
                        mode="all",
                    ),
                    control_sac_config=dict(
                        mode="predict2_2b_720",
                    ),
                    control_n_dense_blocks=0,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer1_sacpred2_b1_gradacc2
"""


def perf_buttercup_layer1_sacpred2_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer1_sacpred2_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=28,
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer28_sacpred2_gnacontrol4dense_b1_gradacc2
"""


def perf_buttercup_layer28_sacpred2_gnacontrol4dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer28_sacpred2_gnacontrol4dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=1,
                    control_n_dense_blocks=4,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer28_sacpred2_gnacontrol2dense_b1_gradacc2
"""


def perf_buttercup_layer28_sacpred2_gnacontrol2dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer28_sacpred2_gnacontrol2dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=1,
                    control_n_dense_blocks=2,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer14_sacpred2_gnacontrol4dense_b1_gradacc2
"""


def perf_buttercup_layer14_sacpred2_gnacontrol4dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer14_sacpred2_gnacontrol4dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=2,
                    control_n_dense_blocks=4,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer14_sacpred2_gnacontrol2dense_b1_gradacc2
"""


def perf_buttercup_layer14_sacpred2_gnacontrol2dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer14_sacpred2_gnacontrol2dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=2,
                    control_n_dense_blocks=2,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer7_sacpred2_gnacontrol4dense_b1_gradacc2
"""


def perf_buttercup_layer7_sacpred2_gnacontrol4dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer7_sacpred2_gnacontrol4dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=4,
                    control_n_dense_blocks=4,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer7_sacpred2_gnacontrol3dense_b1_gradacc2
"""


def perf_buttercup_layer7_sacpred2_gnacontrol3dense_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer7_sacpred2_gnacontrol3dense_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=4,
                    control_n_dense_blocks=3,
                    control_gna_parameters={
                        "window_size": (-1, 12, 24),
                        "stride": (1, 4, 8),
                        "base_size": (-1, 44, 80),
                    },
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


"""
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=perf_buttercup_layer7_sacpred2_b1_gradacc2
"""


def perf_buttercup_layer7_sacpred2_b1_gradacc2():
    return dict(
        defaults=[
            "/experiment/perf_buttercup_baseline_layer14_sacpred2_b1",
            "_self_",
        ],
        job=dict(
            group="cosmos2_mv",
            name="perf_buttercup_layer7_sacpred2_b1_gradacc2",
        ),
        model=dict(
            config=dict(
                net=dict(
                    vace_block_every_n=4,
                ),
            ),
        ),
        trainer=dict(
            grad_accum_iter=2,
        ),
    )


experiments = [
    perf_buttercup_baseline_layer14_sacpred2_b1(),
    perf_buttercup_layer4_sacpred2_b1(),
    perf_buttercup_layer1_sacpred2_b1(),
    perf_buttercup_layer1_mmonlybase_sacpred2control_b1(),
    perf_buttercup_layer1_allsacbase_sacpred2control_b1(),
    perf_buttercup_layer1_allsacbase_sacpred2control_b1_gradacc2(),
    perf_buttercup_layer1_allsacbase_sacpred2control_b2(),
    perf_buttercup_layer1_allsacbase_allsaccontrol_b1(),
    perf_buttercup_layer4_allsacbase_sacpred2control_gnacontrol_b1(),
    perf_buttercup_layer1_sacpred2_b1_gradacc2(),
    perf_buttercup_layer28_sacpred2_gnacontrol4dense_b1_gradacc2(),
    perf_buttercup_layer28_sacpred2_gnacontrol2dense_b1_gradacc2(),
    perf_buttercup_layer14_sacpred2_gnacontrol4dense_b1_gradacc2(),
    perf_buttercup_layer14_sacpred2_gnacontrol2dense_b1_gradacc2(),
    perf_buttercup_layer7_sacpred2_gnacontrol4dense_b1_gradacc2(),
    perf_buttercup_layer7_sacpred2_gnacontrol3dense_b1_gradacc2(),
    perf_buttercup_layer7_sacpred2_b1_gradacc2(),
]

cs = ConfigStore.instance()

for _item in experiments:
    cs.store(
        group="experiment",
        package="_global_",
        name=_item["job"]["name"],
        node=_item,
    )
