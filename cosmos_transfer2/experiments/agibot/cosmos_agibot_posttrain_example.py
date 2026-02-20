# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Post-training experiments for Agibot 3-view (head_color, hand_left, hand_right) with local data.
# Requires COSMOS_EXPERIMENTAL_CHECKPOINTS=1 for base checkpoint access.
#
# Common CLI overrides (after --): dataloader_train.dataset.dataset_dir, trainer.max_iter, optimizer.lr, etc.
# Checkpoint is resolved via RobotMultiviewControlAgibotSetupArguments.model_key_for_control_type (same as inference).

import os
from typing import Literal

from hydra.core.config_store import ConfigStore

from cosmos_transfer2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS
from cosmos_transfer2.config import MODEL_CHECKPOINTS
from cosmos_transfer2.robot_multiview_control_agibot_config import (
    RobotMultiviewControlAgibotSetupArguments,
)


def _agibot_checkpoints_available() -> bool:
    """True if Agibot checkpoints are in MODEL_CHECKPOINTS (requires COSMOS_EXPERIMENTAL_CHECKPOINTS=1)."""
    model_key = RobotMultiviewControlAgibotSetupArguments.model_key_for_control_type("edge")
    return model_key in MODEL_CHECKPOINTS


AgibotControlType = Literal["depth", "edge", "seg", "vis"]


def _get_agibot_checkpoint(control_type: AgibotControlType):
    """Resolve Agibot checkpoint for the given control type. Requires COSMOS_EXPERIMENTAL_CHECKPOINTS=1."""
    model_key = RobotMultiviewControlAgibotSetupArguments.model_key_for_control_type(control_type)
    if model_key not in MODEL_CHECKPOINTS:
        raise KeyError(
            f"Agibot checkpoint for control_type={control_type!r} not found in MODEL_CHECKPOINTS. "
            "Set COSMOS_EXPERIMENTAL_CHECKPOINTS=1 so base checkpoints are registered."
        )
    return MODEL_CHECKPOINTS[model_key]


# Base experiment names from cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/experiment/agibot.py
# Training params below mirror the transfer2.5 finetune stage (agibot depth) but scaled for short local post-train.
_BASE_EXPERIMENT = "transfer2p5_2b_mv3_res720p_t24_frombase2p5_agibot_captionprefix_tni2v_{control_type}"


def _make_agibot_posttrain_experiment(control_type: AgibotControlType) -> dict:
    checkpoint = _get_agibot_checkpoint(control_type)
    use_s3 = not EXPERIMENTAL_CHECKPOINTS
    return dict(
        defaults=[
            f"/experiment/{_BASE_EXPERIMENT.format(control_type=control_type)}",
            "_self_",
            {"override /data_train": f"example_agibot_multiview_train_data_{control_type}"},
        ],
        job=dict(
            project="cosmos_transfer_v2p5",
            group="agibot_posttrain",
            name=f"transfer2_agibot_posttrain_{control_type}_example",
        ),
        checkpoint=dict(
            save_iter=500,
            load_path=checkpoint.s3.uri,
            load_training_state=False,
            strict_resume=False,
            load_from_object_store=dict(enabled=use_s3),
            save_to_object_store=dict(enabled=False),
        ),
        upload_reproducible_setup=use_s3,
        model=dict(config=dict(base_load_from=None)),
        dataloader_train=dict(
            dataset=dict(
                control_input_type=control_type,
            ),
        ),
        optimizer=dict(lr=1e-4),
        scheduler=dict(
            f_max=[0.99],
            f_min=[0.4],
            warm_up_steps=[500],
            cycle_lengths=[30_000],  # one cycle = full run for default max_iter=30k
        ),
        trainer=dict(
            logging_iter=100,
            max_iter=10_000,
            straggler_detection=dict(enabled=False),  # Avoid internal straggler package for local post-train
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(hit_thres=50, every_n=100, save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(every_n=200, save_s3=False),
                every_n_sample_ema=dict(every_n=200, save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
                frame_loss_log=dict(save_s3=False),
            ),
        ),
        model_parallel=dict(context_parallel_size=int(os.environ.get("WORLD_SIZE", "1"))),
    )


# Only register Agibot experiments when checkpoints are available (COSMOS_EXPERIMENTAL_CHECKPOINTS=1).
# Otherwise importing this module would raise KeyError and break inference/training that does not use Agibot.
if _agibot_checkpoints_available():
    transfer2_agibot_posttrain_edge_example = _make_agibot_posttrain_experiment("edge")
    transfer2_agibot_posttrain_depth_example = _make_agibot_posttrain_experiment("depth")
    transfer2_agibot_posttrain_seg_example = _make_agibot_posttrain_experiment("seg")
    transfer2_agibot_posttrain_vis_example = _make_agibot_posttrain_experiment("vis")

    cs = ConfigStore.instance()
    for _item in [
        transfer2_agibot_posttrain_edge_example,
        transfer2_agibot_posttrain_depth_example,
        transfer2_agibot_posttrain_seg_example,
        transfer2_agibot_posttrain_vis_example,
    ]:
        name = _item["job"]["name"]
        cs.store(group="experiment", package="_global_", name=name, node=_item)
