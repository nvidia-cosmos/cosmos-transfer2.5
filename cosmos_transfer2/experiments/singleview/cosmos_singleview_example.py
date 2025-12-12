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
Post-training configurations for Cosmos-Transfer2 with local single-view datasets.

These configurations demonstrate how to post-train Transfer2 models on your own data.
They inherit from the base 2B control experiment and override dataset and checkpoint settings.

Example usage:
    torchrun --nproc_per_node=8 -m scripts.train \\
        --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \\
        -- experiment=transfer2_singleview_posttrain_edge_example \\
        dataloader_train.dataset.dataset_dir=/path/to/your/dataset
"""

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.config import DEFAULT_BASE_EXPERIMENT, MODEL_CHECKPOINTS, ModelKey, ModelVariant

# Default model keys and checkpoints for each control type
DEFAULT_EDGE_MODEL_KEY = ModelKey(variant=ModelVariant.EDGE)
DEFAULT_DEPTH_MODEL_KEY = ModelKey(variant=ModelVariant.DEPTH)
DEFAULT_SEG_MODEL_KEY = ModelKey(variant=ModelVariant.SEG)
DEFAULT_VIS_MODEL_KEY = ModelKey(variant=ModelVariant.VIS)

EDGE_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_EDGE_MODEL_KEY]
DEPTH_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_DEPTH_MODEL_KEY]
SEG_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_SEG_MODEL_KEY]
VIS_CHECKPOINT = MODEL_CHECKPOINTS[DEFAULT_VIS_MODEL_KEY]


# =============================================================================
# Post-training with Edge Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_edge_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_edge"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_edge_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=EDGE_CHECKPOINT.hf.path,  # pyrefly: ignore  # Use consolidated .pt file from HuggingFace
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="edge",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="edge",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Depth Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_depth_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_depth"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_depth_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=DEPTH_CHECKPOINT.hf.path,  # pyrefly: ignore  # Use consolidated .pt file from HuggingFace
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="depth",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="depth",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Segmentation Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_seg_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_seg"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_seg_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=SEG_CHECKPOINT.hf.path,  # pyrefly: ignore  # Use consolidated .pt file from HuggingFace
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="seg",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="segcolor",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Post-training with Vis (Blur) Control (2B Model)
# =============================================================================

transfer2_singleview_posttrain_vis_example = dict(
    defaults=[
        DEFAULT_BASE_EXPERIMENT,
        {"override /data_train": "example_singleview_train_data_vis"},
    ],
    job=dict(
        project="cosmos_transfer2_posttrain",
        group="local_single_view",
        name="transfer2_singleview_posttrain_vis_example",
    ),
    checkpoint=dict(
        save_iter=1000,
        load_path=VIS_CHECKPOINT.hf.path,  # pyrefly: ignore  # Use consolidated .pt file from HuggingFace
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(enabled=False),
        save_to_object_store=dict(enabled=False),
    ),
    model=dict(
        config=dict(
            hint_keys="vis",
            base_load_from=None,  # Disable base model loading (already loading from checkpoint.load_path)
        ),
    ),
    dataloader_train=dict(
        dataset=dict(
            control_input_type="vis",
        ),
    ),
    trainer=dict(
        max_iter=5000,
        straggler_detection=dict(enabled=False),  # Disable for local training
        callbacks=dict(
            heart_beat=dict(save_s3=False),
            iter_speed=dict(save_s3=False),
            device_monitor=dict(save_s3=False),
            every_n_sample_reg=dict(save_s3=False, every_n=200),
            every_n_sample_ema=dict(save_s3=False, every_n=200),
            wandb=dict(save_s3=False),
            wandb_10x=dict(save_s3=False),
            dataloader_speed=dict(save_s3=False),
            frame_loss_log=dict(save_s3=False),
        ),
    ),
    scheduler=dict(
        cycle_lengths=[5000],
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)


# =============================================================================
# Register all experiments
# =============================================================================

cs = ConfigStore.instance()

for _item in [
    transfer2_singleview_posttrain_edge_example,
    transfer2_singleview_posttrain_depth_example,
    transfer2_singleview_posttrain_seg_example,
    transfer2_singleview_posttrain_vis_example,
]:
    _name: str = _item["job"]["name"]  # pyrefly: ignore
    cs.store(
        group="experiment",
        package="_global_",
        name=_name,
        node=_item,
    )
