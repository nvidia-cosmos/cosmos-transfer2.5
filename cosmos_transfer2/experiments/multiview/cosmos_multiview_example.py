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

import os

from hydra.core.config_store import ConfigStore

from cosmos_transfer2.multiview_config import DEFAULT_CHECKPOINT

transfer2_auto_multiview_post_train_example = dict(
    defaults=[
        f"/experiment/{DEFAULT_CHECKPOINT.experiment}",
        {"override /data_train": "example_multiview_train_data_control_input_hdmap"},
    ],
    job=dict(project="cosmos_transfer_v2p5", group="auto_multiview", name="2b_cosmos_multiview_post_train_example"),
    checkpoint=dict(
        save_iter=200,
        load_path=DEFAULT_CHECKPOINT.s3.uri,
        load_training_state=False,
        strict_resume=False,
        load_from_object_store=dict(
            enabled=False,  # Loading from local filesystem, not S3
        ),
        save_to_object_store=dict(
            enabled=False,
        ),
    ),
    model=dict(
        config=dict(
            base_load_from=None,
        ),
    ),
    trainer=dict(
        logging_iter=100,
        max_iter=5_000,
        callbacks=dict(
            heart_beat=dict(
                save_s3=False,
            ),
            iter_speed=dict(
                hit_thres=200,
                save_s3=False,
            ),
            device_monitor=dict(
                save_s3=False,
            ),
            every_n_sample_reg=dict(
                every_n=200,
                save_s3=False,
            ),
            every_n_sample_ema=dict(
                every_n=200,
                save_s3=False,
            ),
            wandb=dict(
                save_s3=False,
            ),
            wandb_10x=dict(
                save_s3=False,
            ),
            dataloader_speed=dict(
                save_s3=False,
            ),
            frame_loss_log=dict(
                save_s3=False,
            ),
        ),
    ),
    model_parallel=dict(
        context_parallel_size=int(os.environ.get("WORLD_SIZE", "1")),
    ),
)

cs = ConfigStore.instance()

# Register the configuration with Hydra ConfigStore
for _item in [
    transfer2_auto_multiview_post_train_example,
]:
    experiment_name = [name.lower() for name, value in globals().items() if value is _item][0]  # noqa: RUF015

    cs.store(
        group="experiment",
        package="_global_",
        name=experiment_name,
        node=_item,
    )
