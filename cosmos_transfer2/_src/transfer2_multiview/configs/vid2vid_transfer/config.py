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

from typing import Any, List

import attrs

import cosmos_transfer2._src.transfer2_multiview.datasets.augmentor_provider  # noqa: F401
from cosmos_transfer2._src.imaginaire import config
from cosmos_transfer2._src.imaginaire.flags import INTERNAL
from cosmos_transfer2._src.imaginaire.trainer import ImaginaireTrainer as Trainer
from cosmos_transfer2._src.imaginaire.utils.config_helper import import_all_modules_from_package
from cosmos_transfer2._src.predict2.configs.common.defaults.checkpoint import register_checkpoint
from cosmos_transfer2._src.predict2.configs.common.defaults.ckpt_type import register_ckpt_type
from cosmos_transfer2._src.predict2.configs.common.defaults.ema import register_ema
from cosmos_transfer2._src.predict2.configs.common.defaults.optimizer import register_optimizer
from cosmos_transfer2._src.predict2.configs.common.defaults.scheduler import register_scheduler
from cosmos_transfer2._src.predict2.configs.common.defaults.tokenizer import register_tokenizer
from cosmos_transfer2._src.predict2_multiview.configs.vid2vid.defaults.dataloader import (
    register_alpamayo_dataloader,
    register_alpamayo_mads_joint_dataloaders,
)
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.defaults.callbacks import register_callbacks
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.conditioner import register_conditioner
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.data import register_data_ctrlnet
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.dataloader import (
    register_training_and_val_data,
)
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.model import register_model
from cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.defaults.net import register_net


@attrs.define(slots=False)
class Config(config.Config):
    # default config groups that will be used unless overwritten
    # see config groups in registry.py
    defaults: List[Any] = attrs.field(
        factory=lambda: [
            "_self_",
            {"data_train": "mock"},
            {"data_val": "mock"},
            {"optimizer": "fusedadamw"},
            {"scheduler": "lambdalinear"},
            {"model": "ddp_multiview_control"},
            {"callbacks": "basic"},
            {"net": None},
            {"conditioner": "video_prediction_multiview_control_conditioner"},
            {"ema": "power"},
            {"tokenizer": "wan2pt1_tokenizer"},
            {"checkpoint": "s3"},
            {"ckpt_type": "dummy"},
            # the list is with order, we need global experiment to be the last one
            {"experiment": None},
        ]
    )


def make_config() -> Config:
    c = Config(
        model=None,
        optimizer=None,
        scheduler=None,
        dataloader_train=None,
        dataloader_val=None,
    )

    # Specifying values through instances of attrs
    c.job.project = "cosmos_transfer2_multiview"  # this decides the wandb project name
    c.job.group = "debug"
    c.job.name = "delete_${now:%Y-%m-%d}_${now:%H-%M-%S}"

    c.trainer.type = Trainer
    c.trainer.straggler_detection.enabled = False
    c.trainer.max_iter = 400_000
    c.trainer.logging_iter = 100
    c.trainer.validation_iter = 100
    c.trainer.run_validation = False
    c.trainer.callbacks = None
    register_checkpoint()
    register_ckpt_type()
    register_ema()
    register_callbacks()
    register_optimizer()
    register_scheduler()
    register_tokenizer()
    register_training_and_val_data()
    register_data_ctrlnet()
    register_alpamayo_mads_joint_dataloaders()
    register_alpamayo_dataloader()
    register_conditioner()
    register_model()
    register_net()
    import_all_modules_from_package(
        "cosmos_transfer2._src.transfer2_multiview.configs.vid2vid_transfer.experiment.alpamayo", reload=True
    )
    if not INTERNAL:
        import_all_modules_from_package("cosmos_transfer2.experiments", reload=True)
    return c
