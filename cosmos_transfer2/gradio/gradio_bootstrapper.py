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
import gc

import torch
from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_app import GradioApp
from cosmos_gradio.gradio_app.gradio_ui import create_gradio_UI
from loguru import logger as log

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.control2world import Control2WorldInference, Control2WorldParams
from cosmos_transfer2.gradio.model_config import Config as ModelConfig
from cosmos_transfer2.multiview2world import MultiviewInference, MultiviewParams


def create_control2world(hint_key: str):
    log.info("Creating predict pipeline and validator")
    global_env = DeploymentEnv()
    pipeline = Control2WorldInference(num_gpus=global_env.num_gpus, hint_key=hint_key)
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_edge():
    return create_control2world(hint_key=["edge"])


def create_vis():
    return create_control2world(hint_key=["vis"])


def create_depth():
    return create_control2world(hint_key=["depth"])


def create_seg():
    return create_control2world(hint_key=["seg"])


def create_multiview():
    log.info("Creating predict pipeline and validator")
    global_env = DeploymentEnv()
    # we cannot hard-code: user needs to create 8-gpu instance and start 8 workers
    assert global_env.num_gpus == 8, "Multiview currently requires 8 GPUs"
    pipeline = MultiviewInference(
        num_gpus=global_env.num_gpus,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def validate_control2world(kwargs):
    params = Control2WorldParams.create(kwargs)
    _ = params.control_modalities  # validate paths
    return params.to_kwargs()


def validate_multiview(kwargs):
    params = MultiviewParams.create(kwargs)
    _ = params.input_and_control_paths  # validate paths
    return params.to_kwargs()


if __name__ == "__main__":
    model_cfg = ModelConfig()
    deploy_cfg = DeploymentEnv()

    log.info(f"Starting Gradio app with deployment config: {deploy_cfg!s}")

    factory_module = {
        "vis": "cosmos_transfer2.gradio.gradio_bootstrapper",
        "depth": "cosmos_transfer2.gradio.gradio_bootstrapper",
        "edge": "cosmos_transfer2.gradio.gradio_bootstrapper",
        "seg": "cosmos_transfer2.gradio.gradio_bootstrapper",
        "multiview": "cosmos_transfer2.gradio.gradio_bootstrapper",
    }

    factory_function = {
        "vis": "create_vis",
        "depth": "create_depth",
        "edge": "create_edge",
        "seg": "create_seg",
        "multiview": "create_multiview",
    }

    validators = {
        "vis": validate_control2world,
        "depth": validate_control2world,
        "edge": validate_control2world,
        "seg": validate_control2world,
        "multiview": validate_multiview,
    }

    app = GradioApp(
        num_gpus=deploy_cfg.num_gpus,
        validator=validators[deploy_cfg.model_name],
        factory_module=factory_module[deploy_cfg.model_name],
        factory_function=factory_function[deploy_cfg.model_name],
        output_dir=deploy_cfg.output_dir,
    )

    interface = create_gradio_UI(
        app.infer,
        header=model_cfg.header[deploy_cfg.model_name],
        default_request=model_cfg.default_request[deploy_cfg.model_name],
        help_text=model_cfg.help_text[deploy_cfg.model_name],
        uploads_dir=deploy_cfg.uploads_dir,
        output_dir=deploy_cfg.output_dir,
        log_file=deploy_cfg.log_file,
    )

    interface.launch(
        server_name="0.0.0.0",
        server_port=8080,
        share=False,
        debug=True,
        max_file_size="500MB",
        allowed_paths=[deploy_cfg.output_dir, deploy_cfg.uploads_dir],
    )
