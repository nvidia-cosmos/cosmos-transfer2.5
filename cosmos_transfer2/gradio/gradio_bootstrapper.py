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
from typing import Literal

import torch
from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.gradio_app.gradio_server import launch_gradio_server
from loguru import logger as log

from cosmos_transfer2.gradio.model_config import Config as ModelConfig


def create_control2world():
    from cosmos_transfer2.gradio.control2world_worker import Control2World_Worker

    global_env = DeploymentEnv()
    log.info(f"Creating control2world pipeline with {global_env=}")
    pipeline = Control2World_Worker(
        model=global_env.model_name,
        num_gpus=global_env.num_gpus,
        disable_guardrails=global_env.disable_guardrails,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_multiview():
    from cosmos_transfer2.gradio.multiview_worker import Multiview_Worker

    global_env = DeploymentEnv()
    log.info(f"Creating multiview pipeline with {global_env=}")
    # we cannot hard-code: user needs to create 8-gpu instance and start 8 workers
    assert global_env.num_gpus == 8, "Multiview currently requires 8 GPUs"
    pipeline = Multiview_Worker(
        num_gpus=global_env.num_gpus,
        disable_guardrails=global_env.disable_guardrails,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def _create_robot_multiview_agibot(control_type: Literal["depth", "edge", "vis", "seg"]):
    from cosmos_transfer2.gradio.robot_multiview_agibot_worker import RobotMultiviewAgibotWorker

    global_env = DeploymentEnv()
    log.info(f"Creating robot multiview agibot pipeline with {global_env=}, control_type={control_type}")
    assert global_env.num_gpus >= 4, "Robot multiview agibot requires minimum 4 GPUs"
    pipeline = RobotMultiviewAgibotWorker(
        num_gpus=global_env.num_gpus,
        control_type=control_type,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def create_robot_multiview_agibot_depth():
    return _create_robot_multiview_agibot("depth")


def create_robot_multiview_agibot_edge():
    return _create_robot_multiview_agibot("edge")


def create_robot_multiview_agibot_vis():
    return _create_robot_multiview_agibot("vis")


def create_robot_multiview_agibot_seg():
    return _create_robot_multiview_agibot("seg")


def create_multiview_many_camera():
    from cosmos_transfer2.gradio.multiview_many_camera_worker import MultiviewManyCamera_Worker

    global_env = DeploymentEnv()
    log.info(f"Creating multiview many-camera pipeline with {global_env=}")
    pipeline = MultiviewManyCamera_Worker(
        num_gpus=global_env.num_gpus,
        disable_guardrails=global_env.disable_guardrails,
    )
    gc.collect()
    torch.cuda.empty_cache()

    return pipeline


def validate_control2world(kwargs):
    from cosmos_transfer2.config import InferenceArguments

    params = InferenceArguments(**kwargs)
    return params.model_dump(mode="json")


def validate_multiview(kwargs):
    from cosmos_transfer2.multiview_config import MultiviewInferenceArguments

    params = MultiviewInferenceArguments(**kwargs)
    return params.model_dump(mode="json")


def validate_robot_multiview_agibot(kwargs):
    from cosmos_transfer2.robot_multiview_control_agibot_config import RobotMultiviewControlAgibotInferenceArguments

    params = RobotMultiviewControlAgibotInferenceArguments(**kwargs)
    return params.model_dump(mode="json")


def validate_multiview_many_camera(kwargs):
    from cosmos_transfer2.plenoptic_config import PlenopticInferenceArguments

    params = PlenopticInferenceArguments(**kwargs)
    return params.model_dump(mode="json")


if __name__ == "__main__":
    model_cfg = ModelConfig()
    deploy_cfg = DeploymentEnv()

    log.info(f"Starting Gradio app with deployment config: {deploy_cfg!s}")

    factory_function = {
        "vis": "create_control2world",
        "depth": "create_control2world",
        "edge": "create_control2world",
        "edge/distilled": "create_control2world",
        "seg": "create_control2world",
        "multicontrol": "create_control2world",
        "multiview": "create_multiview",
        "robot/multiview-agibot-depth": "create_robot_multiview_agibot_depth",
        "robot/multiview-agibot-edge": "create_robot_multiview_agibot_edge",
        "robot/multiview-agibot-vis": "create_robot_multiview_agibot_vis",
        "robot/multiview-agibot-seg": "create_robot_multiview_agibot_seg",
        "robot/multiview-many-camera": "create_multiview_many_camera",
    }

    validators = {
        "vis": validate_control2world,
        "depth": validate_control2world,
        "edge": validate_control2world,
        "edge/distilled": validate_control2world,
        "seg": validate_control2world,
        "multicontrol": validate_control2world,
        "multiview": validate_multiview,
        "robot/multiview-agibot-depth": validate_robot_multiview_agibot,
        "robot/multiview-agibot-edge": validate_robot_multiview_agibot,
        "robot/multiview-agibot-vis": validate_robot_multiview_agibot,
        "robot/multiview-agibot-seg": validate_robot_multiview_agibot,
        "robot/multiview-many-camera": validate_multiview_many_camera,
    }

    launch_gradio_server(
        factory_module="cosmos_transfer2.gradio.gradio_bootstrapper",
        factory_function=factory_function[deploy_cfg.model_name],
        validator=validators[deploy_cfg.model_name],
        num_gpus=deploy_cfg.num_gpus,
        output_dir=deploy_cfg.output_dir,
        uploads_dir=deploy_cfg.uploads_dir,
        log_file=deploy_cfg.log_file,
        default_request=model_cfg.default_request[deploy_cfg.model_name],
        header=model_cfg.header[deploy_cfg.model_name],
        help_text=model_cfg.help_text[deploy_cfg.model_name],
        allowed_paths=deploy_cfg.allowed_paths,
    )
