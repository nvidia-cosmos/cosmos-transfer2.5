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

import json
import os
from pathlib import Path

from cosmos_gradio.deployment_env import DeploymentEnv
from cosmos_gradio.model_ipc.model_server import ModelServer

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2.config import InferenceArguments, SetupArguments
from cosmos_transfer2.gradio.sample_data import sample_request_edge, sample_request_mv

global_env = DeploymentEnv()


def test_transfer_args():
    log.info(json.dumps(SetupArguments.model_json_schema(), indent=4))
    log.info(json.dumps(InferenceArguments.model_json_schema(), indent=4))
    params = InferenceArguments(**sample_request_edge)
    log.info(json.dumps(params.model_dump(mode="json"), indent=4))


def test_transfer(model_name, params):
    from cosmos_transfer2.gradio.control2world_worker import Control2World_Worker

    params = InferenceArguments(**params)
    log.info(f"params: {json.dumps(params.model_dump(mode='json'), indent=4)}")

    pipeline = Control2World_Worker(num_gpus=1)

    params = params.model_dump(mode="json")
    params["output_dir"] = f"outputs/transfer2/{model_name}"
    pipeline.infer(params)


def test_multiview_args():
    from cosmos_transfer2.multiview_config import MultiviewInferenceArguments, MultiviewSetupArguments

    log.info(json.dumps(MultiviewSetupArguments.model_json_schema(), indent=4))
    params = MultiviewSetupArguments(model="auto/multiview", output_dir=Path("outputs"), disable_guardrails=True)
    log.info(json.dumps(params.model_dump(mode="json"), indent=4))

    log.info(json.dumps(MultiviewInferenceArguments.model_json_schema(), indent=4))
    params = MultiviewInferenceArguments(**sample_request_mv)
    log.info(json.dumps(params.model_dump(mode="json"), indent=4))


def test_transfer_mv():
    from cosmos_transfer2.multiview_config import MultiviewInferenceArguments

    params = MultiviewInferenceArguments(**sample_request_mv)

    # must run on 8 GPUs
    with ModelServer(
        num_gpus=8, factory_module="cosmos_transfer2.gradio.gradio_bootstrapper", factory_function="create_multiview"
    ) as pipeline:
        params = params.model_dump(mode="json")
        params["output_dir"] = "outputs/transfer2/multiview/"
        pipeline.infer(params)


# Note that multiview requires 8 GPUs and cannot be tested w/o torchrun
if __name__ == "__main__":
    log.info(f"test_worker current dir={os.getcwd()}")
    log.info(f"global_env: {global_env}")

    if global_env.model_name == "edge":
        test_transfer_args()
        test_transfer("edge", sample_request_edge)
    elif global_env.model_name == "multiview":
        test_multiview_args()
        test_transfer_mv()
