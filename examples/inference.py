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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse

from cosmos_transfer2.config import get_params_from_json
from cosmos_transfer2.control2world import Control2WorldInference


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the mulitcontrol inference script."""
    parser = argparse.ArgumentParser(description="Transfer2.5 inference script")
    parser.add_argument("--params_file", type=str, required=True)
    parser.add_argument("--num_gpus", type=int, default=1, required=False)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if os.path.exists(args.params_file):
        params = get_params_from_json(args.params_file)
    else:
        raise ValueError(f"Params file {args.params_file} does not exist")

    pipe = Control2WorldInference(
        num_gpus=args.num_gpus,
        hint_key=params.hint_key,
        disable_guardrails=params.disable_guardrails,
        offload_guardrail_models=params.offload_guardrail_models,
    )
    pipe.infer(params)


if __name__ == "__main__":
    main()
