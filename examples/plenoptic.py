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

"""Plenoptic multiview camera inference script.

This script runs autoregressive multiview video generation from a single input video,
generating novel camera views based on specified camera motion trajectories.

Example usage:
    # Single GPU inference
    python examples/plenoptic.py \
        -i assets/plenoptic_example/sample.json \
        -o outputs/plenoptic/ \
        --base-path assets/plenoptic_example/

    # Multi-GPU inference (8 GPUs)
    torchrun --nproc_per_node=8 examples/plenoptic.py \
        -i assets/plenoptic_example/sample.json \
        -o outputs/plenoptic/ \
        --base-path assets/plenoptic_example/ \
        --context_parallel_size=8
"""

from pathlib import Path
from typing import Annotated

import pydantic
import tyro
from cosmos_oss.init import cleanup_environment, init_environment, init_output_dir

from cosmos_transfer2.config import handle_tyro_exception, is_rank0
from cosmos_transfer2.plenoptic_config import (
    PlenopticInferenceArguments,
    PlenopticInferenceOverrides,
    PlenopticSetupArguments,
)


class Args(pydantic.BaseModel):
    """Command-line arguments for plenoptic multiview inference."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    input_files: Annotated[list[Path], tyro.conf.arg(aliases=("-i",))]
    """Path to the inference parameter file(s) (JSON or JSONL format).
    If multiple files are provided, the model will be loaded once and all samples will be processed sequentially.
    """
    setup: PlenopticSetupArguments
    """Setup arguments for model configuration. These can only be provided via CLI."""
    overrides: PlenopticInferenceOverrides
    """Inference parameter overrides. These can be provided in the input JSON file or via CLI. 
    CLI overrides will overwrite values in the input file."""


def main(args: Args) -> list[str]:
    """Main entry point for plenoptic inference.

    Args:
        args: Parsed command-line arguments.

    Returns:
        List of output video paths.
    """
    inference_samples, _ = PlenopticInferenceArguments.from_files(args.input_files, overrides=args.overrides)
    init_output_dir(args.setup.output_dir, profile=args.setup.profile)

    from cosmos_transfer2.plenoptic import inference

    output_paths = inference(args.setup, inference_samples)
    return output_paths


if __name__ == "__main__":
    init_environment()

    try:
        args = tyro.cli(Args, description=__doc__, console_outputs=is_rank0(), config=(tyro.conf.OmitArgPrefixes,))
    except Exception as e:
        handle_tyro_exception(e)

    # pyrefly: ignore  # unbound-name
    main(args)

    cleanup_environment()
