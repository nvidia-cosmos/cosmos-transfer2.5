#!/bin/bash
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


# Mandatory for the script to fail if any command fails
set -eux

# cd to repository root
cd $(git rev-parse --show-toplevel)

# Export the mandatory variables for every test case! CI will fail if these are not set
export OUTPUT_DIR=outputs/robot_edge
export GENERATED_OUTPUT_FILE=$OUTPUT_DIR/output.mp4

# create the output directory
mkdir -p $OUTPUT_DIR

# Run the robot inference command
NUM_GPUS=1
torchrun --nproc_per_node=$NUM_GPUS --master_port=12345 examples/inference.py \
  --num_gpus $NUM_GPUS \
  --params_file assets/robot_example/edge/robot_edge_spec.json

# Verify that the output file exists
if [ ! -f "$GENERATED_OUTPUT_FILE" ]; then
    echo "Output file $GENERATED_OUTPUT_FILE does not exist! Check the inference generation command."
    exit 1
fi

# Run the car inference command
export OUTPUT_DIR=outputs/car_edge
export GENERATED_OUTPUT_FILE=$OUTPUT_DIR/output.mp4
NUM_GPUS=1
torchrun --nproc_per_node=$NUM_GPUS --master_port=12345 examples/inference.py \
  --num_gpus $NUM_GPUS \
  --params_file assets/car_example/edge/car_edge_spec.json

# Verify that the output file exists
if [ ! -f "$GENERATED_OUTPUT_FILE" ]; then
    echo "Output file $GENERATED_OUTPUT_FILE does not exist! Check the inference generation command."
    exit 1
fi
