#!/usr/bin/env -S bash -euxo pipefail
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


export DATASET_DIR="$INPUT_DIR/assets/multiview_hdmap_posttrain_dataset"

# Train the model
torchrun --standalone --nproc_per_node=$NUM_GPUS scripts/train.py \
    --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
    -- \
    experiment=transfer2_auto_multiview_post_train_example \
    job.wandb_mode=disabled \
    ~trainer.callbacks.wandb \
    ~trainer.callbacks.wandb_10x \
    dataloader_train.dataset.dataset_dir="$DATASET_DIR" \
    dataloader_train.sampler.dataset.dataset_dir="$DATASET_DIR"


# Get path to the latest checkpoint
CHECKPOINTS_DIR="${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints"
CHECKPOINT_ITER="$(cat "$CHECKPOINTS_DIR/latest_checkpoint.txt")"
CHECKPOINT_DIR="$CHECKPOINTS_DIR/$CHECKPOINT_ITER"

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py "$CHECKPOINT_DIR/model" "$CHECKPOINT_DIR"

torchrun --standalone --nproc_per_node=$NUM_GPUS examples/multiview.py \
    -i "$INPUT_DIR/assets/multiview_example/multiview_spec.json" \
    -o "$OUTPUT_DIR" \
    --checkpoint-path "$CHECKPOINT_DIR/model_ema_bf16.pt" \
    --experiment transfer2_auto_multiview_post_train_example
