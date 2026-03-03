# Auto Multiview Post-training for HDMap

This guide provides instructions on running post-training with the Cosmos-Transfer2.5 Auto Multiview 2B model.

**For more information and examples about post-training the Auto Multiview model, checkout our Cosmos Cookbook [Auto Multiview Post-training](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/transfer2_5/av_world_scenario_maps/post_training.html) recipe.**

## Table of Contents

<!--TOC-->

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [1. Preparing Data](#1-preparing-data)
  - [1.1 Prepare Transfer Multiview Training Dataset](#11-prepare-transfer-multiview-training-dataset)
  - [1.2 Verify the dataset folder format](#12-verify-the-dataset-folder-format)
- [2. Post-training](#2-post-training)
- [3. Inference with the Post-trained checkpoint](#3-inference-with-the-post-trained-checkpoint)
  - [3.1 Converting DCP Checkpoint to Consolidated PyTorch Format](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [3.2 Running Inference](#32-running-inference)

<!--TOC-->

## Prerequisites

Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Transfer2.5.

## 1. Preparing Data

### 1.1 Prepare Transfer Multiview Training Dataset

The first step is preparing a dataset with videos.

You must provide a folder containing a collection of videos in **MP4 format**, preferably 720p, as well as a corresponding folder containing a collection of the hdmap control input videos in  **MP4 format**. The views for each samples should be further stratified by subdirectories with the camera name. We have an example dataset that can be used at `assets/multiview_hdmap_posttrain_dataset`

### 1.2 Verify the dataset folder format

Dataset folder format:

```
assets/multiview_hdmap_posttrain_dataset/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       └── *.json
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
├── videos/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
```

## 2. Post-training

Run the following command to execute an example post-training job with multiview data.

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=transfer2_auto_multiview_post_train_example job.wandb_mode=disabled
```

The model will be post-trained using the multiview dataset. See the [data config](../cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/defaults/dataloader_local.py) to understand how the dataloader is defined.

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`. By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

In the above example, `PROJECT` is `cosmos_transfer_v2p5`, `GROUP` is `auto_multiview`, `NAME` is `2b_cosmos_multiview_post_train_example`.

See the job config to understand how they are determined.

```python
transfer2_auto_multiview_post_train_example = dict(
    dict(
        ...
        job=dict(
            project="cosmos_transfer_v2p5",
            group="auto_multiview",
            name="2b_cosmos_multiview_post_train_example"
        ),
        ...
    )
)
```

## 3. Inference with the Post-trained checkpoint

### 3.1 Converting DCP Checkpoint to Consolidated PyTorch Format

Since the checkpoints are saved in DCP format during training, you need to convert them to consolidated PyTorch format (.pt) for inference. Use the `convert_distcp_to_pt.py` script:

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

This conversion will create three files:

- `model.pt`: Full checkpoint containing both regular and EMA weights
- `model_ema_fp32.pt`: EMA weights only in float32 precision
- `model_ema_bf16.pt`: EMA weights only in bfloat16 precision (recommended for inference)

### 3.2 Running Inference

After converting the checkpoint, you can run inference with your post-trained model using a JSON configuration file that specifies the inference parameters (see `assets/multiview_example/multiview_spec.json` for an example).

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_spec.json -o outputs/postrained-auto-mv --checkpoint_path $CHECKPOINT_DIR/model_ema_bf16.pt --experiment transfer2_auto_multiview_post_train_example
```

Generated videos will be saved to the output directory (e.g., `outputs/postrained-auto-mv/`).

For more inference options and advanced usage, see [docs/inference.md](./inference.md).
