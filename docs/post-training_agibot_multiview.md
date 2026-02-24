# Agibot Multiview Post-training

This guide covers post-training the Cosmos-Transfer2.5 Agibot 3-view control models (head_color, hand_left, hand_right) with edge, depth, seg, or vis control using your own local data.

## Table of Contents

<!--TOC-->

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [1. Preparing Data](#1-preparing-data)
  - [1.1 Dataset layout](#11-dataset-layout)
  - [1.2 Example layout](#12-example-layout)
  - [1.3 Example dataset for preparing post-train data](#13-example-dataset-for-preparing-post-train-data)
- [2. Post-training](#2-post-training)
  - [2.1 Environment variables](#21-environment-variables)
  - [2.2 Run command](#22-run-command)
  - [2.3 Overriding training hyperparameters](#23-overriding-training-hyperparameters)
- [3. See also](#3-see-also)

<!--TOC-->

## Prerequisites

Before proceeding, read the [Post-training Guide](./post-training.md) for environment setup, Hugging Face configuration, and checkpointing. In addition:

- Set `COSMOS_EXPERIMENTAL_CHECKPOINTS=1` so base checkpoints resolve (e.g. from Hugging Face when not internal). Without this, the run will **train from scratch** instead of loading the pre-trained checkpoint.

## 1. Preparing Data

### 1.1 Dataset layout

Use a single directory containing:

- **Videos:** `videos/head_color/*.mp4`, `videos/hand_left/*.mp4`, `videos/hand_right/*.mp4` (same sample IDs across views, e.g. `episode_000000.mp4`).
- **Captions:** `captions/head_color/*.json`, `captions/hand_left/*.json`, `captions/hand_right/*.json`. Each JSON must have `{"caption": "..."}`. The caption for `head_color` is used as the training prompt; other views can share or duplicate. Use `scripts/write_agibot_captions.py` to generate placeholder captions if needed.
- **Control (optional):** For **depth** or **seg** control only, add `control_input_depth/` or `control_input_seg/` with the same view subdirs and `.mp4` files (same stems as in `videos/`). For **edge** or **vis**, control is derived from the video at runtime.

### 1.2 Example layout

```
your_dataset/
├── captions/
│   ├── head_color/
│   │   ├── episode_000000.json
│   │   └── ...
│   ├── hand_left/
│   │   └── ...
│   └── hand_right/
│       └── ...
├── videos/
│   ├── head_color/
│   │   ├── episode_000000.mp4
│   │   └── ...
│   ├── hand_left/
│   │   └── ...
│   └── hand_right/
│       └── ...
└── (optional) control_input_depth/ or control_input_seg/
    ├── head_color/
    ├── hand_left/
    └── hand_right/
```

### 1.3 Example dataset for preparing post-train data

The repository includes an example layout at **`assets/robot_multiview_control_posttrain-agibot`** (when using the Cosmos-Assets dataset). You can use it as a reference for preparing your own post-train data:

- **videos/** — 3-view videos (head_color, hand_left, hand_right) with matching sample IDs.
- **captions/** — Per-view caption JSONs (`{"caption": "..."}`); head_color caption is used as the training prompt.
- **control_input_depth/** and **control_input_seg/** — Optional pre-computed control videos for depth and seg experiments (same view structure and stems as `videos/`).

For **edge** and **vis**, control is computed from the video at runtime, so no control folders are required. The docs test script `tests/docs_test/post-training_agibot_multiview.sh` runs post-train for all four control types using this dataset.

## 2. Post-training

### 2.1 Environment variables

```bash
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1
```

### 2.2 Run command

```shell exclude=true
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
  -- \
  experiment=transfer2_agibot_posttrain_edge_example \
  dataloader_train.dataset.dataset_dir=assets/robot_multiview_control_posttrain-agibot \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  job.wandb_mode=disabled
```

Replace `/path/to/your/dataset` with your actual dataset directory (the example command set it to `assets/robot_multiview_control_posttrain-agibot`).

- **Experiments:** Use `transfer2_agibot_posttrain_edge_example`, `transfer2_agibot_posttrain_depth_example`, `transfer2_agibot_posttrain_seg_example`, or `transfer2_agibot_posttrain_vis_example` for the desired control type.
- **Dataset path:** Set `dataloader_train.dataset.dataset_dir` to your dataset directory (absolute or relative to cwd). Do not use the literal `datasets/your_dataset` unless that folder exists.
- **Sampler:** `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'` keeps the sampler pointing at the overridden dataset.
- **GPUs:** Set `--nproc_per_node` to your number of GPUs (e.g. 4 or 8).

### 2.3 Overriding training hyperparameters

Override any of these from the command line (same `--` block as above):

| Param | Override key | Example | Default (post-train) |
|-------|--------------|---------|----------------------|
| Max iterations | `trainer.max_iter` | `trainer.max_iter=30000` | 5_000 |
| Logging interval | `trainer.logging_iter` | `trainer.logging_iter=100` | 50 |
| Learning rate | `optimizer.lr` | `optimizer.lr=1e-4` | 5e-5 (from base) |
| Checkpoint save interval | `checkpoint.save_iter` | `checkpoint.save_iter=1000` | 500 |
| LR warm-up steps | `scheduler.warm_up_steps` | `scheduler.warm_up_steps=[500]` | [1000] (from base) |
| LR cycle length | `scheduler.cycle_lengths` | `scheduler.cycle_lengths=[10000]` | [400000] (from base) |

Example with custom lr and iterations:

```shell exclude=true
torchrun --nproc_per_node=8 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
  -- \
  experiment=transfer2_agibot_posttrain_edge_example \
  dataloader_train.dataset.dataset_dir=assets/robot_multiview_control_posttrain-agibot \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  trainer.max_iter=10000 \
  optimizer.lr=1e-4 \
  job.wandb_mode=disabled
```

## 3. See also

- [Inference: Robot multiview control (Agibot)](./inference_robot_multiview_control.md) for running inference with the same models.
- [Post-training Guide](./post-training.md) for general setup and checkpointing.
