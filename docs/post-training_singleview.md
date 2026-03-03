# Post-Training Cosmos-Transfer2 Models with Local Single-View Data

This guide provides instructions for post-training Cosmos-Transfer2 models with your own local video data.

## Table of Contents

<!--TOC-->

- [Table of Contents](#table-of-contents)
- [Prerequisites](#prerequisites)
- [Model Support](#model-support)
- [Control Modalities](#control-modalities)
- [Quick Start](#quick-start)
  - [Step 1: Prepare Dataset](#step-1-prepare-dataset)
  - [Step 2: Set Environment](#step-2-set-environment)
  - [Step 3: Start Training](#step-3-start-training)
- [Custom Data Preparation](#custom-data-preparation)
  - [1. Prepare Your Own Videos and Captions](#1-prepare-your-own-videos-and-captions)
  - [2. Verify Dataset Folder Format](#2-verify-dataset-folder-format)
  - [3. (Optional) Pre-compute Control Inputs](#3-optional-pre-compute-control-inputs)
  - [4. Final Verification](#4-final-verification)
- [Training Configuration](#training-configuration)
  - [Key Parameters](#key-parameters)
  - [Customizing via Command Line](#customizing-via-command-line)
- [Launch Training](#launch-training)
  - [1. Run Training](#1-run-training)
- [Monitoring](#monitoring)
  - [Sample Generations](#sample-generations)
  - [Training Time (2000 iterations)](#training-time-2000-iterations)
- [Checkpoint Management](#checkpoint-management)
  - [Convert DCP to PyTorch](#convert-dcp-to-pytorch)
  - [Directory Structure](#directory-structure)
  - [Resume Training](#resume-training)
- [Inference](#inference)
- [FAQ](#faq)
  - [Which control type to start with?](#which-control-type-to-start-with)
  - [How much data do I need?](#how-much-data-do-i-need)
  - [GPU out of memory?](#gpu-out-of-memory)
  - [What is `context_parallel_size`?](#what-is-context_parallel_size)
  - [How to know if training is working?](#how-to-know-if-training-is-working)
- [Troubleshooting](#troubleshooting)
  - ["CUDA out of memory"](#cuda-out-of-memory)
  - ["Video has only X frames, need Y"](#video-has-only-x-frames-need-y)
  - [Training loss not decreasing](#training-loss-not-decreasing)
  - [Generated videos have artifacts](#generated-videos-have-artifacts)
- [Additional Resources](#additional-resources)
- [Citation](#citation)

<!--TOC-->

## Prerequisites

1.**Env**:  Before proceeding, please read the [Post-training Guide](./post-training.md) for detailed setup steps and important post-training instructions, including checkpointing and best practices. This will ensure you are fully prepared for post-training with Cosmos-Transfer2.5.
2. **Hardware**: 8x H100/A100 (80GB) for 2B model
3. **Storage**: Sufficient space for dataset, checkpoints, and outputs
4. **Cache Setup**: For containers with limited disk space, configure environment variables to redirect cache

## Model Support

| Model | Parameters | Control Types | GPU Requirements |
|-------|------------|---------------|------------------|
| **Cosmos-Transfer2-2B** | 2B | edge, depth, seg, vis | 8x H100/A100 80GB |

**Recommended**: Start with 2B + edge control (no preprocessing required)

## Control Modalities

| Control | Preprocessing | Best For |
|---------|--------------|----------|
| **edge** ⚡ | None (on-the-fly) | Quick start, general scenes |
| **vis** ⚡ | None (on-the-fly) | Style transfer, denoising |
| **depth** | VideoDepthAnything | 3D-aware, robotics |
| **seg** | SAM2 | Object control, compositing |

**All control types use the same defaults**: `state_t=24`, `num_frames=93`, `context_parallel_size=8`

## Quick Start

Get started in 3 simple steps using the VideoUFO dataset:

**About VideoUFO:**
- **Dataset**: [WenhaoWang/VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO) on HuggingFace
- **Size**: 1M+ videos with detailed captions
- **Paper**: [VideoUFO: A Million-Scale User-Focused Dataset](https://openreview.net/forum?id=wwlwRuKle7) (NeurIPS 2025)
- **License**: CC BY 4.0

### Step 1: Prepare Dataset

Download and prepare the VideoUFO dataset using our automated script (no external dependencies required):

```bash
# Download and prepare 128 videos from VideoUFO (recommended for testing)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 128
```

The script will:
1. Download the VideoUFO metadata CSV (~1.1 GB)
2. Download tar files containing videos (~4 GB per tar file)
3. Extract videos and filter in real-time to keep only those with ≥93 frames
4. Continue extracting until it has the requested number of valid videos
5. Create caption JSON files (combining brief and detailed captions)
6. Organize everything into the required structure

**Script Options:**
- `--storage_dir`: Where to save the dataset (required)
- `--num_videos`: Number of videos to keep (default: 8)
- `--num_tars`: Number of tar files to download (default: 1, each contains ~5,400 videos)
- `--min_frames`: Minimum frames required per video (default: 93, matching training requirements)
- `--use_brief_caption`: Use brief captions only instead of brief + detailed
- `--no_concat_brief`: Use detailed captions only (don't concatenate with brief)
- `--skip_download`: Skip downloading if tar files already exist
- `--skip_metadata`: Skip downloading if metadata CSV already exists

**Note:** The script automatically filters videos during extraction, keeping only those with ≥93 frames. It continues extracting until it has the requested number of valid videos, ensuring all videos meet the training requirements (`num_frames=93`).

**Storage Requirements:** ~5-7 GB for 128 videos, ~30 GB for 1,000 videos, ~130 GB for 5,000 videos (from 1 tar file)

### Step 2: Set Environment

```bash
# Set output directory for checkpoints and samples
export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs
```

### Step 3: Start Training

Train with edge control (recommended - no preprocessing required):

```bash
# Train on 128 videos with edge control
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=assets/videoufo \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

**What this does:**
- Trains on 128 videos from VideoUFO
- Uses edge control (computed on-the-fly, no preprocessing)
- Runs for 2000 iterations (~0.8 hours on 8x A100)
- Saves checkpoints every 500 iterations
- Checkpoints saved to: `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_edge_example_*/checkpoints/`

**That's it!** Your model is now training.

**Next Steps:**
- Monitor training progress in the terminal output
- Once training completes, convert the checkpoint to PyTorch format (see [Checkpoint Management](#checkpoint-management))
- Run inference with your fine-tuned model (see [Inference](#inference))
- For custom datasets and advanced options, see the detailed sections below

---

## Custom Data Preparation

### 1. Prepare Your Own Videos and Captions

Organize your videos in the following structure:

```
datasets/your_dataset/
├── videos/
│   ├── video1.mp4 (720p, ≥3s, 10-60fps)
│   ├── video2.mp4
│   └── ...
└── captions/
    ├── video1.json  ({"caption": "Your description here"})
    ├── video2.json
    └── ...
```

**Video Requirements:**
- Format: MP4
- Resolution: 720p recommended
- Duration: ≥3 seconds
- Frame rate: 10-60 fps

**Caption Format:**
Each JSON file should contain:
```json
{
    "caption": "A detailed description of the video content"
}
```

### 2. Verify Dataset Folder Format

Your dataset should follow this structure:

```
datasets/your_dataset/
├── videos/
│   └── *.mp4
└── captions/
    └── *.json
```

For datasets with pre-computed control inputs (depth/seg):

```
datasets/your_dataset/
├── videos/
│   └── *.mp4
├── captions/
│   └── *.json
├── depth/
│   └── *.mp4 (optional, for depth control)
└── seg/
    └── *.mp4 (optional, for segmentation control)
```

**Note**: Text captions are encoded on-the-fly during training using the model's built-in text encoder (similar to multiview training).

### 3. (Optional) Pre-compute Control Inputs

**Skip for edge/vis** - these are computed on-the-fly during training

**For depth:**
Depth control requires pre-computed depth maps. Use the built-in DepthAnything V2 pipeline:

```bash

# Generate depth video for a single video
python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/depth/video1.mp4 \
    --encoder vits

# Process multiple videos (example loop)
for video in datasets/your_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/your_dataset/depth/$basename" \
        --encoder vits
done
```

**Parameters**:
- `--input_video`: Path to input RGB video
- `--output_video`: Path to save depth video (MP4 format)
- `--encoder`: Model size (`vits` for small/fast, `vitl` for large/accurate)

**Output**: Grayscale depth video in MP4 format, same resolution and frame count as input.

**For seg:**
Segmentation control requires pre-computed segmentation masks. Use the built-in SAM2 pipeline:

```bash
# Generate segmentation video using text prompts (recommended)
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/seg/video1.mp4 \
    --mode prompt \
    --prompt "person, car, vehicle, building, tree, road, sky" \
    --visualize

# Process multiple videos (example loop)
for video in datasets/your_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/your_dataset/seg/$basename" \
        --mode prompt \
        --prompt "person, car, vehicle, building, tree, road, sky" \
        --visualize
done
```

**Segmentation modes**:
1. **Prompt mode** (recommended): `--mode prompt --prompt "person, car, building"`
2. **Box mode**: `--mode box --box "300,0,500,400"`
3. **Points mode**: `--mode points --points "200,300" --labels "1"`

**Parameters**:
- `--input_video`: Path to input RGB video
- `--output_video`: Path to save segmentation video (MP4 format)
- `--mode`: Segmentation mode (`prompt`, `box`, or `points`)
- `--prompt`: Text description of objects to segment (for prompt mode)
- `--visualize`: Required flag to enable video output

**Output**: Color-coded segmentation video in MP4 format where each object instance has a unique color.


### 4. Final Verification

Before training, verify your dataset has all required components:
```
datasets/your_dataset/
├── videos/
│   └── *.mp4
├── captions/
│   └── *.json
└── depth/  (optional, for depth control only)
    └── *.mp4
```

Make sure:
- Each video in `videos/` has a corresponding JSON file in `captions/`
- File names match (e.g., `video1.mp4` → `video1.json`)
- Caption JSON files contain valid text descriptions

## Training Configuration

Four experiments are available in `cosmos_transfer2.experiments.singleview.cosmos_singleview_example`:

1. **Edge Control** (Recommended): `experiment=transfer2_singleview_posttrain_edge_example`
2. **Depth Control**: `experiment=transfer2_singleview_posttrain_depth_example`
3. **Seg Control**: `experiment=transfer2_singleview_posttrain_seg_example`
4. **Visual Blur Control**: `experiment=transfer2_singleview_posttrain_vis_example`

**Text Encoding:** All experiments use Qwen2.5-VL-7B (`reason1p1_7B`) for on-the-fly caption encoding to match the pretrained model's training distribution.

### Key Parameters

All control types (edge, depth, seg, vis) share the same default parameters:

```python
# Dataset
dataset_dir="datasets/your_dataset"          # Your dataset path
num_frames=93                                 # (state_t-1)*4+1 = (24-1)*4+1 = 93
video_size=(704, 1280)                        # (H, W) - 720p, 16:9 aspect ratio
hint_key="control_input_edge"                 # Control type: edge, depth, seg, or vis

# Model
state_t=24                                    # Temporal latent size
context_parallel_size=8                       # Must divide state_t (adjust based on GPUs)

# Text Encoder (matches pretrained model)
text_encoder_class="reason1p1_7B"             # Qwen2.5-VL-7B
embedding_concat_strategy="FULL_CONCAT"       # All 28 layers
crossattn_proj_in_channels=100352             # 3584 * 28
crossattn_emb_channels=1024                   # Projected dimension

# Training
max_iter=5000
save_iter=1000                                # Checkpoint save frequency
lr=5e-5                                       # Learning rate
warm_up_steps=1000                            # LR warmup to prevent gradient spikes
grad_accum_iter=4

# Checkpoint (auto-downloaded from HuggingFace)
load_path=get_checkpoint_path(...)            # Auto-downloads on first run
dcp_async_mode_enabled=False                  # Disabled for stability
```

### Customizing via Command Line

```bash
# Example with custom dataset
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500

# Example with VideoUFO dataset
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=assets/videoufo \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500
```

**Required Parameters:**
- `dataloader_train.dataset.dataset_dir`: Path to your dataset directory (e.g., `datasets/your_dataset` or `assets/videoufo`)
- `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'`: Links sampler to dataset (ensures multi-GPU data partitioning works correctly)

**Optional Parameters:**
- `trainer.max_iter`: Total training iterations (default: 5000)
- `checkpoint.save_iter`: Checkpoint save frequency (default: 1000)
- `optimizer.lr`: Learning rate (default: 5e-5)
- `scheduler.warm_up_steps`: LR warmup steps (default: [1000])
- `job.wandb_mode`: W&B mode (online/offline/disabled)
- `checkpoint.load_path`: Override auto-download with custom checkpoint path

## Launch Training

**Note:** Before training, set your output directory: `export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs`

### 1. Run Training

**8 GPUs (2B model):**
```bash
# Edge control
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Depth control (requires pre-computed depth videos)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Seg control (requires pre-computed segmentation masks)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_seg_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Visual blur control (requires pre-computed visual blur videos)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_vis_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled
```

**Key Parameters:**
- `experiment`: Choose the control type (edge, depth, seg, or vis)
- `dataloader_train.dataset.dataset_dir`: Path to your dataset
- `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'`: Links sampler to dataset for multi-GPU training
- `job.wandb_mode=disabled`: Disable W&B logging (optional)

**Note:** The sampler parameter uses `${...}` syntax to reference the dataset object, ensuring proper data partitioning across GPUs.

**Note:** Pretrained checkpoints are **automatically downloaded** from HuggingFace on first run! No manual download needed.

**Checkpoint Output:**

Checkpoints are saved to `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`. By default, `IMAGINAIRE_OUTPUT_ROOT` is `/tmp/imaginaire4-output`. We strongly recommend setting `IMAGINAIRE_OUTPUT_ROOT` to a location with sufficient storage space for your checkpoints.

In the example above (using the `edge` experiment), `PROJECT`, `GROUP`, and `NAME` come from the experiment configuration's `job` dict:
- `PROJECT` = `cosmos_transfer2_posttrain`
- `GROUP` = `local_single_view`
- `NAME` = `transfer2_singleview_posttrain_edge_example_2025-11-21_16-30-45` (timestamp is added automatically)

So the full checkpoint path would be:
```
${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_vis_example_2025-11-21_16-30-45/checkpoints/
```

## Monitoring
```
[Iteration 100/2000] Loss: 0.234, LR: 4.95e-05, Time: 1.23s/it  # After warmup
[Iteration 200/2000] Saving checkpoint...
```

### Sample Generations
- Every 50 iterations (default)
- Saved to: `${IMAGINAIRE_OUTPUT_ROOT}/<project>/<group>/<name>/samples/`

### Training Time (2000 iterations)

| Dataset | Model | GPUs | Time/Iter | Total |
|---------|-------|------|-----------|-------|
| 128 videos | 2B | 8x A100 | ~1.2s | ~0.7h |
| 1000 videos | 2B | 8x A100 | ~1.5s | ~0.8h |
| 128 videos | 14B | 16x A100 | ~2.0s | ~1.1h |

## Checkpoint Management

### Convert DCP to PyTorch

```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer2_posttrain/local_single_view/2B_edge_posttrain_*/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

Creates:
- `model_ema_bf16.pt` ← **Use this for inference**
- `model_ema_fp32.pt`
- `model.pt` (full checkpoint)

### Directory Structure

```
${IMAGINAIRE_OUTPUT_ROOT}/
└── cosmos_transfer2_posttrain/
    └── local_single_view/
        └── 2B_edge_posttrain_2025-11-19_10-30-00/
            ├── checkpoints/
            │   ├── iter_000000200/model_ema_bf16.pt ← Use this
            │   └── latest_checkpoint.txt
            └── samples/
```

### Resume Training

Simply rerun the same command - it auto-resumes from latest checkpoint.

## Inference

```bash
# Using the inference script
torchrun --nproc_per_node=8 examples/inference.py \
    -i assets/edge.jsonl \
    -o outputs/ \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment transfer2_singleview_posttrain_edge_example
```

## FAQ

### Which control type to start with?
**Edge control** - no preprocessing, fast iteration, optimized memory config.

### How much data do I need?
- Minimum: 50-100 videos
- Recommended: 200-500 videos
- Optimal: 1000+ videos

### GPU out of memory?
1. Increase `context_parallel_size` (must divide `state_t` evenly, e.g., 12 or 24)
2. Reduce `num_frames` and `state_t` together (they must match via the formula `num_frames = (state_t-1)*4+1`):
   - 93 frames → 77 frames: `dataloader_train.dataset.num_frames=77 model.config.state_t=20`
   - 93 frames → 61 frames: `dataloader_train.dataset.num_frames=61 model.config.state_t=16`
3. Enable `dcp_async_mode_enabled=True` (disabled by default for stability)
4. Save less frequently: increase `save_iter`
5. Use more GPUs to distribute memory load

### What is `context_parallel_size`?
Splits sequence across GPUs to reduce per-GPU memory. Must evenly divide `state_t`:
- Default: `state_t=24` ÷ `context_parallel_size=8` = 3 latent frames/GPU
- Example: `state_t=24` ÷ `context_parallel_size=4` = 6 latent frames/GPU

The relationship between latent frames (`state_t`) and pixel frames (`num_frames`):
```
num_frames = (state_t - 1) * 4 + 1
# e.g., state_t=24 → num_frames = (24-1)*4+1 = 93
```

### How to know if training is working?
- Loss: ~0.5 → ~0.1-0.2
- Sample quality improves over iterations
- No NaN/Inf in logs
- Smooth W&B curves

## Troubleshooting

### "CUDA out of memory"
1. Increase `context_parallel_size` (must divide `state_t`, e.g., 12 or 24)
2. Reduce `num_frames` and `state_t` together:
   - `dataloader_train.dataset.num_frames=77 model.config.state_t=20`
   - `dataloader_train.dataset.num_frames=61 model.config.state_t=16`
3. Use more GPUs

### "Video has only X frames, need Y"
- Use longer videos (≥4s at 24 FPS for 93 frames)
- Reduce `num_frames` and `state_t` together:
  - `dataloader_train.dataset.num_frames=77 model.config.state_t=20` (requires ≥3.2s)
  - `dataloader_train.dataset.num_frames=61 model.config.state_t=16` (requires ≥2.5s)
- Filter short videos during preprocessing

### Training loss not decreasing
- Check learning rate (try different values)
- Verify data quality
- Confirm checkpoint loaded correctly
- Add more data

### Generated videos have artifacts
- Train longer
- Use higher quality training data
- Adjust guidance scale (inference)
- Try EMA checkpoint vs. regular

---

## Additional Resources

**Key Files:**
- Training experiments: `cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py`
- Config wrapper: `cosmos_transfer2/singleview_config.py`
- Dataset loader: `projects/cosmos/transfer2/datasets/local_datasets/singleview_dataset.py`
- Dataloader config: `projects/cosmos/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py`
- VideoUFO dataset preparation script: `scripts/prepare_videoufo_dataset.py`

**Related Docs:**
- [Setup Guide](setup.md)
- [Inference Guide](inference.md)
- [Multiview Training](post-training_auto_multiview.md)
- [Post-Training Overview](post-training.md)

**Example Datasets:**
- [VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO) - 1M+ videos with detailed captions (use `scripts/prepare_videoufo_dataset.py` for easy setup)

## Citation

```bibtex
@article{cosmos2025,
  title={Cosmos World Foundation Models},
  author={NVIDIA Research},
  year={2025}
}
```
