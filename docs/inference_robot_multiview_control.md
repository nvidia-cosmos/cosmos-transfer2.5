# Robot Multiview Control-Conditioned Inference

We recommend first reading the [Inference Guide](inference.md).
This guide covers control-conditioned video generation for robot multiview scenarios, specifically for the Agibot dataset with 3-camera views (head, left hand, right hand).

## Prerequisites

[Setup Guide](setup.md)

## Overview

Control-conditioned inference allows you to guide video generation using various control signals in addition to the ego video input. This provides more precise control over the generated output by incorporating structural information.

### Supported Control Types

Four control types are available for Agibot multiview models:

| Control Type | Description | Generation Method | Use Case |
|--------------|-------------|-------------------|----------|
| **edge** | Canny edge detection maps | Generated on-the-fly from input videos | Preserve sharp boundaries and structural edges |
| **vis** | Visual blur maps | Generated on-the-fly from input videos | Control visual clarity and focus |
| **depth** | Depth estimation maps | Pre-computed, loaded from disk | Preserve 3D scene structure |
| **seg** | Segmentation maps | Pre-computed, loaded from disk | Maintain object boundaries and semantic regions |

### Camera Configuration

The Agibot setup uses 3 synchronized cameras:
- **View 0**: `head_color` - Head-mounted camera
- **View 1**: `hand_left` - Left hand camera
- **View 2**: `hand_right` - Right hand camera

## Example Visualizations

Below are example outputs for each control type. Each row shows the control signal (left) and the corresponding generated output (right):

### Edge Control

<table>
<tr>
<td width="50%">

**Control Input**

![296_656371_chunk0_control_edge](https://github.com/user-attachments/assets/2d4adef2-66fa-49ed-9d02-5a8698c1cc6d)

</td>
<td width="50%">

**Generated Output**

![296_656371_chunk0_output_edge](https://github.com/user-attachments/assets/40f204ba-d8e0-4b10-aa3a-82678ca18b7c)

</td>
</tr>
</table>

### Vis (Visual Blur) Control

<table>
<tr>
<td width="50%">

**Control Input**

![296_656371_chunk0_control_vis](https://github.com/user-attachments/assets/ba9ef039-595b-4fb1-b655-aa4521350565)

</td>
<td width="50%">

**Generated Output**

![296_656371_chunk0_output_vis](https://github.com/user-attachments/assets/bf0bbc5e-e57d-412b-999d-43e9abfff7b2)

</td>
</tr>
</table>

### Depth Control

<table>
<tr>
<td width="50%">

**Control Input**

![296_656371_chunk0_control_depth](https://github.com/user-attachments/assets/c4b5360a-6cbb-4d61-bc30-25e96f8992ae)

</td>
<td width="50%">

**Generated Output**

![296_656371_chunk0_output_depth](https://github.com/user-attachments/assets/f38a1f74-b1ec-4eff-bc5b-9f6a69ebd1d4)

</td>
</tr>
</table>

### Seg (Segmentation) Control

<table>
<tr>
<td width="50%">

**Control Input**

![296_656371_chunk0_control_seg](https://github.com/user-attachments/assets/f2b1aae5-08af-49a0-8dc8-6f7a826600a4)

</td>
<td width="50%">

**Generated Output**

![296_656371_chunk0_output_seg](https://github.com/user-attachments/assets/b46361e5-ff20-4460-be42-82c8ccb9eb26)

</td>
</tr>
</table>

## Image-to-Video (I2V) vs Text-to-Video (T2V)

The Agibot models support two inference modes:

- **Image-to-Video (I2V)**: Generate videos conditioned on both control signals and an input frame (`num_conditional_frames=1`, default)
- **Text-to-Video (T2V)**: Generate videos conditioned only on control signals and text captions (`num_conditional_frames=0`)

**The examples shown above use I2V mode (default).** To use T2V mode instead, add `"num_conditional_frames": 0` to your JSON configuration file.

**T2V Mode with Edge/Vis Controls** (input videos required):

```json
{
    "name": "my_t2v_edge_example",
    "seed": 42,
    "guidance": 3.0,
    "num_conditional_frames": 0,
    "prompt": "A robotic arm assembling circuit boards in a well-lit factory",
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

**T2V Mode with Depth/Seg Controls** (input videos optional):

```json
{
    "name": "my_t2v_depth_example",
    "seed": 42,
    "guidance": 3.0,
    "num_conditional_frames": 0,
    "prompt": "A robotic arm carefully picking up delicate electronic components",
    "head_color": {
        "control_path": "depth/296_656371_chunk0_head_color_depth.mp4"
    },
    "hand_left": {
        "control_path": "depth/296_656371_chunk0_hand_left_depth.mp4"
    },
    "hand_right": {
        "control_path": "depth/296_656371_chunk0_hand_right_depth.mp4"
    }
}
```

**Input Video Requirements for T2V Mode:**
- **Edge/Vis controls**: Input videos in `input_path` are **required** (needed to generate control signals on-the-fly), but won't be used as conditioning for generation
- **Depth/Seg controls**: Input videos in `input_path` are **optional** - if not provided, control videos will be used as mock input

## Setup and Data Preparation

### Using Example Data

Example data for all 4 control types is automatically available in `assets/robot_multiview_control-agibot/`. You can use these examples to test the models without preparing your own data.

### Preparing Your Own Data

To use your own robot multiview data, organize it as follows:

```
your_data_root/
├── videos/                          # Required: Input RGB videos
│   ├── {video_id}_head_color_rgb.mp4
│   ├── {video_id}_hand_left_rgb.mp4
│   └── {video_id}_hand_right_rgb.mp4
├── captions/                        # Recommended: Text captions (see note below)
│   ├── {video_id}_head_color.txt
│   ├── {video_id}_hand_left.txt
│   └── {video_id}_hand_right.txt
├── depth/                           # Required for depth control only
│   ├── {video_id}_head_color_depth.mp4
│   ├── {video_id}_hand_left_depth.mp4
│   └── {video_id}_hand_right_depth.mp4
└── seg/                             # Required for seg control only
    ├── {video_id}_head_color_seg.mp4
    ├── {video_id}_hand_left_seg.mp4
    └── {video_id}_hand_right_seg.mp4
```

### Control-Specific Requirements

- **Edge Control**: Only requires input videos in `videos/` folder. Edge maps are generated automatically.
- **Visual Control**: Only requires input videos in `videos/` folder. Visual blur maps are generated automatically.
- **Depth Control**: Requires both input videos in `videos/` AND pre-computed depth maps in `depth/` folder.
- **Seg Control**: Requires both input videos in `videos/` AND pre-computed segmentation maps in `seg/` folder.

### Inference Configuration JSON Format

The JSON configuration specifies per-camera input and control paths. Each camera view requires:
- `input_path`: Path to the input RGB video (required)
- `control_path`: Path to the control video (required for depth/seg, not needed for edge/vis)

#### Edge/Visual Blur Controls (On-the-Fly Generation)

For edge and visual blur controls, only `input_path` is needed. The control maps are computed automatically from the input videos.

```json
{
    "name": "my_edge_example",
    "seed": 42,
    "guidance": 3.0,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

#### Depth/Segmentation Controls (Pre-computed)

For depth and segmentation controls, both `input_path` and `control_path` are required.

```json
{
    "name": "my_depth_example",
    "seed": 42,
    "guidance": 3.0,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_head_color_depth.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_hand_left_depth.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_hand_right_depth.mp4"
    }
}
```

**Required fields:**
- `name`: Name for this inference run. This will be used as the output filename (e.g., `outputs/my_test_example.mp4`)
- `head_color`, `hand_left`, `hand_right`: Per-camera configuration with `input_path` (and optionally `control_path`)
- `seed`: Random seed for reproducibility
- `guidance`: Classifier-free guidance scale (typically 3.0-7.0)

**Note on Paths**: The paths in `input_path` and `control_path` are relative to the `--input-root` directory specified on the command line.

**Optional fields:**
- `num_conditional_frames`: Set to `0` for T2V mode, `1` for I2V mode (default: `1`)
- `num_steps`: Number of diffusion steps (default: `35`)
- `fps`: Frames per second (default: `10`)
- `target_height`: Output height (default: `720`)
- `target_width`: Output width (default: `1280`)
- `prompt`: Text prompt override applied to all camera views. Takes highest priority over caption files and camera prefixes. **Recommended**: Provide descriptive prompts for better quality
- `preset_edge_threshold`: Edge detection strength for edge control: `very_low`, `low`, `medium`, `high`, `very_high` (default: `medium`)
- `preset_blur_strength`: Blur strength for vis control: `very_low`, `low`, `medium`, `high`, `very_high` (default: `medium`)
- `control_weight`: Control signal strength from `0.0` to `1.0` (default: `1.0`). Higher values make the output adhere more strongly to the control signal. (default to 1.0)
- `save_combined_views`: If `true` (default), saves only one combined horizontal video. If `false`, saves 3 individual view videos plus one combined horizontal video
- `enable_autoregressive`: Enable autoregressive mode for longer video generation (default: `false`)
- `num_chunks`: Number of chunks to generate in autoregressive mode (default: `2`, only used when `enable_autoregressive=true`)
- `chunk_overlap`: Number of overlapping video frames between consecutive chunks (default: `1`, only used when `enable_autoregressive=true`)

## Running Inference

### Basic Command Template

```shell exclude=true
torchrun \
    --nproc_per_node=<NUM_GPUS> \
    examples/robot_multiview_agibot_control.py \
    -i <PATH_TO_TEST_JSON> \
    -o <OUTPUT_DIR> \
    --input-root <DATA_ROOT> \
    --control-type <CONTROL_TYPE>
```

**Note**: All commands should be run from the `packages/cosmos-transfer2/` directory.

### Parameters

- `--nproc_per_node`: Number of GPUs to use (**minimum 4 required**)
- `-i`: Path to test JSON file
- `-o`: Output directory for generated videos
- `--input-root`: Root directory containing your data
- `--control-type`: One of `edge`, `vis`, `depth`, or `seg`

**Default Output Resolution**: The model generates 720×1280 (height×width) output videos by default. This can be customized per sample in the inference JSON files using `target_height` and `target_width` parameters.

> **GPU Requirements**:
>
> The model requires **minimum 4 GPUs** due to context parallelism constraints:
>
> **Valid GPU counts: 4, 8, or 16**
> - 4 GPUs: Recommended for development/testing
> - 8 GPUs: Good balance of speed and resource usage
> - 16 GPUs: Maximum parallelization for production

### Example Commands

#### Edge Control (On-the-fly Generation)

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/edge_test.json \
    -o outputs/agibot_edge_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type edge
```

#### Visual Blur Control (On-the-fly Generation)

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/vis_test.json \
    -o outputs/agibot_vis_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type vis
```

#### Depth Control (Pre-computed)

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/depth_test.json \
    -o outputs/agibot_depth_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type depth
```

#### Segmentation Control (Pre-computed)

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/seg_test.json \
    -o outputs/agibot_seg_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type seg
```

#### Autoregressive Mode (Longer Videos)

For generating videos longer than the model's native temporal capacity, use autoregressive mode. Create a JSON config with autoregressive parameters:

```json
{
    "name": "agibot_edge_test_autoregressive",
    "seed": 42,
    "guidance": 3.0,
    "enable_autoregressive": true,
    "num_chunks": 2,
    "chunk_overlap": 1,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_long_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_long_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_long_hand_right_rgb.mp4"
    }
}
```

Then run inference:

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/edge_test_autoregressive.json \
    -o outputs/agibot_edge_test_autoregressive \
    --input-root assets/robot_multiview_control-agibot \
    --control-type edge
```

The model will generate multiple overlapping chunks and stitch them together for a longer output video. The total number of output frames will be: `num_video_frames_per_chunk + (num_video_frames_per_chunk - chunk_overlap) * (num_chunks - 1)`.

## Output Format

Generated videos can be saved in two modes:

### Combined View Mode (default, `save_combined_views: true`)

Saves **only one video** with all 3 camera views arranged horizontally:

```
<OUTPUT_DIR>/
└── <name>.mp4  # Combined video with all 3 views (head_color | hand_left | hand_right)
```

For example, with `"name": "my_test_example"`, the output will be `outputs/my_test_example.mp4`.

### Split View Mode (`save_combined_views: false`)

Saves **four videos**: three individual camera view videos plus one combined horizontal video:

```
<OUTPUT_DIR>/
├── <name>_head_color.mp4      # Individual head camera view
├── <name>_hand_left.mp4       # Individual left hand camera view
├── <name>_hand_right.mp4      # Individual right hand camera view
└── <name>_grid.mp4            # Combined horizontal view (head_color | hand_left | hand_right)
```

For example, with `"name": "my_test_example"`, the outputs will be `my_test_example_head_color.mp4`, etc.

The grid video is the combined view - it shows all 3 views arranged horizontally in a single video.

## Technical Details

### Control Signal Processing

**On-the-fly Controls (Edge, Visual)**:
- Edge maps: Generated using Canny edge detection on input videos
- Visual blur maps: Generated using blur detection on input videos
- Processed at inference time, no pre-computation needed
- Consistent across all frames of the input sequence

**Pre-computed Controls (Depth, Seg)**:
- Loaded from disk as video files
- Must match the temporal length of input videos
- Should be synchronized frame-by-frame with input videos
- Typically generated using external depth estimation or segmentation models

### Model Architecture

The control-conditioned models are based on Transfer2.5 multiview architecture with additional control conditioning layers. They:
- Accept 3-view synchronized input videos
- Incorporate control signals as additional conditioning
- Generate 3-view synchronized output videos
- Support context parallelism for efficient multi-GPU inference

### Text Prompts and Captions

**How Captions Work**

The inference system supports multiple ways to provide text prompts, with the following priority order:

1. **`prompt` field in JSON config** (highest priority): Applies the same prompt to all camera views
2. **Caption files**: Per-camera captions loaded from `{input_root}/captions/{video_id}_{camera}.txt`
3. **Camera prefix fallback** (lowest priority): Uses camera-specific descriptive prefixes

**Caption Prefixes**

When captions are loaded from files or when using the fallback, camera-specific prefixes can be automatically added:
- View 0 (head_color): "The video is captured from a camera mounted on the head of the subject, facing forward."
- View 1 (hand_left): "The video is captured from a camera mounted on the left hand of the subject."
- View 2 (hand_right): "The video is captured from a camera mounted on the right hand of the subject."

**Using Prompt Override in JSON**

You can provide a prompt override for all cameras using the `prompt` field:

```json
{
    "name": "my_example",
    "seed": 42,
    "guidance": 3.0,
    "prompt": "A robotic arm carefully manipulating delicate electronic components on a workbench under bright LED lighting",
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

**Recommendation**: For best quality results, always provide descriptive captions that capture:
- The scene context and environment
- Actions or movements being performed
- Lighting conditions and visual details
- Any important objects or interactions

## Related Documentation

- [Post-Training](post-training.md) - Fine-tuning on custom datasets
- [Troubleshooting](troubleshooting.md) - General troubleshooting guide
