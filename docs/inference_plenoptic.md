# Plenoptic Multiview Inference Guide

This guide explains how to run plenoptic multiview camera inference to generate novel camera views from a single input video.

## Prerequisites

1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download, and hardware requirements.

## Overview

Plenoptic multiview inference generates videos from novel camera perspectives based on a single input video. The model uses autoregressive generation to progressively create new camera views while conditioning on previously generated views.

### Supported Camera Motions

The following camera motion types are supported:

| Motion Type | Description |
|-------------|-------------|
| `static` | No camera movement (input view) |
| `rot_left` / `rot_right` | Rotation around vertical axis |
| `arc_left` / `arc_right` | Circular arc movement |
| `azimuth_left` / `azimuth_right` | Azimuthal rotation |
| `tilt_up` / `tilt_down` | Vertical tilt |
| `translate_up_rot` / `translate_down_rot` | Translation with rotation |
| `elevation_up_1` / `elevation_up_2` | Elevation changes |
| `zoom_in` / `zoom_out` | Zoom in/out |
| `distance_away_1` / `distance_away_2` | Distance from subject |

## Input Format

Create a JSON file with the following structure:

```json
{
    "name": "sample_001",
    "input_path": "videos/input.mp4",
    "prompt": "A scenic outdoor landscape with mountains and trees",
    "camera_sequence": ["static", "rot_left", "arc_right", "azimuth_right"],
    "seed": 1,
    "guidance": 7
}
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Unique identifier for this sample |
| `input_path` | string | Path to input video (relative to base_path) |
| `prompt` | string | Text description of the scene |
| `camera_sequence` | list[string] | Ordered list of camera motions to generate |
| `seed` | int | Random seed for reproducibility (default: 0) |
| `guidance` | int | Guidance scale 0-7 (default: 7) |

### CLI Override Parameters

The following parameters can be overridden via command line arguments:

| Parameter | Type | Description |
|-----------|------|-------------|
| `--focal_length` | int | Focal length for camera intrinsics in mm (default: 24) |
| `--extrinsic_scale` | float | Scale factor for camera extrinsics (default: 1.5) |
| `--input_video_res` | string | Input video resolution (only `480p` supported) |
| `--guidance` | int | Override guidance scale for all samples |
| `--seed` | int | Override random seed for all samples |

### Camera Parameter Ranges

When customizing camera parameters, the following ranges have been validated in our tests:

| Parameter | Min | Max | Description |
|-----------|-----|-----|-------------|
| **Focal Length (Intrinsics)** | 12 | 100 | Camera focal length in mm (default: 24). Values outside this range may produce artifacts. |
| **Extrinsic Scale** | 0.5 | 3.0 | Scaling factor applied to camera translation and rotation. Controls the magnitude of camera motion relative to the input templates. |

> **Note:** The `focal_length` parameter requires a corresponding `intrinsics_focal{focal_length}.txt` file in the `cameras/` directory. The default assets provide `intrinsics_focal24.txt` and `intrinsics_focal50.txt`. For custom focal lengths, you'll need to create the appropriate intrinsics file.

## Examples

### Single GPU Inference

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 python examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic/ \
    --base-path assets/plenoptic_example/
```

### Multi-GPU Inference (8 GPUs)

For best performance, use 8 GPUs with context parallelism (automatically detected from `torchrun`):

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1  torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic/ \
    --base-path assets/plenoptic_example/
```

### Batch Inference

Process multiple samples using a JSONL file:

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/batch.jsonl \
    -o outputs/plenoptic_batch/ \
    --base-path assets/plenoptic_example/
```

### Custom Parameters

Override default parameters via CLI:

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic_custom/ \
    --base-path assets/plenoptic_example/ \
    --guidance=5 \
    --seed=42 \
    --focal_length=24
```

## Data Preparation

### Directory Structure

Your data directory should have the following structure:

```
base_path/
├── videos/
│   ├── input1.mp4
│   └── input2.mp4
├── cameras/
│   ├── static.txt (or static.pt)
│   ├── rot_left.txt
│   ├── rot_right.txt
│   ├── arc_left.txt
│   ├── arc_right.txt
│   ├── azimuth_left.txt
│   ├── azimuth_right.txt
│   ├── tilt_up.txt
│   ├── tilt_down.txt
│   ├── translate_up_rot.txt
│   ├── translate_down_rot.txt
│   ├── elevation_up_1.txt
│   ├── elevation_up_2.txt
│   ├── zoom_in.txt
│   ├── zoom_out.txt
│   ├── distance_away_1.txt
│   ├── distance_away_2.txt
│   ├── intrinsics_focal24.txt
│   └── intrinsics_focal50.txt
├── metadata.csv (optional)
└── sample.json
```

### Camera File Formats

Camera extrinsics can be provided in two formats (`.pt` or `.txt`). The code tries `.pt` first, then falls back to `.txt`.

**Extrinsics (`.txt` format):** Each line contains a flattened 3×4 camera pose matrix (12 values) for one frame:
```
R11 R12 R13 Tx R21 R22 R23 Ty R31 R32 R33 Tz
```

For 24 latent frames, the file should have 24 lines. Example for `static.txt` (identity pose, no movement):
```
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
... (24 lines total)
```

**Extrinsics (`.pt` format):** A PyTorch tensor of shape `[24, 4, 4]` containing camera pose matrices.

**Intrinsics (`intrinsics_focal*.txt`):** A single line with 4 values:
```
fx fy cx cy
```
Where `fx`, `fy` are focal lengths and `cx`, `cy` are principal point coordinates.

### Creating Custom Camera Trajectories

The provided example assets in `assets/plenoptic_example/cameras/` contain pre-defined camera trajectories. To create custom trajectories:

1. **Use the provided examples as templates** - copy and modify the values
2. **For rotation trajectories** - gradually change the rotation values (R11, R13, R31, R33) across frames
3. **For translation trajectories** - gradually change the translation values (Tx, Ty, Tz) across frames

See the example files in `assets/plenoptic_example/cameras/` for reference trajectories.

## Sample Outputs

The following examples show the input video and generated output videos for different camera motions:

<table>
  <tr>
    <th>Camera Motion</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <code>rot_left</code><br>Rotation around vertical axis (left)
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/de384ce4-b58e-4f86-82c3-3111e466689f" width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <code>rot_right</code><br>Rotation around vertical axis (right)
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/8f65a028-b8f2-4533-a466-aabf841ab3eb" width="100%" controls></video>
    </td>
  </tr>
</table>

### Getting Help

For more information on available parameters:

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 python examples/plenoptic.py --help
```

## Related Documentation

- [Setup Guide](setup.md)
- [Troubleshooting](troubleshooting.md)
- [Auto Multiview Inference](inference_auto_multiview.md)
