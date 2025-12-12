# World Scenario Video Generation

This tool generates control videos from 3D scene annotations for Cosmos Transfer2.5. It renders world models into videos by projecting 3D elements (polylines, polygons, and cuboids) onto camera views.

**Supported Input Formats:**
- **Parquet format**: Structured scene annotations in parquet files (see [World Scenario Parquet File Structure](world_scenario_parquet.md))
- **RDS-HQ format**: NVIDIA's internal format from the [Cosmos-Drive-Dreams dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)

## Table of Contents

<!--TOC-->

- [Table of Contents](#table-of-contents)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Setup](#setup)
  - [Quick Start](#quick-start)
  - [Example 1: Parquet Input](#example-1-parquet-input)
  - [Example 2: RDS-HQ Input](#example-2-rds-hq-input)
- [Usage](#usage)
  - [Basic Commands](#basic-commands)
  - [Options](#options)
  - [Available Cameras](#available-cameras)
- [Data Format](#data-format)
  - [Input Structure](#input-structure)
  - [Output Structure](#output-structure)
- [What Gets Rendered](#what-gets-rendered)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
- [Integration](#integration)
- [Rendering Specifications](#rendering-specifications)
  - [Color Palette](#color-palette)
  - [Dynamic Objects](#dynamic-objects)
  - [Lanelines](#lanelines)
  - [Traffic Lights](#traffic-lights)
  - [Map Elements](#map-elements)
- [Pipeline Overview](#pipeline-overview)
  - [Frame Rate Configuration](#frame-rate-configuration)
  - [1. Interpolation](#1-interpolation)
  - [2. Coordinate Systems](#2-coordinate-systems)
  - [3. Rendering Process](#3-rendering-process)

<!--TOC-->

---

## Getting Started

### Requirements

- Python 3.10+
- UV (for dependency management)
- GPU with EGL support (for headless OpenGL rendering)
- 3D scene annotation data in Parquet or RDS-HQ format

### Setup

First, set up your environment:

```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Quick Start

Generate world scenario videos from your 3D scene annotations:

```bash
python scripts/generate_control_videos.py -i /path/to/input_root -o /path/to/save_root
```

The script automatically detects whether your input is in Parquet or RDS-HQ format.

### Example 1: Parquet Input

Here's a full working example you can copy and paste to try it out:

```bash
# Download example data
wget -P assets https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example1.zip && unzip -oq assets/multiview_example1.zip -d assets

# Generate control videos for the example scene
python scripts/generate_control_videos.py -i assets/multiview_example1/scene_annotations -o outputs/multiview_example1_world_scenario_videos
```

Additional examples available for download:
```bash
wget https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example2.zip
wget https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example3.zip
```

### Example 2: RDS-HQ Input

Using data from the [Cosmos-Drive-Dreams dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) (see dataset documentation for RDS-HQ format details):

```bash
wget -P scripts https://raw.githubusercontent.com/nv-tlabs/Cosmos-Drive-Dreams/main/scripts/download.py
python scripts/download.py --odir ./assets/rdshq-data --limit 1
python scripts/generate_control_videos.py -i assets/rdshq-data -o outputs/rdshq-generated
```

## Usage

### Basic Commands

```bash
# All cameras (default)
python scripts/generate_control_videos.py -i {input_root}/ -o {save_root}/

# Specific cameras
python scripts/generate_control_videos.py -i {input_root}/ -o {save_root}/ \
    --cameras "camera:front:wide:120fov,camera:cross:right:120fov"
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--cameras` | `all` | Camera names or "all" for all 7 cameras |

### Available Cameras

- `camera:front:wide:120fov`
- `camera:front:tele:sat:30fov`
- `camera:cross:right:120fov`
- `camera:cross:left:120fov`
- `camera:rear:left:70fov`
- `camera:rear:right:70fov`
- `camera:rear:tele:30fov`

## Data Format

### Input Structure

The tool accepts two input formats:

**1. Parquet Format** - Structured scene annotations in parquet files:
```
scene_annotations_directory/
├── uuid.obstacle.parquet              (required)
├── uuid.calibration_estimate.parquet  (required)
├── uuid.egomotion_estimate.parquet    (required)
├── uuid.lane.parquet                  (optional)
├── uuid.lane_line.parquet             (optional)
└── ... (other optional parquet files)
```

See [World Scenario Parquet File Structure](world_scenario_parquet.md) for detailed schema and requirements.

**2. RDS-HQ Format** - NVIDIA's recording format containing sensor data and annotations. The script automatically extracts the required scene information. See the [Cosmos-Drive-Dreams dataset documentation](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams) for format details.

### Output Structure

Both input formats produce the same output structure:
```
save_root/
└── uuid/
    ├── uuid.camera_front_wide_120fov.mp4
    ├── uuid.camera_front_tele_sat_30fov.mp4
    ├── uuid.camera_cross_right_120fov.mp4
    ├── uuid.camera_cross_left_120fov.mp4
    ├── uuid.camera_rear_left_70fov.mp4
    ├── uuid.camera_rear_right_70fov.mp4
    ├── uuid.camera_rear_tele_30fov.mp4
```

## What Gets Rendered

**Always rendered:** 3D bounding boxes for vehicles, pedestrians, and other dynamic objects

**Optionally rendered** (when data available):
- Lane lines, lanes, and road boundaries
- Crosswalks, road markings, and wait lines
- Poles, traffic lights, and traffic signs

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **ModernGL/EGL errors** | Install GPU drivers and EGL libraries. Ubuntu/Debian: `apt install libegl1-mesa-dev libgl1-mesa-dri` |
| **Missing parquet files** | Ensure required files exist: `obstacle`, `calibration_estimate`, `egomotion_estimate` |
| **Memory issues** | Reduce the number of cameras processed simultaneously using `--cameras` |
| **Invalid camera names** | Run with `--help` to see valid options |

## Integration

The generated control videos serve as conditioning inputs for Cosmos Transfer2.5 model inference, providing spatial context through HD map visualizations for controlled video generation.

## Rendering Specifications

This rendering specification builds on [Cosmos Drive Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams).

<p align="center">
  <img src="../assets/docs/world_scenario_image.png" alt="World Scenario Image" width="800">
</p>

<p align="center">
  <img src="../assets/docs/rendering_diagram.png" alt="Rendering Diagram" width="800">
</p>

### Color Palette

<p align="center">
  <img src="../assets/docs/color_palette.png" alt="Color Palette" width="800">
</p>

### Dynamic Objects
**Config:** [`config_color_bbox.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_bbox.json)

Dynamic objects are rendered as solid 3D cuboids with light gray edges and front-to-back color gradients.

**Label Mapping:** [`bbox_utils.py`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/utils/bbox_utils.py) maps DCP labels to five categories:
1. **Car**: automobile, other_vehicle, vehicle
2. **Truck**: heavy_truck, bus, train_or_tram_car, trailer
3. **Pedestrian**: person
4. **Cyclist**: rider
5. **Others**: protruding_object, animal, stroller

<p align="center">
  <img src="../assets/docs/dynamic_objects.png" alt="Dynamic Objects" width="800">
</p>

### Lanelines
**Config:** [`config_color_geometry_laneline.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_geometry_laneline.json)

Lanelines are categorized into 15 types based on:
- **Colors:** yellow, white, other
- **Styles:** solid, dashed, dotted, solid-dashed combinations

**Example:** `yellow solid dashed` means a yellow solid line (right) + yellow dashed line (left) in the polyline direction.

<p align="center">
  <img src="../assets/docs/lanelines.png" alt="Lanelines" width="800">
</p>

### Traffic Lights
**Config:** [`config_color_traffic_light.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_traffic_light.json)

Traffic lights are rendered as cuboids with four states: Red, Yellow, Green, Unknown.

<p align="center">
  <img src="../assets/docs/traffic_lights.png" alt="Traffic Lights" width="800">
</p>

### Map Elements
**Config:** [`config_color_hdmap.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_hdmap.json)

Map elements use three geometry types:
- **Polylines**: poles, road boundaries, wait lines
- **Polygons**: crosswalks, road markings
- **Cuboids**: traffic signs

<p align="center">
  <img src="../assets/docs/map_elements.png" alt="Map Elements" width="800">
</p>

## Pipeline Overview

### Frame Rate Configuration

The pipeline uses two configurable frame rates:
- **`INPUT_POSE_FPS`** (default: 30fps): Processing frame rate for interpolation - determines how many frames are generated
- **`TARGET_RENDER_FPS`** (default: 30fps): Output video playback frame rate - determines playback speed

Source data is typically at 10Hz and is interpolated to the processing frame rate.

### 1. Interpolation
- **Obstacle data**: Interpolated from source frequency (typically 10Hz) to processing frame rate
- **Egomotion data**: Interpolated to same processing frame rate
- **Map data**: Static elements, loaded once without interpolation

The processing frame rate (`INPUT_POSE_FPS`) determines temporal resolution, while output frame rate (`TARGET_RENDER_FPS`) controls playback speed. Both typically default to 30fps.

### 2. Coordinate Systems
- **World coordinates**: Right-handed system (x=forward, y=left, z=up)
- **Camera coordinates**: Camera coordinate system where the camera looks along the positive z-axis
  - x-axis: right
  - y-axis: down
  - z-axis: forward (into scene)
- **FLU convention**: Forward-Left-Up used for vehicle-to-camera transforms

### 3. Rendering Process
1. Load camera calibration
2. Parse and interpolate egomotion trajectory to processing frame rate
3. Interpolate obstacle tracks to match egomotion timestamps
4. Transform all geometries from world to camera coordinates
5. Render each frame using OpenGL
6. Encode output as MP4 video
