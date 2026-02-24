# Frame Extraction and Annotation Projection Documentation

This document describes the frame extraction and annotation projection scripts for 3D bounding box visualization.

## Overview

Two main scripts work together to extract frames with 3D annotations and project them onto images:

1. **`local_extract_frames.py`**: Extracts frames from video with 3D bounding box annotations
2. **`local_project_annotations.py`**: Projects 3D annotations onto images using FThetaCamera

## Scripts

### 1. `local_extract_frames.py`

Extracts individual frames from camera videos with 3D bounding box annotations in JSON format.

#### Purpose

- Extract frames from camera videos
- Generate 3D bounding box annotations for visible objects
- Save camera poses for each frame
- Optionally overlay HD map renderings on frames

#### Key Features

- **Frustum Culling**: Only includes objects visible in camera view
- **Occlusion Filtering**: Filters out fully occluded objects
- **Distance Filtering**: Excludes objects beyond 150m from camera
- **Coordinate Conversion**: Converts FLU world coordinates to RDF camera coordinates for annotations
- **Camera Pose Storage**: Saves camera-to-world transformation matrices

#### Usage

```bash
# Extract frames with annotations (every 30th frame)
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    /path/to/data \
    --camera-names camera_front_wide_120fov \
    --extract-frames \
    --skip-frames 30 \
    --output-dir output/frames

# Extract with HD map overlay
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    /path/to/data \
    --camera-names camera_front_wide_120fov \
    --extract-frames \
    --overlay-camera \
    --alpha 0.5 \
    --skip-frames 30

# Process from S3
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    --s3-input s3://bucket/input/path \
    --s3-output s3://bucket/output/path \
    --extract-frames \
    --skip-frames 30
```

#### Output Structure

```
output_dir/
├── images/
│   └── <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.jpg
├── text/
│   └── <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.json
├── camera_poses/
│   └── <uuid>_<start_ts>_<end_ts>_<camera_name>_frame_<frame_id>.npy
└── meta.json
```

#### Annotation JSON Format

```json
{
  "frame_id": 0,
  "camera": "camera_front_wide_120fov",
  "camera_params": {
    "fx": 640.0,
    "fy": 640.0,
    "cx": 640.0,
    "cy": 360.0
  },
  "camera_pose": [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
  ],
  "annotations": [
    {
      "label": "vehicle_123",
      "bbox_3d": [
        2.5,    // x_center (Right, meters)
        0.0,    // y_center (Down, meters)
        10.0,   // z_center (Forward, meters)
        4.5,    // x_size (width, meters)
        2.0,    // y_size (height, meters)
        1.5,    // z_size (length, meters)
        0.0,    // roll (radians)
        0.0,    // pitch (radians)
        0.1     // yaw (radians)
      ]
    }
  ]
}
```

#### Coordinate Systems

**Annotation Format (RDF Camera Coordinates)**:
- **X**: Right (positive X is to the right of camera)
- **Y**: Down (positive Y is downward)
- **Z**: Forward (positive Z is forward, in front of camera)

**Camera Pose (Camera-to-World Transformation)**:
- 4x4 homogeneous transformation matrix
- Transforms points from RDF camera coordinates to FLU world coordinates
- Format: `[R|t]` where R is 3x3 rotation, t is 3x1 translation

#### Key Functions

- `extract_frames_with_annotations()`: Main extraction function
- `extract_3d_annotations()`: Extracts 3D bbox annotations for a frame
- `world_to_camera_coordinates()`: Converts FLU world → RDF camera coordinates
- `is_bbox_in_camera_view()`: Checks if bbox is visible in camera
- `project_bbox_to_2d()`: Projects 3D bbox to 2D pixel bounds

---

### 2. `local_project_annotations.py`

Projects 3D bounding box annotations onto images using FThetaCamera projection.

#### Purpose

- Load scene data from input directory
- Read 3D annotations from extracted JSON files
- Convert RDF camera coordinates to FLU world coordinates
- Project 3D boxes onto images using FThetaCamera
- Draw projected boxes on images

#### Key Features

- **Coordinate Conversion**: Converts RDF camera → FLU world → RDF camera for projection
- **FThetaCamera Projection**: Uses `ray2pixel_np()` for accurate fisheye projection
- **Wireframe Rendering**: Draws 12 edges of 3D bounding boxes
- **Scene Data Integration**: Can load original scene data for validation

#### Usage

```bash
# Basic usage
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
    --annotations-dir cosmos3/dataset/output/mads-sample-03 \
    --input-dir cosmos3/dataset/test/mads-sample \
    --output-dir cosmos3/dataset/output/mads-sample-03/images_annotated
```

#### Input Requirements

- **annotations-dir**: Directory containing:
  - `images/`: Extracted frame images
  - `text/`: JSON annotation files
  - `camera_poses/`: (Optional) Camera pose .npy files
- **input-dir**: Original scene data directory (for loading camera models)

#### Output

- Annotated images with 3D bounding boxes drawn as wireframes
- Green lines connecting 8 corners of each 3D bounding box

#### Coordinate System Flow

1. **Load Annotation**: `bbox_3d` in RDF camera coordinates
2. **Convert to FLU**: `camera_to_world_coordinates()` → FLU world coordinates
3. **Get Corners**: `get_bbox_corners_3d()` → 8 corners in FLU world coordinates
4. **Transform to RDF**: `world_to_camera = inv(camera_pose)` → RDF camera coordinates
5. **Project**: `FThetaCamera.ray2pixel_np()` → 2D pixel coordinates
6. **Draw**: `draw_bbox_on_image()` → Draws 12 edges on image

#### Key Functions

- `camera_to_world_coordinates()`: Converts RDF camera → FLU world coordinates
- `get_bbox_corners_3d()`: Computes 8 corners of 3D bounding box
- `project_bbox_corners_to_image()`: Projects corners using FThetaCamera
- `draw_bbox_on_image()`: Draws wireframe on image using OpenCV

---

## Coordinate System Reference

### FLU World Coordinates (SceneData Standard)

- **X**: Forward (positive X is forward)
- **Y**: Left (positive Y is to the left)
- **Z**: Up (positive Z is upward)
- **Origin**: Ego vehicle's starting position (first frame)

### RDF Camera Coordinates (FThetaCamera Standard)

- **X**: Right (positive X is to the right of camera)
- **Y**: Down (positive Y is downward)
- **Z**: Forward (positive Z is forward, in front of camera)
- **Origin**: Camera optical center

### Transformations

**World → Camera**:
```python
world_to_camera = inv(camera_pose)  # camera_pose is camera-to-world
camera_coords = world_to_camera @ [world_coords, 1.0]
```

**Camera → World**:
```python
world_coords = camera_pose @ [camera_coords, 1.0]
```

---

## Workflow Examples

### Complete Pipeline: Extract → Project

```bash
# Step 1: Extract frames with annotations
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    /path/to/data \
    --camera-names camera_front_wide_120fov \
    --extract-frames \
    --skip-frames 30 \
    --output-dir output/frames

# Step 2: Project annotations onto images
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
    --annotations-dir output/frames \
    --input-dir /path/to/data \
    --output-dir output/frames/images_annotated
```

### Validation Workflow

```bash
# Extract frames with annotations
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_extract_frames.py \
    /path/to/data \
    --camera-names camera_front_wide_120fov \
    --extract-frames \
    --skip-frames 30 \
    --output-dir output/frames

# Project annotations to verify correctness
uv run python cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/scripts/local_project_annotations.py \
    --annotations-dir output/frames \
    --input-dir /path/to/data \
    --output-dir output/frames/images_annotated

# Compare original images with projected annotations
```

---

## Common Issues and Solutions

### Issue: Annotations not visible in projected images

**Possible Causes**:
1. Objects behind camera (z <= 0 in RDF)
2. Objects outside camera FOV
3. Incorrect camera pose
4. Coordinate system mismatch

**Solutions**:
- Check camera pose is camera-to-world (not world-to-camera)
- Verify objects are in front of camera (z > 0)
- Ensure annotations are in RDF camera coordinates
- Check FThetaCamera model matches image dimensions

### Issue: Boxes appear in wrong location

**Possible Causes**:
1. Camera pose incorrect
2. Coordinate system conversion error
3. Camera model mismatch

**Solutions**:
- Verify camera_pose is 4x4 camera-to-world matrix
- Check coordinate system conversions (RDF ↔ FLU)
- Ensure camera model intrinsics match image dimensions

### Issue: Missing annotations

**Possible Causes**:
1. Objects filtered by frustum culling
2. Objects occluded by other objects
3. Objects beyond distance threshold (150m)

**Solutions**:
- Check `is_bbox_in_camera_view()` filtering logic
- Verify occlusion filtering settings
- Adjust distance threshold if needed

---

## Integration with Other Scripts

### Using with `local_render_frames.py`

The `local_render_frames.py` script uses the same coordinate system approach:

1. Works in FLU world coordinates (SceneData)
2. Converts to RDF camera coordinates for projection
3. Uses FThetaCamera for projection

This ensures consistency across all rendering scripts.

### Using with SceneData

Both scripts integrate with `SceneData` structure:

- `local_extract_frames.py`: Reads from SceneData, outputs annotations
- `local_project_annotations.py`: Can load SceneData for validation
- Both use FLU world coordinates as the primary representation

---

## Performance Considerations

### `local_extract_frames.py`

- **GPU Acceleration**: Uses GPU for rendering (OpenGL)
- **Batch Processing**: Processes multiple cameras efficiently
- **Memory**: Loads videos into memory for fast access
- **Optimization**: Uses persistent VBOs for static geometry

### `local_project_annotations.py`

- **CPU Processing**: Uses NumPy for coordinate transformations
- **Single Frame**: Processes one frame at a time
- **Memory**: Loads images individually to reduce memory usage
- **Optimization**: Batches corner computations

---

## File Format Reference

### Camera Pose (.npy)

- **Format**: NumPy array, shape `(4, 4)`, dtype `float32`
- **Type**: Camera-to-world transformation matrix
- **Structure**: `[R|t]` where R is 3x3 rotation, t is 3x1 translation

### Annotation JSON

- **Format**: JSON file with frame metadata and annotations
- **bbox_3d**: Array of 9 floats `[x, y, z, x_size, y_size, z_size, roll, pitch, yaw]`
- **Coordinate System**: RDF camera coordinates

### Meta JSON

- **Format**: JSON array of metadata entries
- **Structure**: `[{"id": uuid, "conversation": "text/...", "media": "images/..."}, ...]`
- **Purpose**: Links images with their annotations

---

## Additional Resources

- **Coordinate Systems**: See `OVERLAY_REQUIREMENTS.md` for detailed coordinate system documentation
- **Rendering**: See `local_render_frames.py` for overlay rendering implementation
- **SceneData**: See `data_types.py` for SceneData structure documentation
