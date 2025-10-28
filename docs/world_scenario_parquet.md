# World Scenario Parquet File Structure

This document provides a comprehensive guide to the parquet file structure and requirements for generating world scenario videos using the `generate_control_videos.py` script.

## Overview

The world scenario generation pipeline transforms 3D scene annotations into control videos that condition the Cosmos Transfer auto multiview video generation model. These control videos provide spatial context through HD map visualizations, showing:

- Vehicle and pedestrian bounding boxes
- Lane lines and boundaries
- Traffic infrastructure (lights, signs)
- Road markings and crosswalks

The pipeline reads scene data from parquet files containing structured annotations and renders them from multiple camera perspectives to create synchronized control videos.

## File Organization

### Directory Structure
```
scene_annotations_directory/
├── {clip_id}.obstacle.parquet              (required)
├── {clip_id}.calibration_estimate.parquet  (required)
├── {clip_id}.egomotion_estimate.parquet    (required)
├── {clip_id}.lane.parquet                  (optional)
├── {clip_id}.lane_line.parquet             (optional)
├── {clip_id}.road_boundary.parquet         (optional)
├── {clip_id}.crosswalk.parquet             (optional)
├── {clip_id}.pole.parquet                  (optional)
├── {clip_id}.road_marking.parquet          (optional)
├── {clip_id}.wait_line.parquet             (optional)
├── {clip_id}.traffic_light.parquet         (optional)
├── {clip_id}.traffic_sign.parquet          (optional)
```

All files must share the same `clip_id` prefix, which represents a unique clip identifier.

## Coordinate Systems

The world scenario generation pipeline uses a consistent coordinate system across all data:

### World Coordinate System
- **Origin**: The ego vehicle's starting position (first frame) defines the world origin (0,0,0)
- **Convention**: Right-handed coordinate system with:
  - X: Forward (vehicle direction)
  - Y: Left
  - Z: Up
- Ego motion and obstacle positions use world coordinates

### Vehicle Reference Frame
- **Vehicle Origin**: Rear axle center at ground level (Z=0)
- **Camera positions**: Defined *relative* to the rear axle using FLU (Forward-Left-Up) convention
- **Transformations**: The pipeline chains transformations as world → ego → camera

### Temporal Alignment
- **Ego motion**: Typically provided at 10Hz, interpolated to processing frame rate (default: 30 fps, configurable via `INPUT_POSE_FPS` in `scripts/av_utils/render_config.py`)
- **Obstacles**: Typically provided at 10Hz, interpolated to same processing frame rate as ego motion
- **Synchronization**: The pipeline automatically interpolates obstacle tracks to align with ego motion timestamps

## Data Schema Notes

### Version Field
All parquet files include a `version` field (uint64) in each row. This field contains a data version number but is **not used by the rendering pipeline**. The rendering code processes all data regardless of version value. In practice, all rows within a scene typically share the same version number.

### Unused Fields
The following fields are present in the parquet schemas but are **not used by the current rendering pipeline**:
- **key fields**: Only `timestamp_micros` is used from the key structure. Fields like `clip_id`, `label_class_id`, `map_id`, and `map_id_version` are ignored.
- **category fields**: Traffic sign and road marking categories are loaded but not used for rendering (all rendered the same way).
- **lane fields**: `map_end`, `vehicle_types`, and `use_types` are not used.
- **wait_line fields**: `intersection_subtype` is not used.

## Required Files

### 1. Obstacle File (`{clip_id}.obstacle.parquet`)

Contains 3D bounding box annotations for dynamic objects (vehicles, pedestrians, cyclists).

**Schema:**
```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int,      # Timestamp in microseconds
        "label_class_id": str         # Annotation version (not used)
    },
    "obstacle": {
        "trackline_id": str,          # Trackline identity used to group different sensor sightings of the same obstacle together.
        "center": {                   # 3D position in world coordinates
            "x": float,               # meters
            "y": float,               # meters
            "z": float                # meters
        },
        "size": {                     # 3D dimensions
            "x": float,               # length (meters)
            "y": float,               # width (meters)
            "z": float                # height (meters)
        },
        "orientation": {              # Quaternion rotation (x,y,z,w)
            "x": float,
            "y": float,
            "z": float,
            "w": float
        },
        "category": str               # Object type: "automobile", "bus", "person", "rider", "stroller", "trailer"
    },
    "version": uint64                 # Data version number (not used by rendering pipeline)
}
```

**Category Mapping:**
- `automobile`, `other_vehicle`, `vehicle` → `Car`
- `person` → `Pedestrian`
- `rider` → `Cyclist`
- `bus`, `heavy_truck`, `train_or_tram_car`, `trolley_bus`, `trailer` → `Truck`
- All other categories (including `stroller`, `protruding_object`, `animal`) → `Others`

Note: The rendering pipeline uses case-insensitive matching.

### 2. Calibration Estimate File (`{clip_id}.calibration_estimate.parquet`)

Contains camera calibration and vehicle sensor configuration data.

**Schema:**
```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int       # Usually -1 (static calibration, not used)
    },
    "calibration_estimate": {
        "name": str,                  # Rig configuration name
        "rig_json": str               # JSON string containing full calibration
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**Rig JSON Structure:**
The `rig_json` field contains a serialized JSON with complete sensor configuration data. In the following sections we will detail the relevant parts. For additional information, please see [DriveWorks rig documentation](https://developer.nvidia.com/docs/drive/drive-os/6.0.8.1/public/driveworks-nvsdk/nvsdk_dw_html/rigconfiguration_usecase0.html).

### Camera Intrinsic Parameters (FTheta Model)

The system uses the **FTheta camera model**, which maps pixel distance from the optical center to ray angle. Each camera's intrinsics are stored in the `properties` field:

```jsonc
{
  "Model": "ftheta",                    // Camera model type
  "cx": "1919.420",                     // Principal point X (pixels)
  "cy": "1078.106",                     // Principal point Y (pixels)
  "width": "3848",                      // Image width (pixels)
  "height": "2168",                     // Image height (pixels)
  "polynomial": "0 0.000538 2.356e-09 1.907e-12 1.204e-16",  // Polynomial coefficients
  "polynomial-type": "pixeldistance-to-angle",  // Mapping direction
  "linear-c": "1.000000",               // Linear correction factor C
  "linear-d": "0.000000",               // Linear correction factor D
  "linear-e": "0.000000"                // Linear correction factor E
}
```

**Parameter Meanings:**
- **cx, cy**: Principal point (optical center) in pixel coordinates
- **width, height**: Image dimensions in pixels
- **polynomial**: Up to 6 coefficients [k0, k1, k2, k3, k4, k5] for the distortion model
  - For "pixeldistance-to-angle": θ = k0 + k1*r + k2*r² + k3*r³ + k4*r⁴ + k5*r⁵
  - Where r is the pixel distance from (cx, cy) and θ is the ray angle in radians
- **polynomial-type**:
  - `"pixeldistance-to-angle"`: Maps pixel radius to viewing angle (backward polynomial)
  - `"angle-to-pixeldistance"`: Maps viewing angle to pixel radius (forward polynomial)
- **linear-c, linear-d, linear-e**: Additional linear correction factors
  - Default values if not specified: c=1.0, d=0.0, e=0.0

### Camera Extrinsic Parameters

Each camera's pose relative to the vehicle is defined by the `nominalSensor2Rig_FLU` field:

```jsonc
{
  "nominalSensor2Rig_FLU": {
    "t": [1.697, -0.010, 1.443],        // Translation [x, y, z] in meters
    "roll-pitch-yaw": [0.130, 0.468, -0.714]  // Rotation in degrees
  }
}
```

**Coordinate System:**
- **FLU Convention**: Forward-Left-Up coordinate system
  - X: Forward (vehicle direction)
  - Y: Left
  - Z: Up
- **Vehicle Origin**: Rear axle center at ground level (Z=0)
- **Translation**: Camera position relative to the rear axle on the ground
- **Rotation**: Euler angles applied in roll-pitch-yaw order

### Calibration Corrections

The system may also include dynamic corrections applied during calibration:

```jsonc
{
  "correction_sensor_R_FLU": {
    "roll-pitch-yaw": [-0.150, -0.014, 0.231]   // Additional rotation correction
  },
  "correction_rig_T": [0.080, 0.009, 0.009]     // Additional translation correction
}
```

These corrections are applied on top of the nominal transforms to account for mounting tolerances and calibration refinements.

### Supported Cameras

The following camera configurations are typically available:
- `camera:front:wide:120fov` - Wide-angle front camera (120° FOV)
- `camera:front:tele:30fov` - Telephoto front camera (30° FOV)
- `camera:front:tele:sat:30fov` - Secondary telephoto front camera
- `camera:cross:left:120fov` - Left cross-traffic camera (120° FOV)
- `camera:cross:right:120fov` - Right cross-traffic camera (120° FOV)
- `camera:rear:left:70fov` - Rear left camera (70° FOV)
- `camera:rear:right:70fov` - Rear right camera (70° FOV)
- `camera:rear:tele:30fov` - Telephoto rear camera (30° FOV)

**Note on Camera Flexibility**: While these camera names are fixed in the system, you can provide custom intrinsic and extrinsic parameters for each camera. For example, a camera named `camera:front:wide:120fov` is not restricted to exactly 120° field of view - you can modify the polynomial coefficients, principal point, and camera pose to match your actual hardware. However, best results are expected when camera configurations don't deviate too far from the training data specifications.

### Modifying Calibration Data

To use your own camera calibration with the world scenario generation pipeline:

#### 1. Extract Template from Example Data

First, extract the rig JSON from an example calibration file:

```python
import pandas as pd
import json

# Load example calibration
df = pd.read_parquet('example.calibration_estimate.parquet')
rig_json = json.loads(df.iloc[0]['calibration_estimate']['rig_json'])

# Save to editable file
with open('my_rig.json', 'w') as f:
    json.dump(rig_json, f, indent=2)
```

#### 2. Locate Camera Parameters

Within the rig JSON, find your target camera in the `sensors` array. Each camera entry contains:

```jsonc
{
  "name": "camera:front:wide:120fov",
  "properties": {                    // Camera intrinsics go here
    "Model": "ftheta",
    "cx": "1919.420",
    "cy": "1078.106",
    "width": "3848",
    "height": "2168",
    "polynomial": "0 0.000538 ...",
    "polynomial-type": "pixeldistance-to-angle",
    "linear-c": "1.0",
    "linear-d": "0.0",
    "linear-e": "0.0"
  },
  "nominalSensor2Rig_FLU": {        // Camera extrinsics go here
    "t": [1.697, -0.010, 1.443],
    "roll-pitch-yaw": [0.130, 0.468, -0.714]
  },
  "correction_sensor_R_FLU": {...}, // Optional corrections (can be zeroed)
  "correction_rig_T": [...],        // Optional corrections (can be zeroed)
  // Other fields not used by rendering: car-mask, parameter, protocol
}
```

#### 3. Modify Calibration Values

**For intrinsics** (in `properties`):
- Update `cx`, `cy` with your principal point
- Update `width`, `height` with your image dimensions
- Replace `polynomial` coefficients with your distortion model coefficients
- Set `polynomial-type` to match your model:
  - `"pixeldistance-to-angle"` for backward polynomial (pixel → angle)
  - `"angle-to-pixeldistance"` for forward polynomial (angle → pixel)
- Set `linear-c`, `linear-d`, `linear-e` to your linear correction factors (defaults: 1.0, 0.0, 0.0)

**For extrinsics** (in `nominalSensor2Rig_FLU`):
- Update `t` with your camera position [x, y, z] in meters
- Update `roll-pitch-yaw` with your camera rotation in degrees

**To remove calibration corrections** (if starting fresh):
```jsonc
"correction_sensor_R_FLU": {"roll-pitch-yaw": [0.0, 0.0, 0.0]},
"correction_rig_T": [0.0, 0.0, 0.0]
```

#### 4. Create New Calibration Parquet

```python
# Create calibration dataframe
calibration_data = {
    'key': {'clip_id': 'your_clip_id', 'timestamp_micros': -1},
    'calibration_estimate': {
        'name': 'CUSTOM_RIG',
        'rig_json': json.dumps(rig_json)
    },
    'version': 1
}

df = pd.DataFrame([calibration_data])
df.to_parquet('your_clip_id.calibration_estimate.parquet')
```

#### 5. Important Notes


- **Minimal required structure**: For world scenario generation, you only need:
  - Camera entries in the `sensors` array with proper `name`, `properties`, and `nominalSensor2Rig_FLU`
  - The wrapping rig JSON structure
  - Other sensors (IMU, GPS, etc.) can be omitted if not needed

### 3. Egomotion Estimate File (`{clip_id}.egomotion_estimate.parquet`)

Contains vehicle trajectory data (position and orientation over time).

**Schema:**
```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int       # Timestamp in microseconds
    },
    "egomotion_estimate": {
        "name": str,                  # Estimation method (e.g., "egomotion_deepmap")
        "location": {                 # 3D position in world coordinates
            "x": float,               # meters
            "y": float,               # meters
            "z": float                # meters
        },
        "orientation": {              # Quaternion rotation (x,y,z,w)
            "x": float,
            "y": float,
            "z": float,
            "w": float
        }
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

## Optional Map Files

These files enhance the visualization with static map elements. Each follows a similar structure with different geometry types.

### Lane File (`{clip_id}.lane.parquet`)

Lane boundaries for driving lanes.

**Schema:**
```python
{
    "key": {
        "clip_id": str,               # (not used)
        "label_class_id": str,        # (not used)
        "map_id": str,                # (not used)
        "map_id_version": str         # (not used)
    },
    "lane": {
        "left_rail": [{"x": float, "y": float, "z": float}, ...],   # Polyline points
        "right_rail": [{"x": float, "y": float, "z": float}, ...],  # Polyline points
        "left_edge_styles": [str, ...],   # Style per segment: "VIRTUAL", "SOLID", "DASHED", "ROAD_BOUNDARY", etc.
        "right_edge_styles": [str, ...],
        "left_edge_colors": [str, ...],   # Color per segment: "WHITE", "YELLOW", "MIXED", etc.
        "right_edge_colors": [str, ...],
        "lane_direction": str,             # Optional: STRAIGHT, LEFT_TURN
        "speed_limit": float,              # Optional: speed limit (mph)
        "vehicle_types": [str, ...],       # Optional: allowed vehicle types
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

### Lane Line File (`{clip_id}.lane_line.parquet`)

Individual lane markings (center lines, edge lines).

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "lane_line": {
        "line_rail": [{"x": float, "y": float, "z": float}, ...],   # Polyline
        "styles": [str, ...],        # Per-point style: "SOLID_SINGLE", "DASHED_SINGLE", etc.
        "colors": [str, ...],        # Per-point color: "WHITE", "YELLOW"
        "left_driving_direction": str,    # Optional
        "right_driving_direction": str    # Optional
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**Style Types:**
- `SOLID_SINGLE`, `SOLID_DOUBLE`
- `DASHED_SINGLE`, `LONG_DASHED_SINGLE`
- `VIRTUAL` (not rendered)

### Road Boundary File (`{clip_id}.road_boundary.parquet`)

Physical road boundaries (curbs, barriers).

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "road_boundary": {
        "category": str,             # "tall_curb", "road_boundary", "barrier", "fence", "wall"
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polyline
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

### Crosswalk File (`{clip_id}.crosswalk.parquet`)

Pedestrian crossing areas.

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "crosswalk": {
        "category": str,             # "PEDESTRIAN"
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polygon vertices
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

### Traffic Light File (`{clip_id}.traffic_light.parquet`)

Traffic light positions and orientations.

**Note:** The current parquet format does NOT include traffic light state information (red/yellow/green). While the rendering code has infrastructure to support different traffic light colors based on state, all traffic lights are currently rendered in a uniform gray color ("unknown" state) because state data is not available in the parquet files.

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "traffic_light": {
        "center": {"x": float, "y": float, "z": float},
        "dimensions": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float},  # Quaternion
        "category": str              # "traffic_light"
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

### Traffic Sign File (`{clip_id}.traffic_sign.parquet`)

Road signs and information boards.

**Common categories include:**
- Regulatory signs: `TRAFFIC_SIGN_REGULATORY_R1_STOP`, `R1_YIELD`, `R2_SPEED_LIMIT`, `R3_NO_RIGHT_TURN`, etc.
- Warning signs: `TRAFFIC_SIGN_WARNING_W11_PEDESTRAIN_CROSSING`, `W3_STOP_AHEAD`, `W4_MERGE`, etc.
- School signs: `TRAFFIC_SIGN_SCHOOL_S1_SCHOOL`
- Information signs: `TRAFFIC_SIGN_INFORMATION`
- Destination/distance signs: `TRAFFIC_SIGN_DESTINATION_DISTANCE_D11_BIKE_ROUTE`

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "traffic_sign": {
        "center": {"x": float, "y": float, "z": float},
        "dimensions": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float},  # Quaternion
        "category": str              # "TRAFFIC_SIGN_INFORMATION", "TRAFFIC_SIGN_STOP", etc. (not used for rendering)
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

### Additional Map Elements

**Pole File** (`pole.parquet`): Vertical structures
- Example categories: `LIGHT`, `SIGN`, `TEL`, `TREE`, `SENTRY`, `UNKNOWN`
- Contains base and top positions as 2-point line segments

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file (no fields used)
    "pole": {
        "category": str,             # e.g., "LIGHT", "SIGN", "TREE" (not used for rendering)
        "location": [{"x": float, "y": float, "z": float}, ...]  # 2 points: base and top
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**Road Marking File** (`road_marking.parquet`): Painted road markings
- Example categories:
  - `ROI_POLYGON_ROAD_MARKING_STOP` - Stop line markings
  - `ROI_POLYGON_ROAD_MARKING_PED_XING` - Pedestrian crossing markings
  - `ROI_POLYGON_BIKE_PAINT` - Bike lane markings
  - `ROI_POLYGON_DIRECTION_PAINT_LEFT`, `_RIGHT`, `_FORWARD` - Directional arrows
  - `ROI_POLYGON_SPEED_LIMIT_PAINT_25` - Speed limit markings
  - `ROI_POLYGON_TEXT_PAINT` - General text markings
- Stored as polygons

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file (no fields used)
    "road_marking": {
        "category": str,             # e.g., "ROI_POLYGON_ROAD_MARKING_STOP" (not used for rendering)
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polygon vertices
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**Wait Line File** (`wait_line.parquet`): Stop lines at intersections

**Schema:**
```python
{
    "key": {...},  # Same structure as lane file
    "wait_line": {
        "category": str,             # "STOP", "UNKNOWN"
        "location": [{"x": float, "y": float, "z": float}, ...],  # Polyline
        "is_implicit": bool,         # Whether line is implicit or explicitly marked
        "intersection_subtype": str  # Whether this wait line is entry, exit or part of buffer zone at intersections or not applicable. (not used for rendering)
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```
