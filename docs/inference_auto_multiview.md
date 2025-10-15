# Auto Multiview Inference Guide

## Prerequisites

1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

## Examples
Multiview requires **8 GPUs**.

Run multiview2world:

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview --params_file assets/multiview_example/multiview_spec.json --num_gpus=$NUM_GPUS
```

## End to End Multiview Example

Complete workflow from 3D scene annotations to multiview video output. Scene annotations (object positions, camera calibration, vehicle trajectory) are rendered into world scenario videos that condition the multiview generation. This example uses only rendered control videos, not raw footage.

**Step 1: Download scene annotations**
```bash
mkdir -p datasets && curl -Lf https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/3d_scene_metadata.zip | tar -xz -C datasets
```

**Step 2: Generate world scenario videos**
```bash
# See world_scenario_video_generation.md for detailed instructions
python scripts/generate_control_videos.py datasets/3d_scene_metadata assets/multiview_example1/world_scenario_videos
```

For full instructions, see [world_scenario_video_generation.md](world_scenario_video_generation.md).

**Step 3: Run multiview inference**
**Key difference:** Set { "num_conditional_frames": 0 } in the params JSON file.
```bash

# Run inference (num_conditional_frames=0 since we're not using raw footage)
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview --params_file assets/multiview_example/multiview_spec.json --num_gpus=$NUM_GPUS
```
