# Auto Multiview Inference Guide

## Prerequisites

1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

## Examples
Multiview requires 8 GPUs

Run multiview2world:

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/multiview.py -i assets/multiview_example/multiview_spec.json -o outputs/multiview/
```

For an explanation of all the available parameters run:
```bash
python examples/multiview.py --help

python examples/multiview.py control:view-config --help # for information specific to view configuration
```

Run autoregressive multiview (for generating longer videos):

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_autoregressive_spec.json -o outputs/multiview_autoregressive
```

## End to End Multiview Example

Complete workflow from 3D scene annotations to multiview video output. Scene annotations (object positions, camera calibration, vehicle trajectory) are rendered into world scenario videos that condition the multiview generation. This example uses only rendered control videos, not raw footage.

For full instructions, see [world_scenario_video_generation.md](world_scenario_video_generation.md).

**Step 1: Download scene annotations**
```bash
mkdir -p datasets && curl -Lf https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/3d_scene_metadata.zip -o temp.zip && unzip temp.zip -d datasets && rm temp.zip
```

**Step 2: Generate world scenario videos**
```bash
# See world_scenario_video_generation.md for detailed instructions
python scripts/generate_control_videos.py assets/multiview_example1/scene_annotations outputs/multiview_example1_world_scenario_videos
```

**Step 3: Run inference on the generated world scenario videos**
```bash
# Run inference (num_conditional_frames=0 since we're not using raw footage)
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example1/multiview_spec.json -o outputs/multiview_e2w/
```
