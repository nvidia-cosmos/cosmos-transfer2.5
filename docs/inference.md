# Cosmos-Transfer2.5-2B: World Generation with Adaptive Multimodal Control
This guide provides instructions on running inference with Cosmos-Transfer2.5/general models.

![Architecture](../assets/Cosmos-Transfer2-2B-Arch.png)

### Pre-requisites
1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

### Hardware Requirements

The following table shows the GPU memory requirements for different Cosmos-Transfer2.5 models for single-GPU inference:

| Model | Required GPU VRAM |
|-------|-------------------|
| Cosmos-Transfer2.5-2B | 65.4 GB |

### Inference performance

#### Segmentation
The table below shows generation times(*) across different NVIDIA GPU hardware for single-GPU inference:

| GPU Hardware | Cosmos-Transfer2.5-2B 93 frame generation time | Cosmos-Transfer2.5-2B E2E time (**)|
|--------------|---------------|---------------|
| NVIDIA B200 | 92.25 sec | 186.92 |
| NVIDIA H100 NVL | 445.52 sec | 895.33 |
| NVIDIA H100 PCIe | 264.13 sec | 533.58 |
| NVIDIA H20 | 683.65 sec | 1370.39 |

\* Generation times are listed for 720P video with 16FPS with segmentation control input and disabled guardrails. \
\** E2E time is measured for input video with 121 frames, which results in two 93 frame "chunk" generations.

#### Edge
The table below compares base vs. distilled Transfer 2.5 Edge inference performance across GPU architectures.

| Metric | GPUs | RTX PRO 6000 Blackwell SE | H20 | H100 NVL | H200 NVL | B200 | B300 |
|--------|------|----------------------------|-----|----------|----------|------|------|
| **Avg. Base Model Diffusion Time (s)** | 1 | 78.5 | 176.4 | 64.5 | 49.8 | 24.2 | 53.2 |
| | 4 | 33.7 | 62.7 | 27.4 | 20.4 | 12.6 | 25.6 |
| | 8 | 25.0 | 44.0 | 20.4 | 16.9 | 11.1 | 19.9 |
| **Avg. Distilled Diffusion Time (s)** | 1 | 605.7 | 1374.6 | 502.6 | 374.4 | 179.7 | 415.5 |
| | 4 | 196.1 | 373.4 | 154.5 | 117.0 | 62.3 | 127.7 |
| | 8 | 118.8 | 201.5 | 92.5 | 82.4 | 41.8 | 76.1 |
| **Avg. Performance Improvement** | 1 | 7.7x | 7.8x | 7.8x | 7.5x | 7.4x | 7.8x |
| | 4 | 5.8x | 6.0x | 5.6x | 5.7x | 5.0x | 5.0x |
| | 8 | 4.7x | 4.6x | 4.5x | 4.9x | 3.8x | 3.8x |

## Inference with Pre-trained Cosmos-Transfer2.5 Models

**For more detailed guidance about the control modalities and examples, checkout our Cosmos Cookbook [Control-Modalities](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html) recipe.**

Individual control variants can be run on a single GPU:
```bash
python examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
```

For multi-GPU inference on a single control or to run multiple control variants, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):
```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
```

We provide example parameter files for each individual control variant along with a multi-control variant:

| Variant | Parameter File  |
| --- | --- |
| Depth | `assets/robot_example/depth/robot_depth_spec.json` |
| Edge | `assets/robot_example/edge/robot_edge_spec.json` |
| Segmentation | `assets/robot_example/seg/robot_seg_spec.json` |
| Blur | `assets/robot_example/vis/robot_vis_spec.json` |
| Multi-control | `assets/robot_example/multicontrol/robot_multicontrol_spec.json` |
| Distilled/Edge | `assets/robot_example/distilled/edge/robot_edge_spec.json`   |

For an explanation of all the available parameters run:
```bash
python examples/inference.py --help

python examples/inference.py control:edge --help # for information specific to edge control
```

Parameters can be specified as json:

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/robot_example/robot_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/robot_multicontrol",

    // Path to the input video
    "video_path": "assets/robot_example/robot_input.mp4",

    // Inference settings:
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // If a control is not provided, it will be computed on the fly.
        "control_path": "assets/robot_example/depth/robot_depth.mp4",

        // Control weight for the depth control
        "control_weight": 0.5
    },

    // Edge control settings
    "edge": {
        // Path to the control video
        "control_path": "assets/robot_example/edge/robot_edge.mp4",
        // Default control weight of 1.0 for edge control
    },

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/robot_example/seg/robot_seg.mp4",

        // Control weight for the seg control
        "control_weight": 1.0
    },

    // Blur control settings
    "vis":{
        // Control video computed on the fly
        "control_weight": 0.5
    }
}
```

If you would like the control inputs to only be used for some regions, you can define binary spatiotemporal masks with the corresponding control input modality in mp4 format. White pixels means the control will be used in that region, whereas black pixels will not. Example below:


```jsonc
{
    "depth": {
        "control_path": "assets/robot_example/depth/robot_depth.mp4",
        "mask_path": "/path/to/depth/mask.mp4",
        "control_weight": 0.5
    }
}
```

If you would like to run inference with distilled model, we need 2 changes on top of Transfer 2.5 inference: (1) specify the sampling steps `num_steps` in the JSON file (or `--num-steps` in the CLI), where the distilled model is trained with 4 sampling steps; (2) specify `--model=edge/distilled` in the inference command. Note that the distilled model is intended for short videos (strictly 93 sampled frames).

Example json file for edge distilled Transfer 2.5 model:
```jsonc
{
    "name": "robot_edge",
    "prompt_path": "/path/to/prompt/robot_prompt.txt",
    "video_path": "/path/to/input/robot_input.mp4",
    "guidance": 3,
    "num_steps": 4,
    "edge": {
        "control_path": "/path/to/edge/robot_edge.mp4",
        "control_weight": 1.0
    }
}
```

Example command to run edge distilled Transfer 2.5 inference:
```
# 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
    -i assets/robot_example/distilled/edge/robot_edge_spec.json \
    -o outputs/distilled/edge \
    --model=edge/distilled

# 1 GPU
python examples/inference.py \
    -i assets/robot_example/distilled/edge/robot_edge_spec.json \
    -o outputs/distilled/edge \
    --model=edge/distilled
```

## Outputs

### Multi-control

https://github.com/user-attachments/assets/337127b2-9c4e-4294-b82d-c89cdebfbe1d
