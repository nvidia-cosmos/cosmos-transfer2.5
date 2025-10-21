# Cosmos-Transfer2-2B: World Generation with Adaptive Multimodal Control
This guide provides instructions on running inference with Cosmos-Transfer2.5/general models.

![Architecture](../assets/Cosmos-Transfer2-2B-Arch.png)

### Pre-requisites
1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

### Hardware Requirements

The following table shows the GPU memory requirements for different Cosmos-Transfer2 models for single-GPU inference:

| Model | Required GPU VRAM |
|-------|-------------------|
| Cosmos-Transfer2-2B | 65.4 GB |

### Inference performance

The following table shows generation times(*) across different NVIDIA GPU hardware for single-GPU inference:

| GPU Hardware | Cosmos-Transfer2-2B (Segmentation) |
|--------------|---------------|
| NVIDIA B200 | 285.83 sec |
| NVIDIA H100 NVL | 719.4 sec |
| NVIDIA H100 PCIe | 870.3 sec |
| NVIDIA H20 | 2326.6 sec |

\* Generation times are listed for 720P video with 16FPS for 5 seconds length (93 frames) with segmentation control input.

## Inference with Pre-trained Cosmos-Transfer2 Models

Individual control variants can be run on a single GPU:
```bash
python examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
```

For multi-GPU inference on a single control or to run multiple control variants, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):
```bash
torchrun --nproc_per_node=8 --master_port=12341 -m examples.inference -i assets/multicontrol.jsonl -o outputs/multicontrol
```

We provide example parameter files for each individual control variant along with a multi-control variant:

| Variant | Parameter File  |
| --- | --- |
| Depth | `assets/robot_example/depth/robot_depth_spec.json` |
| Edge | `assets/robot_example/edge/robot_edge_spec.json` |
| Segmentation | `assets/robot_example/seg/robot_seg_spec.json` |
| Blur | `assets/robot_example/vis/robot_vis_spec.json` |
| Multi-control | `assets/robot_example/multicontrol/robot_multicontrol_spec.json` |

Parameters can be specified as json:

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/robot_example/robot_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/robot_multicontrol",

    // Path to the input video
    "video_path": "assets/robot_example/robot_input.mp4",

    // Inference settings
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // For "vis" and "edge", if a control is not provided, it will be computed on the fly.
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

## Outputs

### Multi-control

https://github.com/user-attachments/assets/337127b2-9c4e-4294-b82d-c89cdebfbe1d
