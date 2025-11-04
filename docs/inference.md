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
    // REQUIRED: Name for the generation task. Outputs will include this name.
    "name": "robot_multicontrol_generation",

    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/robot_example/robot_prompt.txt",

    // Optional: Negative prompt to guide what to avoid in the generation
    "negative_prompt": "unrealistic, cartoonish, poor quality",

    // Path to the input video
    "video_path": "assets/robot_example/robot_input.mp4",

    // Inference settings
    "guidance": 3,              // Guidance scale (range: 0-7, default: 3)
    "seed": 2025,              // Random seed for reproducibility (default: 2025)
    "num_steps": 35,           // Number of denoising steps (default: 35)
    
    // Optional advanced settings
    "sigma_max": "70",         // Maximum sigma value for noise schedule. Max is 200.
    "show_control_condition": false, // Add controls (depth, edge, seg, vis) to the output video.

    // Depth control settings
    "depth": {
        // Path to the control video
        // Path to the control video (computed on the fly if not provided)
        "control_path": "assets/robot_example/depth/robot_depth.mp4",

        // Control weight for the depth control
        "control_weight": 0.5
    },

    // Edge control settings
    "edge": {
        // Path to the control video (computed on the fly if not provided)
        "control_path": "assets/robot_example/edge/robot_edge.mp4",

        // Control weight for the edge control
        "control_weight": 1.0,

        // Optional: Preset Canny edge detection threshold
        // Options: "very_low", "low", "medium", "high", "very_high" (default: "medium")
        "preset_edge_threshold": "medium"
    },

    // Segmentation control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/robot_example/seg/robot_seg.mp4",

        // Control weight for the segmentation control
        "control_weight": 1.0,

        // Optional: Text prompt to control what objects are tracked by on the fly segmentation
        "control_prompt": "robot"
    },

    // Blur/Visibility control settings
    "vis": {
        // Path to the control video (computed on the fly if not provided)
        "control_path": null,

        // Control weight for the blur control
        "control_weight": 0.5,

        // Optional: Preset blur strength
        // Options: "very_low", "low", "medium", "high", "very_high" (default: "medium")
        "preset_blur_strength": "medium"
    }
}
```

If you would like the control inputs to only be used for some regions, you can define binary spatiotemporal masks with the corresponding control input modality in mp4 format. White pixels means the control will be used in that region, whereas black pixels will not. You can provide masks in two ways:

1. **Video mask file** - Provide a binary video via `mask_path`:

```jsonc
{
    "depth": {
        "control_path": "assets/robot_example/depth/robot_depth.mp4",
        "mask_path": "/path/to/depth/mask.mp4",
        "control_weight": 0.5
    }
}
```

2. **Text-based mask** - Generate a mask dynamically via `mask_prompt`:

```jsonc
{
    "depth": {
        "control_path": "assets/robot_example/depth/robot_depth.mp4",
        "mask_prompt": "robot . table",
        "control_weight": 0.5
    }
}
```

## Outputs

### Multi-control

https://github.com/user-attachments/assets/337127b2-9c4e-4294-b82d-c89cdebfbe1d
