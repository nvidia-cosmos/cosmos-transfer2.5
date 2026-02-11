# Cosmos-Transfer2.5-2B: World Generation with Guided Generation
This guide provides instructions on running inference with guided generation. The Guided generation enables domain 
randomization by transferring simulation videos to realistic-looking footage while maintaining structural consistency 
through various control inputs, without any additional model training. Instead of allowing the model to 
freely reinterpret the entire scene, we encode simulation frames into the model’s latent space and apply spatial 
constraints during the denoising process. This selectively anchors important regions—such as surgical tools, robotic 
arms, or human phantoms—while leaving the rest of the scene unconstrained. As a result, the model can enhance 
global realism (lighting, textures, background complexity) while preserving the geometric structure and 
identity of critical foreground elements.

### Pre-requisites
Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

## Inference with Pre-trained Cosmos-Transfer2.5 Models

Individual control variants can be run on a single GPU:
```bash
python examples/inference.py -i assets/humanoid_example/seg/humanoid_seg_guided_spec.json -o outputs/seg_guided_generation
```

For multi-GPU inference on a single control or to run multiple control variants, use [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html):
```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/humanoid_example/seg/humanoid_seg_guided_spec.json -o outputs/seg_guided_generation
```

We provide example parameter files for each individual control variant:

| Variant | Parameter File  |
| --- | --- |
| Segmentation | `assets/humanoid_example/seg/humanoid_seg_guided_spec.json` |
| Depth | `assets/humanoid_example/depth/humanoid_depth_guided_spec.json` |
| Edge | `assets/humanoid_example/edge/humanoid_edge_guided_spec.json` |
| Blur | `assets/humanoid_example/vis/humanoid_vis_guided_spec.json` |

For an explanation of all the available parameters run:
```bash
python examples/inference.py --help

python examples/inference.py control:seg --help # for information specific to seg control
```

Parameters can be specified as json:

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/humanoid_example/humanoid_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/humanoid_multicontrol",

    // Path to the input video
    "video_path": "assets/humanoid_example/humanoid_input.mp4",

    // Path to the binary mask video indicating the foreground elements for guided generation
    "guided_generation_mask": "assets/humanoid_example/humanoid_guided_mask.mp4",

    // Number of steps for guided generation. Using more guidance steps provides stronger guidance of foreground 
    // elements in the generated videos. By default, 25 steps are used for guided generation.
    "guided_generation_step_threshold": 25,

    // Inference settings:
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // If a control is not provided, it will be computed on the fly.
        "control_path": "assets/humanoid_example/depth/humanoid_depth.mp4",

        // Control weight for the depth control
        "control_weight": 0.5
    },

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/humanoid_example/seg/humanoid_seg.mp4",

        // Control weight for the seg control
        "control_weight": 0.5
    },

    // Edge control settings
    "edge": {
        // Control video computed on the fly
        "control_weight": 0.0
    },

    // Blur control settings
    "vis":{
        // Control video computed on the fly
        "control_weight": 0.0
    }
}
```

### Example Input

<table>
  <tr>
    <th>Input Video</th>
    <th>Input Seg Control</th>
    <th>Guided Generation Mask</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/9fb7f172-97a9-4ab1-92e6-cf4ba1b6656f" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/55ebd525-6226-4dfd-9219-956e33f988e1" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/7caf916b-59ea-490e-a63a-484dd29cbdbe" width="100%" controls></video>
    </td>
  </tr>
</table>

### Example Output
<table>
  <tr>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/7ee3e5ff-1b36-47ee-99e2-45c77041f555" width="60%" controls></video>
    </td>
  </tr>
</table>

