# Image Inference Guide
Cosmos-Transfer2.5 image inference runs our model on single frames or on control videos that use an image as a style reference. This guide covers the setup prerequisites, the image-to-image and style-reference workflow examples, relevant JSON parameters, and torchrun commands for multi-GPU scaling.

## Prerequisites

1. Follow the [Setup guide](setup.md) for environment setup, checkpoint download and hardware requirements.

## Image-to-Image

Transform a single image or video frame using control signals and text prompts:

```bash
python examples/inference.py -i assets/image_example/image2image.json -o outputs/image2image
```

## Image Prompt: using an image as a style reference

**For more detailed guidance and example about image prompting, checkout our Cosmos Cookbook [Style-Guided Inference](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-image/inference.html) recipe.**

Use an image as a style reference to guide video generation with a particular visual aesthetic.


```bash
python examples/inference.py -i assets/image_example/image_style.json -o outputs/image_style
```
Or use torchrun for multi-GPU inference:
```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/image_example/image_style.json -o outputs/image_style/
```

For an explanation of all the available parameters run:
```bash
python examples/inference.py --help

python examples/inference.py control:edge --help # for information specific to edge control
```
## Configuration

### Image-to-Image

```jsonc
{
    "name": "image_to_image",
    "prompt": "A scenic drive unfolds along a coastal highway...",

    // The input video. We'll extract the {max_frames} frames from the video.
    "video_path": "coastal_highway.mp4",
    "max_frames": 1,

    // Generate only the first frame
    "num_video_frames_per_chunk": 1,

    "seed": 1,
    "edge": {}  // Control computed on the fly
}
```

### Image Prompt

```jsonc
{
    "name": "image_style",
    "prompt": "The camera moves steadily forward...",

    // Input video that determines the control signals for the generation
    "video_path": "calm_street.mp4",

    // Reference image that determines the style of the generated video
    "image_context_path": "sunset.jpg",

    "seed": 1,
    "edge": {}
}
```

## Examples

### Image Prompt
<table>
  <tr>
    <th>Input Video</th>
    <th>Reference Image</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/3d7e8e93-b190-470c-8eb7-8b1ccf38904b" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <img src="https://github.com/user-attachments/assets/1c00a2c2-99a6-4fd7-a0bf-79333ac227c6" width="100%" alt="Reference image">
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/a925ec8e-6005-4e14-9c05-58597a9be61c" width="100%" controls></video>
    </td>
  </tr>
</table>
