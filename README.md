<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos/"> Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B">Hugging Face</a>&nbsp | <a href="https://arxiv.org/abs/2511.00062">Paper</a> | <a href="https://research.nvidia.com/labs/dir/cosmos-transfer2.5/">Paper Website</a> | <a href="https://github.com/nvidia-cosmos/cosmos-cookbook">Cosmos Cookbook</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as autonomous vehicles (AVs), robots, and video analytics AI agents.

Cosmos World Foundation Models come in three model types which can all be customized in post-training: [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1).

## News
* [February 23, 2026] Released Transfer2.5 Distilled Edge [model](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/distilled/general/edge) and [inference](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/inference.md), enabling low latency (edge deployment) inference. More distilled controlnets coming soon.
* [December 19, 2025] Released Image2Image and ImagePrompt capabilities. See the inference guide [here](docs/inference_image.md).
* [December 12, 2025] Released updated checkpoints for [Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B) (blur, depth, segmentation, edge), fixed an issue with autoregressive multiview when num_conditional_frames == 0, optimized control video rendering, refreshed documentation, and added a new post-training example for [single-view](docs/post-training_singleview.md) edge/depth/seg/blur modalities.
* [November 25, 2025] Added Blackwell + ARM inference support, Auto/Multiview code fixes, along with fixes for the help menu and CLI overrides, improved guardrail offloading, and LFS enablement for large assets.
* [November 11, 2025] Refactored the Cosmos-Transfer2.5-2B Auto/Multiview code, and updated the Auto/Multiview checkpoints in Hugging Face.
* [November 7, 2025] We added autoregressive sliding window generation mode for generating longer videos. We also added a new multiview cross-attention module, upgraded dependencies to improve support for Blackwell, and updated inference examples and documentation.
* [November 6, 2025] As part of the Cosmos family, we released the recipe, a reference diffusion model and a tokenizer for [synthetic LiDAR point cloud generation](https://github.com/nv-tlabs/Cosmos-Drive-Dreams/tree/main/cosmos-transfer-lidargen) from RGB image!
* [October 28, 2025] We added [Cosmos Cookbook](https://github.com/nvidia-cosmos/cosmos-cookbook), a collection of step-by-step recipes and post-training scripts to quickly build, customize, and deploy NVIDIA’s Cosmos world foundation models for robotics and autonomous systems.
* [October 28, 2025] We added the autogeneration of spatiotemporal masking for control inputs when prompt is given, added cosmos-oss, new pyrefly annotations, introduced multi-storage backend in easyio, reorganized internal packages, and boosted Transfer2 speed with Torch Compile tokenizer optimizations.
* [October 21, 2025] We added on-the-fly computation support for depth and segmentation, and fixed multicontrol experiments in [inference](docs/inference.md). Also, updated Docker base image version, and Gradio related documentation.
* [October 13, 2025] Updated Transfer2.5 Auto Multiview [post-training datasets](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md), and setup dependencies to support NVIDIA Blackwell.
* [October 6, 2025] We released [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) - the next generation of our world simulation models!
* [June 12, 2025] As part of the Cosmos family, we released [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)

## Cosmos-Transfer2.5

Cosmos-Transfer2.5 is a multi-controlnet designed to accept structured input of multiple video modalities including RGB, depth, segmentation and more. Users can configure generation using JSON-based controlnet_specs, and run inference with just a few commands. It supports both single-video inference, automatic control map generation, and multiple GPU setups.

Physical AI trains upon data generated in two important data augmentation workflows.

### Simulation 2 Real Augmentation

Minimizing the need for achieving high fidelity in 3D simulation.

**Input prompt:**
> A contemporary luxury kitchen with marble tabletops. window with beautiful sunset outside. There is an esspresso coffee maker on the table in front of the white robot arm. Robot arm interacts with a coffee cup and coffee maker on the kitchen table.

<table>
  <tr>
    <th>Input Video</th>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/20d63162-0fd5-483a-a306-7b8021df5ed9" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/131ffe81-cca0-44cd-8547-7b0e49d5253f" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/e4dd3b80-4696-4930-8b05-6d41e37974c2" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/5a816f4d-fdc3-4939-b2b9-141c6ee64d2b" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/56f76740-ea36-4916-9e94-c983d6b84d28" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Real 2 Real Augmentation

Leveraging sensor captured RGB augmentation.

**Input prompt:**
> Dashcam video, driving through a modern urban environment, winter with heavy snow storm, trees and sidewalks covered in snow.
<table>
  <tr>
    <th>Input Video</th>
    <th>Computed Control</th>
    <th>Output Video</th>
  </tr>
  <tr>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/4705c192-b8c6-4ba3-af7f-fd968c4a3eeb" width="100%" alt="Input video" controls></video>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/ba92fa5d-2972-463e-af2e-a637a810a463" width="100%" alt="Control map video" controls></video>
      <details>
        <summary>See more computed controls</summary>
        <video src="https://github.com/user-attachments/assets/f8e6c351-78b5-4bd6-949b-e1845aa19f63" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/7edf3f46-c4da-403f-b630-d8853a165602" width="100%" alt="Control map video" controls></video>
        <video src="https://github.com/user-attachments/assets/ba59f926-c4c2-4232-bdbf-392c53f29a97" width="100%" alt="Control map video" controls></video>
      </details>
    </td>
    <td valign="top" width="33%">
      <video src="https://github.com/user-attachments/assets/8e62af23-3ca4-4e72-97fe-7a337a31d306" width="100%" alt="Output video" controls></video>
    </td>
  </tr>
</table>

### Scaling World State Diversity Examples

Robotic Matrix Diversity Example
<video src="https://github.com/user-attachments/assets/5daee273-5f49-4238-a67f-d63fdb48a4d9" width="100%" alt="Input video" controls></video>

AV Matrix Diversity Example
<video src="https://github.com/user-attachments/assets/51b18d9b-0cb4-44dc-898c-624e3020dcb1" width="100%" alt="Input video" controls></video>

For an example demonstrating how to augment sythentic data with Cosmos Transfer on robotics navigation tasks to improve Sim2Real performance see [Cosmos Transfer Sim2Real for Robotics Navigation Tasks](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer1/inference-x-mobility/inference.html) in the [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/).

## Cosmos-Transfer2.5 Model Family

Cosmos-Transfer supports data generation in multiple industry verticals, outlined below. Please check back as we continue to add more specialized models to the Transfer family!

[**Cosmos-Transfer2.5-2B**](docs/inference.md): General [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), Distilled [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/distilled/general), trained from the ground up for Physical AI and robotics.

[**Cosmos-Transfer2.5-2B/auto**](docs/inference_auto_multiview.md): Specialized checkpoints, post-trained for Autonomous Vehicle applications. [Multiview checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/auto). For an example demonstrating how to augment sythentic data with Cosmos Transfer on Autonomous Vehicle see [Cosmos Transfer 2.5 Sim2Real for Simulator Videos](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-carla-sdg-augmentation/inference.html) in the [Cosmos Cookbook](https://nvidia-cosmos.github.io/cosmos-cookbook/).

[**Cosmos-Transfer2.5-2B/robot-multiview-control**](docs/inference_robot_multiview_control.md): Specialized control-conditioned checkpoints for robot multiview applications. Supports 4 control types (depth, edge, visual blur, segmentation) for precise video generation guided by structural information.

## User Guide

* [Setup Guide](docs/setup.md)
* [Troubleshooting](docs/troubleshooting.md)
* [Inference](docs/inference.md)
  * [Auto Multiview](docs/inference_auto_multiview.md)
  * [Image Inference](docs/inference_image.md)
  * [Robot Multiview Control](docs/inference_robot_multiview_control.md)
* [Post-training](docs/post-training.md)
  * [Single View](docs/post-training_singleview.md)
  * [Auto Multiview](docs/post-training_auto_multiview.md)

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
