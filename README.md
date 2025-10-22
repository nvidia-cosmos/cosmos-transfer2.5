<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos/"> Product Website</a>&nbsp | 🤗 <a href="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B">Hugging Face</a>&nbsp | <a href="https://research.nvidia.com/publication/2025-09_world-simulation-video-foundation-models-physical-ai">Paper</a> | <a href="https://research.nvidia.com/labs/dir/cosmos-transfer2.5/">Paper Website</a>
</p>

NVIDIA Cosmos™ is a platform purpose-built for physical AI, featuring state-of-the-art generative world foundation models (WFMs), robust guardrails, and an accelerated data processing and curation pipeline. Designed specifically for real-world systems, Cosmos enables developers to rapidly advance physical AI applications such as autonomous vehicles (AVs), robots, and video analytics AI agents.

Cosmos World Foundation Models come in three model types which can all be customized in post-training: [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5), and [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1).

## News
* [October 21, 2025] We added on-the-fly computation support for depth and segmentation, and fixed multicontrol experiments in [inference](docs/inference.md). Also, updated Docker base image version, and Gradio related documentation.

* [October 13, 2025] Updated Transfer2.5 Auto Multiview [post-training datasets](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md), and setup dependencies to support NVIDIA Blackwell.
  
* [October 6, 2025] We released [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) and [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) - the next generation of our world simulation models!

* [June 12, 2025] As part of the Cosmos family, we released [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer)

## Cosmos-Transfer2.5

Cosmos-Transfer2.5 is a multi-controlnet designed to accept structured input of multiple video modalities including RGB, depth, segmentation and more. Users can configure generation using JSON-based controlnet_specs, and run inference with just a few commands. It supports both single-video inference, automatic control map generation, and multiple GPU setups.

Physical AI trains upon data generated in two important data augmentation workflows.

### Simulations to Photorealism

Minimizing the need for achieving high fidelity in 3D simulation.

**Input prompt:**
> The video is a demonstration of robotic manipulation, likely in a laboratory or testing environment. It features two robotic arms interacting with a piece of blue fabric. <details> <summary>Click to see more prompt</summary>
> The setting is a room with a beige couch in the background, providing a neutral backdrop for the robotic activity. The robotic arms are positioned on either side of the fabric, which is placed on a yellow cushion. The left robotic arm is white with a black gripper, while the right arm is black with a more complex, articulated gripper. At the beginning, the fabric is laid out on the cushion. The left robotic arm approaches the fabric, its gripper opening and closing as it positions itself. The right arm remains stationary initially, poised to assist. As the video progresses, the left arm grips the fabric, lifting it slightly off the cushion. The right arm then moves in, its gripper adjusting to grasp the opposite side of the fabric. Both arms work in coordination, lifting and holding the fabric between them. The fabric is manipulated with precision, showcasing the dexterity and control of the robotic arms. The camera remains static throughout, focusing on the interaction between the robotic arms and the fabric, allowing viewers to observe the detailed movements and coordination involved in the task.</details>

| Input Video | Computed Control | Output Video |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/bffc031e-3933-4511-a659-136965931ab0" width="100%" alt="Input video" controls></video> | <video src="https://github.com/user-attachments/assets/8ed4c49c-af26-4318-b95a-32f9cf44d992" width="100%" alt="Control map video" controls></video> | <video src="https://github.com/user-attachments/assets/88f7e63b-efe1-46ff-8174-df2f01462c53" width="100%" alt="Output video" controls></video> |

### Scale World State Diversity

Leveraging sensor captured RGB or ground truth augmentations.

**Input prompt:**
> The video is a driving scene through a modern urban environment, likely captured from a dashcam or a similar fixed camera setup inside a vehicle. <details><summary>Click to see more prompt</summary>
> The scene unfolds on a wide, multi-lane road flanked by tall, modern buildings with glass facades. The road is relatively empty, with only a few cars visible, including a black car directly ahead of the camera, maintaining a steady pace. The camera remains static, providing a consistent view of the road and surroundings as the vehicle moves forward.On the left side of the road, there are several trees lining the sidewalk, providing a touch of greenery amidst the urban setting. Pedestrians are visible on the sidewalks, some walking leisurely, while others stand near the buildings. The buildings are a mix of architectural styles, with some featuring large glass windows and others having more traditional concrete exteriors. A few commercial signs and logos are visible on the buildings, indicating the presence of businesses and offices.Traffic cones are placed on the road ahead, suggesting some form of roadwork or lane closure, guiding the vehicles to merge or change lanes. The road markings are clear, with white arrows indicating the direction of travel. The sky is clear, suggesting a sunny day, which enhances the visibility of the scene. Throughout the video, the vehicle maintains a steady speed, and the camera captures the gradual approach towards the intersection, where the road splits into different directions. The overall atmosphere is calm and orderly, typical of a city during non-peak hours.</details>

| Input Video | Computed Control | Output Video |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/4705c192-b8c6-4ba3-af7f-fd968c4a3eeb" width="100%" alt="Input video" controls></video> | <video src="https://github.com/user-attachments/assets/ba92fa5d-2972-463e-af2e-a637a810a463" width="100%" alt="Control map video" controls></video> | <video src="https://github.com/user-attachments/assets/0c5151d4-968b-42ad-a517-cdc0dde37ee5" width="100%" alt="Output video" controls></video> |

## Cosmos-Transfer2.5 Model Family

Cosmos-Transfer supports data generation in multiple industry verticals, outlined below. Please check back as we continue to add more specialized models to the Transfer family!

[**Cosmos-Transfer2.5-2B**](docs/inference.md): General [checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B), trained from the ground up for Physical AI and robotics.

[**Cosmos-Transfer2.5-2B/auto**](docs/inference_auto_multiview.md): Specialized checkpoints, post-trained for Autonomous Vehicle applications. [Multiview checkpoints](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/auto)

## User Guide

* [Setup Guide](docs/setup.md)
* [Inference](docs/inference.md)
  * [Auto Multiview](docs/inference_auto_multiview.md)
* [Post-training](docs/post-training.md)
  * [Auto Multiview](docs/post-training_auto_multiview.md)

## Contributing

We thrive on community collaboration! [NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) wouldn't be where it is without contributions from developers like you. Check out our [Contributing Guide](CONTRIBUTING.md) to get started, and share your feedback through issues.

Big thanks 🙏 to everyone helping us push the boundaries of open-source physical AI!

## License and Contact

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.

NVIDIA Cosmos source code is released under the [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0).

NVIDIA Cosmos models are released under the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). For a custom license, please contact [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com).
