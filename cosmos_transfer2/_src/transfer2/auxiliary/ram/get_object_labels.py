# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

import cv2
import torch
import torchvision.transforms as TS
from PIL import Image
from ram import inference_ram

from cosmos_transfer2._src.transfer2.auxiliary.ram.recognize_anything.ram.models import ram


def get_ram_transform(image_size=384):
    return TS.Compose(
        [
            TS.Resize((image_size, image_size)),
            TS.ToTensor(),
            TS.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def initialize_ram_model(ram_checkpoint_path, image_size, device):
    print("Initializing RAM++ Model...")
    ram_model = ram(pretrained=ram_checkpoint_path, image_size=image_size, vit="swin_l").eval().to(device)
    ram_model.eval()
    return ram_model


def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def retrieve_tags(ram_model, ram_transform, video_path, device="cuda"):
    frame = get_first_frame(video_path)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    ram_input = ram_transform(img).unsqueeze(0).to(device)
    res = inference_ram(ram_input, ram_model)
    s = res[0] if isinstance(res, (list, tuple)) else res
    s = s.replace(" |", ".")
    if s[-1] != ".":
        s += "."
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve RAM++ tags from the first frame of a video.")

    # Positional arguments
    parser.add_argument("video_path", help="Path to the input video.")

    # Optional arguments
    parser.add_argument("--ram_checkpoint", required=True, help="Path to the RAM++ pretrained checkpoint (.pth).")
    parser.add_argument("--image_size", type=int, default=384, help="Image size used for RAM++ (default: 384).")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run RAM++ on (default: cuda if available, else cpu).",
    )

    args = parser.parse_args()

    # Initialize model + transform
    ram_transform = get_ram_transform(args.image_size)
    ram_model = initialize_ram_model(
        ram_checkpoint_path=args.ram_checkpoint, image_size=args.image_size, device=args.device
    )

    # Retrieve tags
    tags = retrieve_tags(
        ram_model=ram_model, ram_transform=ram_transform, video_path=args.video_path, device=args.device
    )

    print("\n=== RAM++ Tags ===")
    print(tags)
