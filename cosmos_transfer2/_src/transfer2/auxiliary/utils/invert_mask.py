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
import shlex
import subprocess


def invert_video(input_binary_mask, output_video_path):
    ffmpeg_cmd = f'''
    ffmpeg -y -i "{input_binary_mask}" \
        -vf "format=gray,lut='if(gt(val\,0)\,255\,0)'" \
        -c:v libx264 -pix_fmt yuv420p "{output_video_path}"
    '''
    process = subprocess.run(shlex.split(ffmpeg_cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Invert a binary mask video.")
    parser.add_argument("input_video", help="Path to input binary mask video")
    parser.add_argument("output_video", help="Path to output inverted binary mask video")

    args = parser.parse_args()

    invert_video(args.input_video, args.output_video)


"""
Usage (MP4 output):
  
python cosmos_transfer2/_src/transfer2/auxiliary/utils/invert_mask.py \
input_video.mp4 \
inverted_mask.mp4
"""

