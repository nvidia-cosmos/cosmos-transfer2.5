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

"""
Config wrapper for single-view Transfer2 post-training.

This is a simple passthrough to the main transfer2 config which already
registers all experiments including the singleview examples.
"""

# Import the main transfer2 config which now includes singleview experiment registration
from cosmos_transfer2._src.transfer2.configs.vid2vid_transfer.config import make_config

__all__ = ["make_config"]
