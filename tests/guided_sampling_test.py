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

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.gpus(1)
def test_guided_sampling():
    # Follow the repository convention: GPU work runs outside the main pytest process.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:env",
            str(Path(__file__).with_name("guided_sampling_checks.py")),
            "-q",
        ],
        check=True,
    )
