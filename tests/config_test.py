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

from pathlib import Path

import pytest
from pydantic import ValidationError

from cosmos_transfer2.config import DEFAULT_MODEL_KEY, SetupArguments
from cosmos_transfer2.multiview_config import MultiviewSetupArguments
from cosmos_transfer2.robot_multiview_control_agibot_config import RobotMultiviewControlAgibotSetupArguments


def test_cfg_parallel_is_base_setup_only():
    assert "cfg_parallel" in SetupArguments.model_fields
    assert "cfg_parallel" not in MultiviewSetupArguments.model_fields
    assert "cfg_parallel" not in RobotMultiviewControlAgibotSetupArguments.model_fields


def test_cfg_parallel_requires_full_even_context_parallel_world(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("WORLD_SIZE", "4")

    args = SetupArguments(
        output_dir=tmp_path,
        model=DEFAULT_MODEL_KEY.name,
        cfg_parallel=True,
        context_parallel_size=4,
    )

    assert args.cfg_parallel

    with pytest.raises(ValidationError, match="even context_parallel_size"):
        SetupArguments(
            output_dir=tmp_path,
            model=DEFAULT_MODEL_KEY.name,
            cfg_parallel=True,
            context_parallel_size=3,
        )

    with pytest.raises(ValidationError, match="match WORLD_SIZE"):
        SetupArguments(
            output_dir=tmp_path,
            model=DEFAULT_MODEL_KEY.name,
            cfg_parallel=True,
            context_parallel_size=2,
        )