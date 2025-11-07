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

import os
import subprocess
from pathlib import Path

import pytest

_CURRENT_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _CURRENT_DIR.parent


def _get_env(tmp_path: Path):
    return (
        {
            "INPUT_DIR": _ROOT_DIR,
            "COSMOS_VERBOSE": "0",
        }
        | dict(os.environ)
        | {
            "COSMOS_INTERNAL": "0",
            "COSMOS_SMOKE": "0",
            "OUTPUT_DIR": f"{_ROOT_DIR}/output",
            "TMP_DIR": f"{tmp_path}/tmp",
            "IMAGINAIRE_OUTPUT_ROOT": f"{_ROOT_DIR}/imaginaire4-output",
        }
    )


@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "test_script",
    [
        pytest.param("vanilla.sh", id="vanilla"),
        pytest.param("vanilla_multicontrol.sh", id="vanilla_multicontrol"),
        pytest.param("auto_multiview.sh", id="auto_multiview"),
        pytest.param("auto_multiview_autoregressive.sh", id="auto_multiview_autoregressive"),
        pytest.param("post-training_auto_multiview.sh", id="post_training_auto_multiview"),
    ],
)
def test_smoke(test_script: str, tmp_path: Path):
    cmd = [
        f"{_CURRENT_DIR}/docs_test/{test_script}",
    ]
    env = _get_env(tmp_path) | {"COSMOS_SMOKE": "1"}
    subprocess.check_call(cmd, cwd=_ROOT_DIR, env=env)


@pytest.mark.parametrize(
    "test_script",
    [
        pytest.param("vanilla.sh", id="vanilla", marks=[pytest.mark.gpus(1), pytest.mark.level(1)]),
        pytest.param(
            "vanilla_multicontrol.sh", id="vanilla_multicontrol", marks=[pytest.mark.gpus(8), pytest.mark.level(1)]
        ),
        pytest.param("auto_multiview.sh", id="auto_multiview", marks=[pytest.mark.gpus(1), pytest.mark.level(1)]),
        pytest.param(
            "auto_multiview_autoregressive.sh",
            id="auto_multiview_autoregressive",
            marks=[pytest.mark.gpus(1), pytest.mark.level(1)],
        ),
        pytest.param(
            "post-training_auto_multiview.sh",
            id="post_training_auto_multiview",
            marks=[pytest.mark.gpus(8), pytest.mark.level(2)],
        ),
    ],
)
def test_full(test_script: str, tmp_path: Path):
    cmd = [
        f"{_CURRENT_DIR}/docs_test/{test_script}",
    ]
    subprocess.check_call(
        cmd,
        cwd=_ROOT_DIR,
        env=_get_env(tmp_path),
    )
