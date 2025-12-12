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
from cosmos_oss.fixtures.script import ScriptConfig, ScriptRunner, extract_bash_commands
from cosmos_oss.fixtures.script import script_runner as script_runner

_CURRENT_DIR = Path(__file__).parent.absolute()
_SCRIPT_DIR = _CURRENT_DIR / "docs_test"
_DOCS_DIR = _CURRENT_DIR.parent / "docs"

INFERENCE_DOCS = sorted([f.name for f in _DOCS_DIR.glob("inference*.md")])
POSTTRAINING_DOCS = sorted([f.name for f in _DOCS_DIR.glob("post-training_*.md")])

DOCS_CONFIG = [ScriptConfig(script=doc) for doc in INFERENCE_DOCS]

SCRIPT_CONFIGS = [
    ScriptConfig(
        script="depth.sh",
    ),
    ScriptConfig(
        script="depth_tokenizer_compile.sh",
        levels=[2],
    ),
    ScriptConfig(
        script="depth_parallel_tokenizer.sh",
        levels=[2],
    ),
    ScriptConfig(
        script="edge.sh",
    ),
    ScriptConfig(
        script="seg.sh",
    ),
    ScriptConfig(
        script="vis.sh",
    ),
    ScriptConfig(
        script="image.sh",
    ),
    ScriptConfig(
        script="vanilla_multicontrol.sh",
        gpus=8,
    ),
    ScriptConfig(
        script="auto_multiview.sh",
        gpus=8,
    ),
    ScriptConfig(
        script="auto_multiview_autoregressive.sh",
        gpus=8,
        levels=[2],
    ),
    ScriptConfig(
        script="post-training_auto_multiview.sh",
        gpus=8,
    ),
    ScriptConfig(
        script="post-training_singleview.sh",
        gpus=8,
    ),
]


@pytest.mark.level(0)
@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.marks) for cfg in SCRIPT_CONFIGS if 0 in cfg.levels]
)
def test_level_0(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_0)


@pytest.mark.level(0)
@pytest.mark.gpus(1)
@pytest.mark.parametrize(
    "cfg", [pytest.param(cfg, id=cfg.name, marks=cfg.marks) for cfg in DOCS_CONFIG if 0 in cfg.levels]
)
def test_docs(cfg: ScriptConfig, script_runner: ScriptRunner, tmp_path: Path):
    """Test individual doc commands."""
    md_path = _DOCS_DIR / cfg.script
    script_path = generate_script_from_doc(md_path, tmp_path)
    script_runner.run(str(script_path), script_runner.env_level_0)


@pytest.mark.level(1)
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(cfg, id=cfg.name, marks=[pytest.mark.gpus(cfg.gpus), *cfg.marks])
        for cfg in SCRIPT_CONFIGS
        if 1 in cfg.levels
    ],
)
def test_level_1(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_1)


@pytest.mark.level(2)
@pytest.mark.parametrize(
    "cfg",
    [
        pytest.param(cfg, id=cfg.name, marks=[pytest.mark.gpus(8), *cfg.marks])
        for cfg in SCRIPT_CONFIGS
        if 2 in cfg.levels
    ],
)
def test_level_2(cfg: ScriptConfig, script_runner: ScriptRunner):
    script_runner.run(f"{_SCRIPT_DIR}/{cfg.script}", script_runner.env_level_2)


def generate_script_from_doc(md_file: Path, tmp_path: Path) -> Path:
    """Generate a bash script from a markdown file."""
    scripts_dir = tmp_path / "generated_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = scripts_dir / f"{md_file.stem}.sh"
    commands = extract_bash_commands(md_file)

    with open(script_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("set -euxo pipefail\n\n")
        f.write(f"# Commands from {md_file.name}\n")
        for cmd in commands:
            f.write(cmd)
            f.write("\n\n")

    return script_path
