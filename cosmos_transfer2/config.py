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

import enum
import json
import os
import sys
from dataclasses import dataclass
from functools import cache, cached_property
from importlib import import_module
from pathlib import Path
from typing import Annotated, Any, Literal, NoReturn, Optional, TypeVar

import pydantic
import tyro
from typing_extensions import Self

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import get_checkpoint_by_uuid


@cache
def is_rank0() -> bool:
    return os.environ.get("RANK", "0") == "0"


def path_to_str(v: Path | None) -> str | None:
    """Convert optional path to optional string."""
    if v is None:
        return None
    return str(v)


def load_callable(name: str):
    idx = name.rfind(".")
    assert idx > 0, "expected <module_name>.<identifier>"
    module_name = name[0:idx]
    fn_name = name[idx + 1 :]

    module = import_module(module_name)
    fn = getattr(module, fn_name)
    return fn


_PydanticModelT = TypeVar("_PydanticModelT", bound=pydantic.BaseModel)


def get_overrides_cls(cls: type[_PydanticModelT], *, exclude: list[str] | None = None) -> type[pydantic.BaseModel]:
    """Get overrides class for a given pydantic model."""
    # pyrefly: ignore  # no-matching-overload
    names = set(cls.model_fields.keys())
    if exclude is not None:
        invalid = set(exclude) - names
        if invalid:
            raise ValueError(f"Invalid exclude: {invalid}")
        names -= set(exclude)
    fields = {name: (Optional[cls.model_fields[name].rebuild_annotation()], None) for name in names}  # type: ignore
    # pyrefly: ignore  # no-matching-overload, bad-argument-type, bad-argument-count
    return pydantic.create_model(f"{cls.__name__}Overrides", **fields)


def _get_root_exception(exception: Exception) -> Exception:
    if exception.__cause__ is not None:
        # pyrefly: ignore  # bad-argument-type
        return _get_root_exception(exception.__cause__)
    if exception.__context__ is not None:
        # pyrefly: ignore  # bad-argument-type
        return _get_root_exception(exception.__context__)
    return exception


def handle_tyro_exception(exception: Exception) -> NoReturn:
    root_exception = _get_root_exception(exception)
    if isinstance(root_exception, pydantic.ValidationError):
        if is_rank0():
            print(root_exception, file=sys.stderr)
        sys.exit(1)
    raise exception


def _resolve_path(v: Path) -> Path:
    """Resolve path to absolute."""
    return v.expanduser().resolve()


ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]
ResolvedDirectoryPath = Annotated[pydantic.DirectoryPath, pydantic.AfterValidator(_resolve_path)]


def _validate_checkpoint_uuid(v: str) -> str:
    """Validate checkpoint UUID."""
    get_checkpoint_by_uuid(v)
    return v


CheckpointUuid = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_uuid)]


def _validate_checkpoint_path(v: str) -> str:
    """Validate checkpoint path or URI."""
    if v.startswith("s3://"):
        return v
    if not os.path.exists(v):
        raise ValueError(f"Checkpoint path '{v}' does not exist.")
    return v


CheckpointPath = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_path)]


class ModelVariant(str, enum.Enum):
    DEPTH = "depth"
    EDGE = "edge"
    SEG = "seg"
    VIS = "vis"
    AUTO_MULTIVIEW = "auto/multiview"
    ROBOT_MULTIVIEW = "robot/multiview"
    ROBOT_MULTIVIEW_AGIBOT = "robot/multiview-agibot"


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    variant: ModelVariant = ModelVariant.EDGE

    @cached_property
    def name(self) -> str:
        return self.variant.value

    def __str__(self) -> str:
        return self.name


MODEL_CHECKPOINTS = {
    ModelKey(variant=ModelVariant.DEPTH): get_checkpoint_by_uuid("0f214f66-ae98-43cf-ab25-d65d09a7e68f"),
    ModelKey(variant=ModelVariant.EDGE): get_checkpoint_by_uuid("ecd0ba00-d598-4f94-aa09-e8627899c431"),
    ModelKey(variant=ModelVariant.SEG): get_checkpoint_by_uuid("fcab44fe-6fe7-492e-b9c6-67ef8c1a52ab"),
    ModelKey(variant=ModelVariant.VIS): get_checkpoint_by_uuid("20d9fd0b-af4c-4cca-ad0b-f9b45f0805f1"),
    ModelKey(variant=ModelVariant.AUTO_MULTIVIEW): get_checkpoint_by_uuid("b5ab002d-a120-4fbf-a7f9-04af8615710b"),
}

MODEL_KEYS = {k.name: k for k in MODEL_CHECKPOINTS.keys()}

BASE_MODEL_VARIANTS = [ModelVariant.EDGE, ModelVariant.DEPTH, ModelVariant.SEG, ModelVariant.VIS]


# pyrefly: ignore  # invalid-annotation
def get_model_literal(variants: list[ModelVariant] | None = None) -> Literal:
    """Get model literal for a given variant."""
    model_names: list[str] = []
    for k in MODEL_CHECKPOINTS.keys():
        if variants is not None and k.variant not in variants:
            continue
        model_names.append(k.name)
    # pyrefly: ignore  # bad-return, invalid-literal
    return Literal[tuple(model_names)]


DEFAULT_MODEL_KEY = ModelKey()
DEFAULT_NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."


class CommonSetupArguments(pydantic.BaseModel):
    """Common arguments for model setup."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)

    # Required parameters
    output_dir: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    """Output directory."""

    # Optional parameters
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal() = DEFAULT_MODEL_KEY.name
    """Model name."""
    checkpoint_path: CheckpointPath | None = None
    """Path to the checkpoint."""
    experiment: str | None = None
    """Experiment name."""
    config_file: str = "cosmos_transfer2/_src/predict2/configs/video2world/config.py"
    """Configuration file for the model."""
    context_parallel_size: pydantic.PositiveInt | None = None
    """Context parallel size. Default to all nodes."""
    disable_guardrails: bool = False
    """Disable guardrails if this is set to True."""
    offload_guardrail_models: bool = True
    """Offload guardrail models to CPU to save GPU memory."""
    keep_going: bool = False
    """Keep going if an error occurs."""
    profile: bool = False
    """Run profiler and save report to output directory."""

    def enable_guardrails(self) -> bool:
        return not self.disable_guardrails

    @cached_property
    def model_key(self) -> ModelKey:
        return MODEL_KEYS[self.model]

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_model(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        model_name: str | None = data.get("model")
        if model_name is None:
            raise ValueError("model is required")
        model_key = MODEL_KEYS[model_name]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        if data.get("checkpoint_path") is None:
            data["checkpoint_path"] = checkpoint.path
        if data.get("experiment") is None:
            data["experiment"] = checkpoint.experiment
        if data.get("context_parallel_size") is None:
            data["context_parallel_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        return data


class SetupArguments(CommonSetupArguments):
    """Base model setup arguments."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal(BASE_MODEL_VARIANTS) = DEFAULT_MODEL_KEY.name


Guidance = Annotated[int, pydantic.Field(ge=0, le=7)]


class CommonInferenceArguments(pydantic.BaseModel):
    """Common inference arguments."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # Required parameters
    name: str
    """Name of the sample."""
    prompt_path: ResolvedFilePath | None = pydantic.Field(None, init_var=True)
    """Path to file containing the prompt."""
    prompt: str | None = None
    """Text prompt for generation."""

    # Optional parameters
    negative_prompt: str | None = None
    """Negative prompt."""

    # Advanced parameters
    seed: int = 0
    """Seed value."""
    guidance: Guidance = 3
    """Guidance value."""

    @pydantic.model_validator(mode="before")
    @classmethod
    def validate_prompt(cls, data: Any) -> Any:
        """
        Sets the 'prompt' field using the content of 'prompt_path' if it's provided.
        """
        if not isinstance(data, dict):
            return data
        prompt: str | None = data.get("prompt")
        if prompt is not None:
            return data
        prompt_path: str | None = data.get("prompt_path")
        if prompt_path is not None:
            # pyrefly: ignore  # annotation-mismatch
            prompt_path: Path = ResolvedFilePath(prompt_path)
            data["prompt"] = prompt_path.read_text().strip()
            return data
        return data

    @classmethod
    def _from_file(cls, path: Path, override_data: dict[str, Any]) -> list[Self]:
        """Load arguments from a json/jsonl/yaml file.

        Returns a list of arguments.
        """
        # Load data from file
        if path.suffix in [".json"]:
            data_list = [json.loads(path.read_text())]
        elif path.suffix in [".jsonl"]:
            data_list = [json.loads(line) for line in path.read_text().splitlines()]
        else:
            raise ValueError(f"Unsupported file extension: {path.suffix}")

        # Validate data
        # Input paths are relative to the file path
        cwd = os.getcwd()
        os.chdir(path.parent)
        objs: list[Self] = []
        for i, data in enumerate(data_list):
            try:
                objs.append(cls.model_validate(data | override_data))
            except pydantic.ValidationError as e:
                if is_rank0():
                    print(f"Error validating parameters from '{path}' at line {i}\n{e}", file=sys.stderr)
                sys.exit(1)
        os.chdir(cwd)

        return objs

    @classmethod
    def from_files(cls, paths: list[Path], overrides: pydantic.BaseModel | None = None) -> tuple[list[Self], list[str]]:
        """Load arguments from a list of json/jsonl/yaml files.

        Returns a list of arguments.
        """
        if not paths:
            if is_rank0():
                print("Error: No inference parameter files", file=sys.stderr)
            sys.exit(1)

        if overrides is None:
            override_data = {}
        else:
            override_data = overrides.model_dump(exclude_none=True)

        # Load arguments from files
        objs: list[Self] = []
        for path in paths:
            objs.extend(cls._from_file(path, override_data))
        if not objs:
            if is_rank0():
                print("Error: No inference samples", file=sys.stderr)
            sys.exit(1)

        # Check if names are unique
        names: set[str] = set()
        batch_hint_keys: set[str] = set()
        for obj in objs:
            if obj.name in names:
                print(f"Error: Inference samplename {obj.name} is not unique", file=sys.stderr)
                sys.exit(1)
            names.add(obj.name)
            for key in CONTROL_KEYS:
                if getattr(obj, key, None) is not None:
                    batch_hint_keys.add(key)
        sorted_batch_hint_keys = sorted(batch_hint_keys, key=lambda x: CONTROL_KEYS.index(x))
        return objs, sorted_batch_hint_keys


ControlWeight = Annotated[float, pydantic.Field(ge=0.0, le=1.0, step=0.01)]

Threshold = Literal["very_low", "low", "medium", "high", "very_high"]


class ControlConfig(pydantic.BaseModel):
    # Control path is required so no default value
    control_path: ResolvedFilePath | None
    mask_path: ResolvedFilePath | None = None
    control_weight: ControlWeight = 1.0


class BlurConfig(ControlConfig):
    preset_blur_strength: Threshold = "medium"
    # Override the control path to be not required
    control_path: ResolvedFilePath | None = None


class EdgeConfig(ControlConfig):
    preset_edge_threshold: Threshold = "medium"
    # Override the control path to not be required
    control_path: ResolvedFilePath | None = None


class SegConfig(ControlConfig):
    control_prompt: str | None = None
    # Override the control path to be not required
    control_path: ResolvedFilePath | None = None


class DepthConfig(ControlConfig):
    # Override the control path to not be required (can be computed automatically)
    # pyrefly: ignore  # bad-override
    control_path: ResolvedFilePath | None = None


CONTROL_KEYS = ["edge", "vis", "depth", "seg"]


class InferenceArguments(CommonInferenceArguments):
    # pyrefly: ignore  # bad-assignment
    video_path: ResolvedFilePath = None
    image_context_path: ResolvedFilePath | None = None

    resolution: str = "720"
    sigma_max: str | None = None
    num_conditional_frames: Literal[0, 1, 2] = 1
    num_video_frames_per_chunk: pydantic.PositiveInt = 93
    num_steps: pydantic.PositiveInt = 35

    show_control_condition: bool = False
    show_input: bool = False
    not_keep_input_resolution: bool = False

    edge: EdgeConfig | None = None
    depth: DepthConfig | None = None
    vis: BlurConfig | None = None
    seg: SegConfig | None = None

    # Override defaults
    guidance: Guidance = 3
    seed: int = 2025
    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    # pyrefly: ignore  # bad-override
    prompt: str

    @cached_property
    def hint_keys(self) -> list[str]:
        return [key for key in CONTROL_KEYS if getattr(self, key, None) is not None]

    def model_post_init(self, __context) -> None:
        if len(self.hint_keys) == 0:
            raise ValueError("No controls provided, please provide at least one control key (edge, blur, depth, seg)")

    @cached_property
    def control_weight_dict(self) -> str:
        # control weight is a comma seperated string in the same order as hint_keys
        control_weight_dict = {}
        for key in self.hint_keys:
            control_weight_dict[key] = str(getattr(self, key).control_weight)
        # pyrefly: ignore  # bad-return
        return control_weight_dict

    @cached_property
    def control_modalities(self) -> dict[str, str | None]:
        control_modalities = {}
        for key in self.hint_keys:
            control_modalities[key] = path_to_str(getattr(self, key).control_path)
            control_modalities[f"{key}_mask"] = path_to_str(getattr(self, key).mask_path)
        return control_modalities

    @cached_property
    def preset_edge_threshold(self) -> Threshold:
        if "edge" in self.hint_keys:
            return getattr(self, "edge").preset_edge_threshold
        return "medium"

    @cached_property
    def preset_blur_strength(self) -> Threshold:
        if "vis" in self.hint_keys:
            return getattr(self, "vis").preset_blur_strength
        return "medium"

    @cached_property
    def seg_control_prompt(self) -> str | None:
        if "seg" not in self.hint_keys or getattr(self, "seg").control_path is not None:
            return None
        if getattr(self, "seg").control_prompt is not None:
            return getattr(self, "seg").control_prompt
        default_prompt = " ".join(self.prompt.split()[:128])
        log.warning(
            f'No "control_prompt" provided for on-the-fly segmentation, using the first 128 words of the input prompt'
        )
        return default_prompt


InferenceOverrides = get_overrides_cls(
    InferenceArguments,
    exclude=[
        "name",
        "edge",
        "depth",
        "vis",
        "seg",
    ],
)
