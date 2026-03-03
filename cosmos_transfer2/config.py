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
from cosmos_oss.checkpoints_transfer2 import register_checkpoints
from pydantic_core import PydanticUndefined
from typing_extensions import Self

from cosmos_transfer2._src.imaginaire.flags import EXPERIMENTAL_CHECKPOINTS, SMOKE
from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.imaginaire.utils.checkpoint_db import CheckpointConfig, get_checkpoint_uri

register_checkpoints()


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
    names = [name for name in cls.model_fields.keys() if exclude is None or name not in exclude]
    fields = {}
    for name in names:
        # pyrefly: ignore  # no-matching-overload
        model_field = cls.model_fields[name]
        behavior_hint = (
            f"(default: {model_field.default})"
            if model_field.default is not PydanticUndefined
            else "(default: None) (required)"
        )
        annotation = Annotated[
            Optional[cls.model_fields[name].rebuild_annotation()],  # pyrefly: ignore  # no-matching-overload
            tyro.conf.arg(help_behavior_hint=behavior_hint),
        ]
        fields[name] = (annotation, pydantic.Field(default=None, description=model_field.description))
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
    return v.expanduser().absolute()


ResolvedFilePath = Annotated[pydantic.FilePath, pydantic.AfterValidator(_resolve_path)]
ResolvedDirectoryPath = Annotated[pydantic.DirectoryPath, pydantic.AfterValidator(_resolve_path)]


def _validate_checkpoint_path(v: str) -> str:
    """Validate and normalize checkpoint path or URI."""
    return get_checkpoint_uri(v, check_exists=True)


CheckpointPath = Annotated[str, pydantic.AfterValidator(_validate_checkpoint_path)]


class ModelVariant(str, enum.Enum):
    DEPTH = "depth"
    EDGE = "edge"
    SEG = "seg"
    VIS = "vis"
    AUTO_MULTIVIEW = "auto/multiview"
    # Transfer2.5 Agibot Control-Conditioned Multiview
    ROBOT_MULTIVIEW_AGIBOT_DEPTH = "robot/multiview-agibot-depth"
    ROBOT_MULTIVIEW_AGIBOT_EDGE = "robot/multiview-agibot-edge"
    ROBOT_MULTIVIEW_AGIBOT_VIS = "robot/multiview-agibot-vis"
    ROBOT_MULTIVIEW_AGIBOT_SEG = "robot/multiview-agibot-seg"
    ROBOT_MULTIVIEW_MANY_CAMERA = "robot/multiview-many-camera"


class CompileMode(str, enum.Enum):
    NONE = "none"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass(frozen=True, kw_only=True)
class ModelKey:
    variant: ModelVariant = ModelVariant.EDGE
    distilled: bool = False

    @cached_property
    def name(self) -> str:
        if self.distilled:
            return f"{self.variant.value}/distilled"
        return self.variant.value

    def __str__(self) -> str:
        return self.name


MODEL_CHECKPOINTS = {
    ModelKey(variant=ModelVariant.DEPTH): CheckpointConfig.from_uri("626e6618-bfcd-4d9a-a077-1409e2ce353f"),
    ModelKey(variant=ModelVariant.EDGE): CheckpointConfig.from_uri("61f5694b-0ad5-4ecd-8ad7-c8545627d125"),
    ModelKey(variant=ModelVariant.SEG): CheckpointConfig.from_uri("5136ef49-6d8d-42e8-8abf-7dac722a304a"),
    ModelKey(variant=ModelVariant.VIS): CheckpointConfig.from_uri("ba2f44f2-c726-4fe7-949f-597069d9b91c"),
    ModelKey(variant=ModelVariant.AUTO_MULTIVIEW): CheckpointConfig.from_uri("4ecc66e9-df19-4aed-9802-0d11e057287a"),
}
if EXPERIMENTAL_CHECKPOINTS:
    MODEL_CHECKPOINTS |= {
        ModelKey(variant=ModelVariant.EDGE, distilled=True): CheckpointConfig.from_uri(
            "41f07f13-f2e4-4e34-ba4c-86f595acbc20"
        ),
        # Transfer2.5 Agibot Control-Conditioned Multiview
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_DEPTH): CheckpointConfig.from_uri(
            "32514ba1-6d05-4ce5-997d-a3b5bf894cab"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_EDGE): CheckpointConfig.from_uri(
            "fffbd388-89c9-4604-ad4f-6c6b36272c48"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_VIS): CheckpointConfig.from_uri(
            "2eca9f80-bf8f-4257-b05f-278065d21500"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_AGIBOT_SEG): CheckpointConfig.from_uri(
            "c5a9a58b-7f3e-4b45-9e5d-8f7b3d4e5a6c"
        ),
        ModelKey(variant=ModelVariant.ROBOT_MULTIVIEW_MANY_CAMERA): CheckpointConfig.from_uri(
            "a8794d70-842c-44a5-95bb-9010d5ace7be"
        ),
    }

MODEL_KEYS = {k.name: k for k in MODEL_CHECKPOINTS.keys()}

BASE_MODEL_VARIANTS = [ModelVariant.EDGE, ModelVariant.DEPTH, ModelVariant.SEG, ModelVariant.VIS]

# Base experiment that all singleview control types inherit from
# This is the shared architecture/training config for all BASE_MODEL_VARIANTS
DEFAULT_BASE_EXPERIMENT = "vid2vid_2B_control_720p_t24_control_layer4_cr1pt1_embedding_rectified_flow"


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
    """Model name. You shouldn't override this for most cases."""
    checkpoint_path: CheckpointPath | None = None
    """Path to the checkpoint. Override this if you have a post-training checkpoint"""
    experiment: str | None = None
    """Experiment name. Override this with your custom experiment when post-training"""
    config_file: str = ""
    """Configuration file for the model. Leave empty to use the default config for the selected model type."""
    context_parallel_size: pydantic.PositiveInt | None = None
    """Context parallel size. Defaults to WORLD_SIZE set by torchrun."""
    disable_guardrails: bool = True if SMOKE else False
    """Option to enable or disable guardrails."""
    offload_guardrail_models: bool = False
    """Offload guardrail models to CPU to save GPU memory."""
    keep_going: bool = True
    """When running batch inference, keep going if an error occurs. If set to False, the batch will stop on the first error."""
    profile: bool = False
    """Run profiler and save report to output directory."""
    benchmark: bool = False
    """Enable benchmarking mode. Runs the first sample as warmup and reports the average run times of all other samples. If single sample specified repeats it 4 times before running."""
    compile_tokenizer: CompileMode = CompileMode.NONE
    """Set tokenizer compilation mode: 'none' (default), 'moderate', or 'aggressive'. 'moderate' and 'aggresive' cause a significant overhead on the first use (use if you want to generate 30+ videos in one run). Aggressive compilation can cause OOM on some systems."""
    enable_parallel_tokenizer: bool = False
    """Enable Context Parallelism for Wan Tokenizer for multi-GPU encoding/decoding."""
    parallel_tokenizer_grid: tuple[int, int] = (-1, -1)
    """
    Specify the grid to use for Parallel Wan Tokenizer. First number represents the splitting factor in height dimension.
    The second number represents the splitting factor in width dimension.
    The latent dimensions of the image or video need to be divisible by these values.
    """

    @cached_property
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
            data["checkpoint_path"] = checkpoint.s3.uri
        if data.get("experiment") is None:
            data["experiment"] = checkpoint.experiment
        # Set config file based on model type (distilled vs non-distilled)
        if not data.get("config_file"):
            if model_key.distilled:
                data["config_file"] = "cosmos_transfer2/_src/interactive/configs/registry_transfer2p5.py"
            else:
                data["config_file"] = "cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py"
        if data.get("context_parallel_size") is None:
            data["context_parallel_size"] = int(os.environ.get("WORLD_SIZE", "1"))
        return data

    @cached_property
    def has_checkpoint_override(self) -> bool:
        model_key = MODEL_KEYS[self.model]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        return self.checkpoint_path != checkpoint.s3.uri

    @cached_property
    def has_experiment_override(self) -> bool:
        model_key = MODEL_KEYS[self.model]
        checkpoint = MODEL_CHECKPOINTS[model_key]
        return self.experiment != checkpoint.experiment


class SetupArguments(CommonSetupArguments):
    """Base model setup arguments."""

    # Override defaults
    # pyrefly: ignore  # invalid-annotation
    model: get_model_literal(BASE_MODEL_VARIANTS) = DEFAULT_MODEL_KEY.name


Guidance = Annotated[int, pydantic.Field(ge=0, le=7)]


class CommonInferenceArguments(pydantic.BaseModel):
    """Common inference arguments."""

    model_config = pydantic.ConfigDict(extra="forbid", use_attribute_docstrings=True)

    # Required parameters
    name: str
    """Name of the sample."""
    prompt_path: ResolvedFilePath | None = pydantic.Field(None, init_var=True)
    """Path to a .txt file containing the prompt. Only one of {prompt} or {prompt_path} should be provided."""
    prompt: str | None = None
    """Text prompt for generation. Only one of {prompt} or {prompt_path} should be provided."""

    # Optional parameters
    negative_prompt: str | None = None
    """Negative prompt - describing what you don't want in the generated video."""

    # Advanced parameters
    seed: int = 0
    "Seed for generation randomness."
    guidance: Guidance = 3
    """Range from 0 to 7: the higher the value, the closer the generated video adheres to the prompt."""

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
    model_config = pydantic.ConfigDict(use_attribute_docstrings=True)

    control_path: ResolvedFilePath | None = None
    """Path to pre-computed control video. If None, control is generated on-the-fly from input video."""
    control_weight: ControlWeight = 1.0
    """Range from 0.0 to 1.0 per control, how strongly the output adheres to the control. For multicontrol, the weights are automaticallynormalized to sum to <=1.0."""
    mask_path: ResolvedFilePath | None = None
    """Path to a pre-computed binary spatiotemporal mask. White pixels are where the control is applied, black pixels are ignored. Only one of {mask_path} or {mask_prompt} should be provided."""
    mask_prompt: str | None = None
    """Prompt for generating a mask on the fly (eg "car building tree"). Passed to the SAM2 model to segment the objects in the prompt and create masks."""


class DepthConfig(ControlConfig):
    """Arguments for depth control. These can only be provided via the json input file."""

    control_path: ResolvedFilePath | None = None
    """Path to pre-computed depth map. If None, depth is generated on-the-fly from input video using VideoDepthAnything."""


class BlurConfig(ControlConfig):
    """Arguments for vis control. These can only be provided via the json input file."""

    control_path: ResolvedFilePath | None = None
    """Path to pre-computed blur map. If None, blur is generated on-the-fly from input video using Bilateral Gaussian Blur."""
    preset_blur_strength: Threshold = "medium"
    """Options: 'very_low', 'low', 'medium', 'high', 'very_high'. Controls the strength of blur when generating blur maps on-the-fly."""


class EdgeConfig(ControlConfig):
    """Arguments for edge control. These can only be provided via the json input file."""

    control_path: ResolvedFilePath | None = None
    """Path to pre-computed edge map. If None, edge is generated on-the-fly from input video using CannyEdge Model."""

    preset_edge_threshold: Threshold = "medium"
    """Options: 'very_low', 'low', 'medium', 'high', 'very_high'. Lower thresholds detect more edges (including noise), higher thresholds detect fewer edges."""


class SegConfig(ControlConfig):
    """Arguments for seg control. These can only be provided via the json input file."""

    control_path: ResolvedFilePath | None = None
    """Path to pre-computed segmentation map. If None, segmentation is generated on-the-fly from input video using GroundDino(base) + SAM2."""
    control_prompt: str | None = None
    """Prompt for on-the-fly segmentation. Describes what should be segmented in the input video (eg "car building tree").
    Default: first 128 words of the input prompt."""


CONTROL_KEYS = ["edge", "vis", "depth", "seg"]


class InferenceArguments(CommonInferenceArguments):
    video_path: ResolvedFilePath
    """Required. Path to input video. Control videos and masks computed on-the-fly are based on this video.
    """
    max_frames: pydantic.PositiveInt | None = None
    """Number of frames to read from {video_path}. Must be less than or equal to the number of frames in {video_path}. 
    Defaults is None, which means the entire {video_path}."""
    context_frame_index: pydantic.PositiveInt | None = None
    """Index of a frame in the input video to use as image context. If provided, this image is used as a style reference for the output."""
    image_context_path: ResolvedFilePath | None = None
    """Path to an image file. If provided, this image is used as a style reference for the output. Ignored if {context_frame_index} is provided.
    If None and {context_frame_idx} is not provided, use a random frame from the input video."""

    num_conditional_frames: Literal[0, 1, 2] = 1
    """Used for chunk-wise long video generation. Number of frames from the previously-generated chunk to condition the next chunk on. 
    Always 0 for the first chunk, and defaults to 1 for the following chunks."""
    resolution: str = "720"
    """Output video resolution (e.g., '720', '480')"""
    sigma_max: str | None = None
    """Range from 0 to 200 for how much noise is added to the input video/image. 200 means pure noise and the output will be completely random."""
    num_video_frames_per_chunk: pydantic.PositiveInt = 93
    """Number of video frames per chunk in the chunk-wise long video generation."""
    num_steps: pydantic.PositiveInt = 1 if SMOKE else 35
    """Number of sampling steps in the diffusion process. Higher values produce better quality but require more time."""

    show_control_condition: bool = False
    """Concatenate control videos and masks to the output video. Controls are stored separately in the output directory, regardless of this setting."""
    show_input: bool = False
    """Concatenate the input video to the output video."""
    keep_input_resolution: bool = True
    """Whether to resize the output video to the input resolution. If True, output will be resized to input resolution. Otherwise, output will use the model's native resolution."""

    edge: EdgeConfig | None = None
    depth: DepthConfig | None = None
    vis: BlurConfig | None = None
    seg: SegConfig | None = None

    seed: int = 2025
    "Seed for generation randomness."
    # pyrefly: ignore  # bad-override
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT
    """Negative prompt - describing what you don't want in the generated video."""
    # pyrefly: ignore  # bad-override
    prompt: str
    """Text prompt describing generation. Only one of {prompt} or {prompt_path} should be provided. """

    @cached_property
    def hint_keys(self) -> list[str]:
        return [key for key in CONTROL_KEYS if getattr(self, key, None) is not None]

    def model_post_init(self, __context) -> None:
        if len(self.hint_keys) == 0:
            raise ValueError("No controls provided, please provide at least one control key (edge, blur, depth, seg)")

        if "vis" in self.hint_keys and self.image_context_path:
            raise ValueError(
                "vis control and image_context_path are both used to transfer style. Using these modes together leads to conflicts. Please only provide one"
            )

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
            control_modalities[f"{key}_mask_prompt"] = getattr(self, key).mask_prompt
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

    @cached_property
    def not_keep_input_resolution(self) -> bool:
        return not self.keep_input_resolution


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
