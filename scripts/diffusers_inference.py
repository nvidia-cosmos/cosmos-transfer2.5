#!/usr/bin/env -S uv run --script
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# https://docs.astral.sh/uv/guides/scripts/#using-a-shebang-to-create-an-executable-file
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "tyro",
#   "pydantic",
#   "numpy",
#   "opencv-python",
#   "Pillow",
#   "easydict",
#   "matplotlib",
#   "imageio",
#   "torch==2.10.0",
#   "transformers==4.57.1",
#   "sam-2 @ git+https://github.com/facebookresearch/sam2.git",
#   "video-depth-anything @ git+https://github.com/jeanachoi/Video-Depth-Anything.git",
#   "cosmos-guardrail @ git+https://github.com/codeJRV/cosmos-guardrail",
#   "diffusers @ git+https://github.com/huggingface/diffusers.git",
# ]
# [tool.uv.sources]
# torch = [
#   { index = "pytorch-cu128", marker = "platform_machine != 'aarch64'" },
#   { index = "pytorch-cu130", marker = "platform_machine == 'aarch64'" },
# ]
# [[tool.uv.index]]
# name = "pytorch-cu128"
# url = "https://download.pytorch.org/whl/cu128"
# explicit = true
# [[tool.uv.index]]
# name = "pytorch-cu130"
# url = "https://download.pytorch.org/whl/cu130"
# explicit = true
# ///

import colorsys
import json
import os
import random
import re
import sys
import tempfile
from contextlib import nullcontext
from dataclasses import field, fields
from pathlib import Path
from typing import Annotated, Literal

import cv2
import numpy as np
import pydantic
import torch
import tyro
from diffusers import Cosmos2_5_TransferPipeline, CosmosControlNetModel
from diffusers.utils import export_to_video, load_image, load_video
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic.dataclasses import dataclass
from video_depth_anything import video_depth

DEFAULT_NEGATIVE_PROMPT = "The video captures a game playing, with bad crappy graphics and cartoonish frames. It represents a recording of old outdated games. The lighting looks very fake. The textures are very raw and basic. The geometries are very primitive. The images are very pixelated and of poor CG quality. There are many subtitles in the footage. Overall, the video is unrealistic at all."
CONTROL_KEYS = ("edge", "vis", "depth", "seg")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(config=pydantic.ConfigDict(extra="ignore"))
class ControlConfig:
    control_path: Path | None = None
    control_weight: Annotated[float | None, pydantic.Field(ge=0.0, le=1.0)] = None
    preset_edge_threshold: Literal["none", "very_low", "low", "medium", "high", "very_high"] | None = None
    preset_blur_strength: Literal["none", "very_low", "low", "medium", "high", "very_high"] | None = None
    control_prompt: str | None = None


@dataclass(config=pydantic.ConfigDict(extra="ignore"))
class SampleConfig(ControlConfig):
    name: str | None = None
    prompt: str | None = None
    prompt_path: Path | None = None
    negative_prompt: str | None = None
    seed: int | None = None
    guidance: float | None = None
    video_path: str | None = None
    max_frames: int | None = None
    num_conditional_frames: int | None = None
    resolution: str | None = None
    num_video_frames_per_chunk: int | None = None
    num_steps: int | None = None
    keep_input_resolution: bool | None = None
    edge: ControlConfig | None = None
    vis: ControlConfig | None = None
    depth: ControlConfig | None = None
    seg: ControlConfig | None = None


DEFAULT_SAMPLE = SampleConfig(
    name=None,
    prompt=None,
    prompt_path=None,
    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
    seed=2025,
    guidance=3.0,
    video_path=None,
    max_frames=None,
    num_conditional_frames=1,
    resolution="720",
    num_video_frames_per_chunk=93,
    num_steps=35,
    keep_input_resolution=True,
    edge=None,
    vis=None,
    depth=None,
    seg=None,
)

SAMPLE_CONFIG_ADAPTER = pydantic.TypeAdapter(SampleConfig)


class Args(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    input_files: Annotated[list[Path] | None, tyro.conf.arg(aliases=("-i",))] = None
    control_key: str | None = None
    sample_index: int = 0

    sample_overrides: SampleConfig = field(default_factory=SampleConfig)

    output_path: Annotated[Path, tyro.conf.arg(aliases=("-o",))]
    output_fps: int = 30

    model_id: str = "nvidia/Cosmos-Transfer2.5-2B"
    revision: str = "diffusers/general"
    controlnet_revision_template: str = "diffusers/controlnet/general/{control_key}"

    device: str = "cuda"
    device_map: str | None = None
    torch_dtype: Literal["bfloat16", "float16", "float32"] = "bfloat16"
    mock_safety_checker: bool = False

    # control generation
    force_generate_controls: bool = False
    save_controls: bool = False
    seg_mode: Literal["point", "auto"] = "auto"

    large_depth_model: bool = False
    sam2_model_id: str = "facebook/sam2-hiera-large"


def _load_frames(path: Path, max_frames: int | None) -> list[Image.Image]:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return [load_image(str(path)).convert("RGB")]
    if suffix in VIDEO_EXTENSIONS:
        frames = load_video(str(path))
        if max_frames is not None:
            frames = frames[:max_frames]
        if len(frames) == 0:
            raise ValueError(f"No frames loaded from {path}")
        return [frame.convert("RGB") for frame in frames]
    raise ValueError(f"Unsupported file extension '{path.suffix}' for {path}")


class MockSafetyChecker:
    def to(self, device):
        del device
        return self

    def check_text_safety(self, text: str) -> bool:
        del text
        return True

    def check_video_safety(self, video):
        return video


def get_edge_controls(source_np: np.ndarray, control_cfg: ControlConfig, args: Args) -> list[Image.Image]:
    del args
    thresholds = {
        "none": (20, 50),
        "very_low": (20, 50),
        "low": (50, 100),
        "medium": (100, 200),
        "high": (200, 300),
        "very_high": (300, 400),
    }
    t_low, t_high = thresholds[control_cfg.preset_edge_threshold or "medium"]
    controls = []
    for frame in source_np:
        edge = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), t_low, t_high)
        controls.append(Image.fromarray(np.repeat(edge[:, :, None], 3, axis=2)))
    return controls


def get_vis_controls(source_np: np.ndarray, control_cfg: ControlConfig, args: Args) -> list[Image.Image]:
    del args
    preset = control_cfg.preset_blur_strength or "medium"
    if preset == "none":
        return [Image.fromarray(frame) for frame in source_np]

    downup = {"very_low": 4, "low": 4, "medium": 10, "high": 16, "very_high": 16}[preset]
    blur_down = {"very_low": 1, "low": 4, "medium": 2, "high": 1, "very_high": 4}[preset]
    h, w = source_np.shape[1:3]
    stage = source_np.copy()
    if blur_down > 1:
        stage = np.stack(
            [
                cv2.resize(frame, (max(1, w // blur_down), max(1, h // blur_down)), interpolation=cv2.INTER_AREA)
                for frame in stage
            ],
            axis=0,
        ).astype(np.uint8)
    stage = np.stack(
        [cv2.bilateralFilter(frame, d=30, sigmaColor=150, sigmaSpace=100) for frame in stage], axis=0
    ).astype(np.uint8)
    if blur_down > 1:
        stage = np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR) for frame in stage], axis=0).astype(
            np.uint8
        )
    if downup > 1:
        down = np.stack(
            [
                cv2.resize(frame, (max(1, w // downup), max(1, h // downup)), interpolation=cv2.INTER_CUBIC)
                for frame in stage
            ],
            axis=0,
        ).astype(np.uint8)
        stage = np.stack([cv2.resize(frame, (w, h), interpolation=cv2.INTER_CUBIC) for frame in down], axis=0).astype(
            np.uint8
        )
    return [Image.fromarray(frame) for frame in stage]


def get_depth_controls(source_np: np.ndarray, control_cfg: ControlConfig, args: Args) -> list[Image.Image]:
    del control_cfg
    runtime_device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    runtime_device_str = str(runtime_device)

    if args.large_depth_model:
        depth_model_id = "depth-anything/Video-Depth-Anything-Large"
        depth_weights = "video_depth_anything_vitl.pth"
        depth_model_config = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}
    else:
        depth_model_id = "depth-anything/Video-Depth-Anything-Small"
        depth_weights = "video_depth_anything_vits.pth"
        depth_model_config = {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]}

    weights_path = hf_hub_download(repo_id=depth_model_id, filename=depth_weights)

    model = video_depth.VideoDepthAnything(**depth_model_config)
    model.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=True)
    model.to(runtime_device).eval()

    with torch.no_grad():
        depth_stack, _ = model.infer_video_depth(source_np, 30, device=runtime_device_str)

    depth_stack = np.asarray(depth_stack, dtype=np.float32)
    if depth_stack.shape[1:3] != source_np.shape[1:3]:
        source_h, source_w = source_np.shape[1:3]
        depth_stack = np.stack(
            [cv2.resize(frame, (source_w, source_h), interpolation=cv2.INTER_LINEAR) for frame in depth_stack], axis=0
        )

    depth_norm = (depth_stack - float(depth_stack.min())) / (float(depth_stack.max() - depth_stack.min()) + 1e-8)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    return [Image.fromarray(np.repeat(frame[:, :, None], 3, axis=2)) for frame in depth_u8]


def _generate_distinct_color(rng: random.Random) -> np.ndarray:
    hue = rng.uniform(0.0, 1.0)
    saturation = rng.uniform(0.1, 1.0)
    value = rng.uniform(0.2, 1.0)
    red, green, blue = colorsys.hsv_to_rgb(hue, saturation, value)
    return (np.array([red, green, blue], dtype=np.float32) * 255.0).astype(np.uint8)


def _resize_bool_mask(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_2d = np.squeeze(mask)
    if mask_2d.ndim != 2:
        return np.zeros((height, width), dtype=bool)
    mask_2d = mask_2d.astype(bool)
    if mask_2d.shape == (height, width):
        return mask_2d
    resized = Image.fromarray(mask_2d.astype(np.uint8) * 255).resize((width, height), resample=Image.NEAREST)
    return np.array(resized, dtype=np.uint8) > 127


def _build_dense_tracked_masks(
    video_segments: dict[int, dict[int, np.ndarray]], num_frames: int, height: int, width: int
) -> np.ndarray:
    obj_ids = sorted({obj_id for frame_masks in video_segments.values() for obj_id in frame_masks})
    if len(obj_ids) == 0:
        return np.zeros((0, num_frames, height, width), dtype=bool)

    obj_to_index = {obj_id: idx for idx, obj_id in enumerate(obj_ids)}
    tracks = np.zeros((len(obj_ids), num_frames, height, width), dtype=bool)

    for frame_idx, frame_masks in video_segments.items():
        if frame_idx < 0 or frame_idx >= num_frames:
            continue
        for obj_id, mask in frame_masks.items():
            tracks[obj_to_index[obj_id], frame_idx] = _resize_bool_mask(mask, height=height, width=width)

    # Fill missing frame slots per object using nearest available masks.
    for track in tracks:
        has_mask = track.reshape(num_frames, -1).any(axis=1)
        if not np.any(has_mask):
            continue
        known_indices = np.flatnonzero(has_mask)
        first_idx = int(known_indices[0])
        last_idx = int(known_indices[-1])

        for frame_idx in range(first_idx + 1, last_idx + 1):
            if not has_mask[frame_idx]:
                track[frame_idx] = track[frame_idx - 1]
        for frame_idx in range(0, first_idx):
            track[frame_idx] = track[first_idx]
        for frame_idx in range(last_idx + 1, num_frames):
            track[frame_idx] = track[last_idx]

    non_empty_tracks = [track for track in tracks if track.any()]
    if len(non_empty_tracks) == 0:
        return np.zeros((0, num_frames, height, width), dtype=bool)
    return np.stack(non_empty_tracks, axis=0)


def _auto_seed_points(width: int, height: int) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for y in np.linspace(0.1, 0.9, num=4):
        for x in np.linspace(0.1, 0.9, num=6):
            points.append((float(x * width), float(y * height)))
    return points


def _parse_seg_point(control_prompt: str | None, width: int, height: int) -> tuple[float, float]:
    center = (width / 2.0, height / 2.0)
    if control_prompt is None:
        return center

    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(control_prompt))]
    if len(nums) < 2:
        print(
            "WARN: seg.control_prompt text is not supported without GroundingDINO; using center point for SAM2",
            file=sys.stderr,
        )
        return center

    x, y = nums[0], nums[1]
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x, y = x * width, y * height
    x = float(np.clip(x, 0.0, max(0.0, width - 1.0)))
    y = float(np.clip(y, 0.0, max(0.0, height - 1.0)))
    return x, y


def _run_sam2_video_tracking(
    source_np: np.ndarray,
    sam2_predictor,
    seed_points: list[tuple[float, float]],
    runtime_device: torch.device,
) -> np.ndarray:
    num_frames, height, width = source_np.shape[:3]
    if len(seed_points) == 0:
        return np.zeros((0, num_frames, height, width), dtype=bool)

    with tempfile.TemporaryDirectory(prefix="sam2_video_") as frames_dir:
        frame_root = Path(frames_dir)
        for frame_idx, frame in enumerate(source_np):
            Image.fromarray(frame).save(frame_root / f"{frame_idx:05d}.jpg", format="JPEG", quality=95)

        video_segments: dict[int, dict[int, np.ndarray]] = {}
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16) if runtime_device.type == "cuda" else nullcontext()
        )

        with torch.inference_mode(), autocast_ctx:
            state = sam2_predictor.init_state(video_path=str(frame_root))
            for obj_id, (x, y) in enumerate(seed_points, start=1):
                x = float(np.clip(x, 0.0, max(0.0, width - 1.0)))
                y = float(np.clip(y, 0.0, max(0.0, height - 1.0)))
                points = np.array([[x, y]], dtype=np.float32)
                labels = np.array([1], dtype=np.int32)
                sam2_predictor.add_new_points_or_box(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                )

            for out_frame_idx, out_obj_ids, out_mask_logits in sam2_predictor.propagate_in_video(state):
                frame_masks: dict[int, np.ndarray] = {}
                for mask_idx, out_obj_id in enumerate(out_obj_ids):
                    mask = (out_mask_logits[mask_idx] > 0.0).detach().cpu().numpy()
                    resized_mask = _resize_bool_mask(mask, height=height, width=width)
                    if resized_mask.any():
                        frame_masks[int(out_obj_id)] = resized_mask
                if len(frame_masks) > 0:
                    video_segments[int(out_frame_idx)] = frame_masks

    return _build_dense_tracked_masks(video_segments, num_frames=num_frames, height=height, width=width)


def _filter_and_dedupe_auto_tracks(tracks: np.ndarray) -> np.ndarray:
    if tracks.shape[0] == 0:
        return tracks

    _, num_frames, height, width = tracks.shape
    frame_area = float(height * width)
    filtered: list[np.ndarray] = []
    for track in tracks:
        frame_pixels = track.reshape(num_frames, -1).sum(axis=1)
        max_coverage = float(frame_pixels.max()) / frame_area
        mean_coverage = float(frame_pixels.mean()) / frame_area
        if max_coverage < 0.005:
            continue
        if mean_coverage < 0.001:
            continue
        if max_coverage > 0.95:
            continue
        filtered.append(track)

    if len(filtered) == 0:
        return np.zeros((0, num_frames, height, width), dtype=bool)

    filtered = sorted(filtered, key=lambda mask: int(mask.sum()), reverse=True)
    unique_tracks: list[np.ndarray] = []
    for track in filtered:
        duplicate = False
        for existing in unique_tracks:
            inter = np.logical_and(track, existing).sum()
            union = np.logical_or(track, existing).sum()
            iou = float(inter) / float(union) if union > 0 else 0.0
            if iou > 0.85:
                duplicate = True
                break
        if not duplicate:
            unique_tracks.append(track)

    if len(unique_tracks) == 0:
        return np.zeros((0, num_frames, height, width), dtype=bool)
    return np.stack(unique_tracks, axis=0)


def _render_point_controls(track: np.ndarray) -> list[Image.Image]:
    controls: list[Image.Image] = []
    for frame_mask in track:
        mask_u8 = frame_mask.astype(np.uint8) * 255
        controls.append(Image.fromarray(np.repeat(mask_u8[:, :, None], 3, axis=2)))
    return controls


def _render_auto_controls(tracks: np.ndarray, seed: int | None) -> list[Image.Image]:
    _, num_frames, height, width = tracks.shape
    if tracks.shape[0] == 0:
        return [Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)) for _ in range(num_frames)]

    num_masks = tracks.shape[0]
    sorted_indices = sorted(range(num_masks), key=lambda idx: int(tracks[idx].sum()), reverse=True)
    rng = random.Random(seed)
    colors = [_generate_distinct_color(rng) for _ in range(num_masks)]

    controls: list[Image.Image] = []
    for frame_idx in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for color_idx, mask_idx in enumerate(sorted_indices):
            mask = tracks[mask_idx, frame_idx]
            if mask.any():
                frame[mask] = colors[color_idx % len(colors)]
        controls.append(Image.fromarray(frame))
    return controls


def get_seg_controls(
    source_np: np.ndarray, control_cfg: ControlConfig, args: Args, sample: SampleConfig
) -> list[Image.Image]:
    runtime_device = torch.device(
        args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    )
    control_prompt = control_cfg.control_prompt
    num_frames, height, width = source_np.shape[:3]

    try:
        from sam2.sam2_video_predictor import SAM2VideoPredictor
    except ImportError as exc:
        raise ImportError(
            "seg control generation requires SAM2VideoPredictor (`sam2` package). "
            "Install with `pip install git+https://github.com/facebookresearch/sam2.git`."
        ) from exc

    sam2_predictor = SAM2VideoPredictor.from_pretrained(args.sam2_model_id).to(runtime_device)
    sam2_predictor.eval()

    if args.seg_mode == "point":
        point = _parse_seg_point(control_prompt=control_prompt, width=width, height=height)
        tracks = _run_sam2_video_tracking(
            source_np=source_np,
            sam2_predictor=sam2_predictor,
            seed_points=[point],
            runtime_device=runtime_device,
        )
        if tracks.shape[0] == 0:
            tracks = np.zeros((1, num_frames, height, width), dtype=bool)
        return _render_point_controls(tracks[0])

    if control_prompt is not None:
        print("WARN: seg_mode=auto ignores control_prompt coordinates/text", file=sys.stderr)

    tracks = _run_sam2_video_tracking(
        source_np=source_np,
        sam2_predictor=sam2_predictor,
        seed_points=_auto_seed_points(width=width, height=height),
        runtime_device=runtime_device,
    )
    tracks = _filter_and_dedupe_auto_tracks(tracks)

    if tracks.shape[0] == 0:
        center_track = _run_sam2_video_tracking(
            source_np=source_np,
            sam2_predictor=sam2_predictor,
            seed_points=[(width / 2.0, height / 2.0)],
            runtime_device=runtime_device,
        )
        tracks = center_track[:1]

    if tracks.shape[0] == 0:
        tracks = np.zeros((1, num_frames, height, width), dtype=bool)
    return _render_auto_controls(tracks, seed=sample.seed)


def load_sample(input_files: list[Path], sample_index: int) -> SampleConfig | None:
    rows: list[dict] = []
    for input_file in input_files or []:
        input_file = input_file.expanduser().absolute()
        if input_file.suffix == ".json":
            loaded = [json.loads(input_file.read_text())]
        elif input_file.suffix == ".jsonl":
            loaded = [json.loads(line) for line in input_file.read_text().splitlines() if line.strip()]
        else:
            raise ValueError(f"Unsupported input file extension: {input_file}")

        root = input_file.parent
        for row in loaded:
            for key in ("prompt_path", "video_path"):
                value = row.get(key)
                if value is not None:
                    value = Path(value).expanduser()
                    row[key] = str((root / value).absolute() if not value.is_absolute() else value.absolute())

            for control_key in CONTROL_KEYS:
                control_cfg = row.get(control_key)
                if not isinstance(control_cfg, dict):
                    continue
                for key in ("control_path",):
                    value = control_cfg.get(key)
                    if value is not None:
                        value = Path(value).expanduser()
                        control_cfg[key] = str(
                            (root / value).absolute() if not value.is_absolute() else value.absolute()
                        )

            rows.append(row)

    sample_json: dict[str, object] = {}
    if len(rows) != 0:
        if sample_index < 0 or sample_index >= len(rows):
            raise ValueError(f"sample_index={sample_index} out of range for {len(rows)} input samples")
        if len(rows) != 1:
            print(f"WARN: loaded {len(rows)} samples, running only sample_index={sample_index}", file=sys.stderr)
        sample_json = dict(rows[sample_index])

    sample = SAMPLE_CONFIG_ADAPTER.validate_python(sample_json)
    return sample


def main(args: Args):
    # load the sample and override with CLI args provided
    sample = load_sample(args.input_files, args.sample_index) or DEFAULT_SAMPLE
    for f in fields(SampleConfig):
        if getattr(args.sample_overrides, f.name) is not None:
            setattr(sample, f.name, getattr(args.sample_overrides, f.name))
        elif getattr(sample, f.name) is None and getattr(DEFAULT_SAMPLE, f.name) is not None:
            setattr(sample, f.name, getattr(DEFAULT_SAMPLE, f.name))

    if sample.prompt is None and sample.prompt_path is not None:
        sample.prompt = Path(sample.prompt_path).read_text().strip()

    if sample.prompt is None:
        raise ValueError("No prompt provided. Set prompt or prompt_path in JSON or via CLI.")

    provided_controls = [key for key in CONTROL_KEYS if getattr(sample, key) is not None]
    if len(provided_controls) > 1:
        print("ERROR: diffusers only supports one control input", file=sys.stderr)
        sys.exit(1)

    control_key = args.control_key or provided_controls[0]
    control_cfg = getattr(sample, control_key)
    source_path = sample.video_path
    if source_path is None:
        source_path = control_cfg.control_path
        if source_path is None:
            raise ValueError("video_path is required unless control_path is provided")
        print("WARN: No video_path provided; using control_path as source", file=sys.stderr)

    source_path = Path(source_path).expanduser().absolute()
    input_frames = _load_frames(source_path, sample.max_frames)
    original_h, original_w = np.array(input_frames[0]).shape[:2]

    control_path = control_cfg.control_path
    generate_controls = args.force_generate_controls or control_path is None
    if generate_controls:
        if args.force_generate_controls and control_path is not None:
            print(f"WARN: force_generate_controls=True, ignoring control_path={control_path}", file=sys.stderr)
        else:
            print(f"WARN: control_path not provided, computing {control_key} on-the-fly", file=sys.stderr)
        source_np = np.stack([np.array(frame, dtype=np.uint8) for frame in input_frames], axis=0)

        if control_key == "edge":
            controls = get_edge_controls(source_np, control_cfg, args)
        elif control_key == "vis":
            controls = get_vis_controls(source_np, control_cfg, args)
        elif control_key == "depth":
            controls = get_depth_controls(source_np, control_cfg, args)
        elif control_key == "seg":
            controls = get_seg_controls(source_np, control_cfg, args, sample)
        else:
            raise ValueError(f"Unsupported control key: {control_key}")
    else:
        control_path = Path(control_path).expanduser().absolute()
        assert control_path.exists(), f"{control_path} does not exist"
        controls = _load_frames(control_path, sample.max_frames)

    target_frames = len(input_frames)
    if len(controls) > target_frames:
        controls = controls[:target_frames]
    elif len(controls) < target_frames:
        controls = controls + [controls[-1]] * (target_frames - len(controls))

    output_arg_path = args.output_path.expanduser().absolute()
    control_base = output_arg_path / sample.name if output_arg_path.suffix == "" else output_arg_path.with_suffix("")
    if args.save_controls:
        os.makedirs(control_base.parent, exist_ok=True)
        control_output_base = control_base.parent / f"{control_base.name}_control"
        if len(controls) == 1:
            control_output_path = control_output_base.with_suffix(".jpg")
            controls[0].save(str(control_output_path))
        else:
            control_output_path = control_output_base.with_suffix(".mp4")
            export_to_video(controls, str(control_output_path), fps=args.output_fps)
        print(f"Saved controls: {control_output_path}", file=sys.stderr)

    digits = "".join(ch for ch in sample.resolution if ch.isdigit())
    height = int(digits) - 16 if digits else 704
    height = max(16, height - (height % 16))

    torch_dtype = getattr(torch, args.torch_dtype)

    controlnet_key = "blur" if control_key == "vis" else control_key
    controlnet = CosmosControlNetModel.from_pretrained(
        args.model_id,
        revision=args.controlnet_revision_template.format(control_key=controlnet_key),
        torch_dtype=torch_dtype,
    )
    pipe = Cosmos2_5_TransferPipeline.from_pretrained(
        args.model_id,
        revision=args.revision,
        controlnet=controlnet,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
    )
    if args.mock_safety_checker:
        pipe.safety_checker = MockSafetyChecker()
        print("WARN: using mock safety checker (guardrails bypassed)", file=sys.stderr)
    if args.device_map is None:
        pipe = pipe.to(args.device)

    generator = None
    if sample.seed is not None:
        generator_device = args.device if args.device_map is None else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(int(sample.seed))

    frames = pipe(
        controls=controls,
        controls_conditioning_scale=float(control_cfg.control_weight)
        if control_cfg.control_weight is not None
        else 1.0,
        prompt=sample.prompt,
        negative_prompt=sample.negative_prompt,
        num_frames=len(controls),
        height=height,
        num_inference_steps=sample.num_steps,
        guidance_scale=sample.guidance,
        num_frames_per_chunk=sample.num_video_frames_per_chunk,
        num_ar_conditional_frames=sample.num_conditional_frames,
        generator=generator,
    ).frames[0]

    if sample.keep_input_resolution:
        out_h = original_h - (original_h % 2)
        out_w = original_w - (original_w % 2)
        resample = getattr(Image, "Resampling", Image).LANCZOS
        frames = [frame.resize((out_w, out_h), resample) for frame in frames]

    output_path = output_arg_path
    if output_path.suffix == "":
        output_path = output_path / f"{sample.name}{'.jpg' if len(frames) == 1 else '.mp4'}"
    elif len(frames) == 1 and output_path.suffix.lower() not in IMAGE_EXTENSIONS:
        print(f"WARN: output extension '{output_path.suffix}' is not image-like, appending .jpg", file=sys.stderr)
        output_path = output_path.with_suffix(output_path.suffix + ".jpg")
    elif len(frames) > 1 and output_path.suffix.lower() != ".mp4":
        print(f"WARN: output extension '{output_path.suffix}' is not .mp4, appending .mp4", file=sys.stderr)
        output_path = output_path.with_suffix(output_path.suffix + ".mp4")

    os.makedirs(output_path.parent, exist_ok=True)

    if len(frames) == 1:
        frames[0].save(str(output_path))
    else:
        export_to_video(frames, str(output_path), fps=args.output_fps)

    print(f"Saved: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main(
        tyro.cli(
            Args,
            description=__doc__,
            config=(tyro.conf.OmitArgPrefixes, tyro.conf.AvoidSubcommands),
        )
    )
