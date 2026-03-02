"""Camera intrinsic cache generation for Agibot view-transfer."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cosmos_transfer2._src.imaginaire.utils import log
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.cache_io import (
    atomic_save_npz,
    file_lock,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.fk_extrinsics import (
    generate_fk_extrinsics_for_episode,
)
from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import (
    CacheCategory,
    cache_episode_dir,
    camera_parameters_cache_path,
    raw_camera_info_json_path,
)

_FOCAL_LENGTH_MM = 1.93
_HEAD_HORIZONTAL_APERTURE_MM = 3.90
_HEAD_VERTICAL_APERTURE_MM = 2.453
_OTHER_HORIZONTAL_APERTURE_MM = 3.48
_OTHER_VERTICAL_APERTURE_MM = 2.40


def get_K_from_properties(
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
) -> np.ndarray:
    """
    USD intrinsics are given as focal length (mm) and aperture size (mm).
    Convert to pinhole K matrix for given image dimensions.
    """
    if width <= 0 or height <= 0:
        raise ValueError(f"Image width/height must be positive. Got width={width}, height={height}.")
    if focal_length <= 0.0:
        raise ValueError(f"focal_length must be positive. Got {focal_length}.")
    if horizontal_aperture <= 0.0 or vertical_aperture <= 0.0:
        raise ValueError(
            "horizontal_aperture and vertical_aperture must be positive. "
            f"Got horizontal_aperture={horizontal_aperture}, vertical_aperture={vertical_aperture}."
        )

    fx = focal_length * width / horizontal_aperture
    fy = focal_length * height / vertical_aperture
    cx = width / 2.0
    cy = height / 2.0
    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def _resolve_image_size_from_camera_info(camera_info: dict, camera_name: str) -> tuple[int, int]:
    width, height = None, None
    for _, camera_parameters in camera_info.items():
        if camera_parameters["name"] == camera_name:
            width, height = (
                int(camera_parameters["width"]),
                int(camera_parameters["height"]),
            )
            break
    assert width is not None and height is not None, f"Camera {camera_name} not found in camera info."
    return width, height


def _apertures_for_camera(camera_name: str) -> tuple[float, float]:
    if camera_name == "head":
        return _HEAD_HORIZONTAL_APERTURE_MM, _HEAD_VERTICAL_APERTURE_MM
    return _OTHER_HORIZONTAL_APERTURE_MM, _OTHER_VERTICAL_APERTURE_MM


def _generate_camera_parameters_for_episode(
    *,
    dataset_dir: str,
    cache_root: str,
    task: str,
    episode: str,
    clip_names: tuple[str, ...],
    urdf_path: str,
    usd_path: str | None,
    camera_prims: dict[str, str] | None,
    base_frame: str,
) -> None:
    if not clip_names:
        raise ValueError("clip_names must be non-empty to generate camera parameters.")

    camera_info_json = raw_camera_info_json_path(raw_root=dataset_dir, task=task, episode=episode)
    camera_info = json.loads(camera_info_json.read_text())

    extrinsics = generate_fk_extrinsics_for_episode(
        dataset_dir=dataset_dir,
        task=task,
        episode=episode,
        clip_names=clip_names,
        urdf_path=urdf_path,
        usd_path=usd_path,
        camera_prims=camera_prims,
        base_frame=base_frame,
    )
    episode_length = extrinsics[clip_names[0]].shape[0]

    for clip_name in clip_names:
        # Process intrinsics
        width, height = _resolve_image_size_from_camera_info(camera_info, clip_name)
        horizontal_aperture, vertical_aperture = _apertures_for_camera(clip_name)
        intrinsics = get_K_from_properties(
            width=width,
            height=height,
            focal_length=_FOCAL_LENGTH_MM,
            horizontal_aperture=horizontal_aperture,
            vertical_aperture=vertical_aperture,
        )
        intrinsics = np.array([intrinsics] * episode_length)
        # Get extrinsics as world-to-camera (w2c) matrices
        w2c = np.linalg.inv(extrinsics[clip_name])
        out_path = camera_parameters_cache_path(cache_root, task, episode, clip_name)
        atomic_save_npz(out_path, intrinsics=intrinsics, w2c=w2c)


def ensure_camera_parameters_cache(
    *,
    dataset_dir: str,
    cache_root: str,
    task: str,
    episode: str,
    clip_name: str,
    clip_names: tuple[str, ...],
    urdf_path: str,
    usd_path: str | None,
    camera_prims: dict[str, str] | None,
    base_frame: str,
    lock_timeout_sec: float,
    lock_poll_sec: float,
) -> Path:
    out_path = camera_parameters_cache_path(cache_root, task, episode, clip_name)
    if out_path.exists():
        return out_path

    log.info(
        f"Camera parameters cache not found for {task}/{episode}/{clip_name}. Preparing camera parameters cache..."
    )
    episode_camera_dir = cache_episode_dir(cache_root, CacheCategory.CAMERA_PARAMETERS, task, episode)
    lock_path = episode_camera_dir / "camera_parameters_generation.lock"
    with file_lock(lock_path, timeout_sec=lock_timeout_sec, poll_sec=lock_poll_sec):
        _generate_camera_parameters_for_episode(
            dataset_dir=dataset_dir,
            cache_root=cache_root,
            task=task,
            episode=episode,
            clip_names=clip_names,
            urdf_path=urdf_path,
            usd_path=usd_path,
            camera_prims=camera_prims,
            base_frame=base_frame,
        )

    if not out_path.exists():
        raise RuntimeError(f"Did not create expected camera parameters cache: {out_path}")
    return out_path
