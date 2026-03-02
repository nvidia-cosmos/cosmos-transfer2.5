"""Path templates for Agibot view-transfer raw data and generated caches."""

from enum import Enum
from pathlib import Path


class CacheCategory(str, Enum):
    DEPTH = "depth"
    CAMERA_PARAMETERS = "camera"
    RENDER = "render"
    POINT_CLOUD = "pcd"


def observations_root(raw_root: str | Path) -> Path:
    return Path(raw_root) / "observations"


def episode_video_dir(raw_root: str | Path, task: str, episode: str) -> Path:
    return observations_root(raw_root) / task / episode / "videos"


def raw_video_path(raw_root: str | Path, task: str, episode: str, clip_name: str) -> Path:
    return episode_video_dir(raw_root, task, episode) / f"{clip_name}_color.mp4"


def raw_depth_pattern(raw_root: str | Path, task: str, episode: str, clip_name: str) -> Path:
    return observations_root(raw_root) / task / episode / "depth" / f"{clip_name}_depth_%06d.png"


def raw_proprio_h5_path(raw_root: str | Path, task: str, episode: str) -> Path:
    return Path(raw_root) / "proprio_stats" / task / episode / "proprio_stats.h5"


def raw_camera_info_json_path(raw_root: str | Path, task: str, episode: str) -> Path:
    return Path(raw_root) / "parameters" / task / episode / "parameters" / "camera" / "rs_camera_info.json"


def cache_episode_dir(cache_root: str | Path, category: CacheCategory, task: str, episode: str) -> Path:
    assert isinstance(category, CacheCategory), f"Expected CacheCategory, got {type(category)}"
    return Path(cache_root) / category.value / task / episode


def _depth_suffix(depth_estimator: str) -> str:
    return "" if depth_estimator == "agibot" else f"__{depth_estimator}"


def depth_cache_path(
    cache_root: str | Path,
    task: str,
    episode: str,
    source_clip: str,
    depth_estimator: str,
) -> Path:
    suffix = _depth_suffix(depth_estimator)
    return cache_episode_dir(cache_root, CacheCategory.DEPTH, task, episode) / f"{source_clip}__depth{suffix}.mkv"


def camera_parameters_cache_path(cache_root: str | Path, task: str, episode: str, clip_name: str) -> Path:
    return cache_episode_dir(cache_root, CacheCategory.CAMERA_PARAMETERS, task, episode) / f"{clip_name}_parameters.npz"


def point_cloud_cache_path(
    cache_root: str | Path,
    task: str,
    episode: str,
    source_clip: str,
    target_clip: str,
    depth_estimator: str,
) -> Path:
    return (
        cache_episode_dir(cache_root, CacheCategory.POINT_CLOUD, task, episode)
        / f"{source_clip}__to__{target_clip}{_depth_suffix(depth_estimator)}.npz"
    )


def render_cache_paths(
    cache_root: str | Path,
    task: str,
    episode: str,
    source_clip: str,
    target_clip: str,
    depth_estimator: str,
) -> tuple[Path, Path]:
    suffix = _depth_suffix(depth_estimator)
    render_dir = cache_episode_dir(cache_root, CacheCategory.RENDER, task, episode)
    render_path = render_dir / f"{source_clip}__to__{target_clip}{suffix}.mp4"
    mask_path = render_dir / f"{source_clip}__to__{target_clip}__mask{suffix}.mkv"
    return render_path, mask_path
