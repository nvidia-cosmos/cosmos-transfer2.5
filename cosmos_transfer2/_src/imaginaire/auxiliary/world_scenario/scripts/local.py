# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

"""
Functions for converting ClipGT data to world scenario v3 aligned renderer.
"""

import io
import re
import subprocess
import tarfile
from pathlib import Path
from typing import Dict, Generator, Optional, Tuple

import click
import imageio
import numpy as np
from loguru import logger

from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_loaders import load_scene
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.data_types import SceneData
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.config import SETTINGS
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.rendering.tiled_multi_camera_renderer import (
    TiledMultiCameraRenderer,
)
from cosmos_transfer2._src.imaginaire.auxiliary.world_scenario.utils.camera.ftheta import FThetaCamera


def write_video_from_generator(
    frame_generator: Generator[Tuple[np.ndarray, int], None, None],
    output_file: Path,
    fps: int = 30,
    overlay_video: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> int:
    """
    Write video from a frame generator.

    Args:
        frame_generator: Generator yielding (frame, frame_id) tuples
        output_file: Path to output video file
        fps: Frames per second for output video
        overlay_video: Optional video array to overlay on frames
        alpha: Alpha value for overlay blending

    Returns:
        Number of frames written
    """
    writer = imageio.get_writer(
        str(output_file),
        fps=fps,
        codec="libx264",
        macro_block_size=None,
        ffmpeg_params=["-crf", "18", "-preset", "slow"],
    )

    frames_written = 0
    try:
        for frame, frame_id in frame_generator:
            # Apply overlay if provided
            if overlay_video is not None and frame_id < len(overlay_video):
                frame = alpha * frame.astype(np.float32) + (1.0 - alpha) * overlay_video[frame_id].astype(np.float32)  # noqa: PLW2901
                frame = np.clip(frame, 0, 255).astype(np.uint8)  # noqa: PLW2901

            writer.append_data(frame)
            frames_written += 1
    finally:
        writer.close()

    return frames_written


def write_chunked_videos_from_generator(
    frame_generator: Generator[Tuple[np.ndarray, int], None, None],
    output_dir: Path,
    clip_id: str,
    camera_folder: str,
    chunk_size: int = 300,
    overlap: int = 30,
    max_chunks: int = -1,
    fps: int = 30,
    overlay_video: Optional[np.ndarray] = None,
    alpha: float = 0.5,
) -> list[Path]:
    """
    Write chunked videos from a frame generator.

    Args:
        frame_generator: Generator yielding (frame, frame_id) tuples
        output_dir: Output directory
        clip_id: Clip ID for naming
        camera_folder: Camera folder name
        chunk_size: Frames per chunk
        overlap: Overlap between chunks
        max_chunks: Maximum number of chunks (-1 for unlimited)
        fps: Frames per second
        overlay_video: Optional video array to overlay
        alpha: Alpha value for overlay

    Returns:
        List of output file paths
    """
    output_files = []
    current_chunk = 0
    chunk_buffer = []
    writer = None

    try:
        for frame, frame_id in frame_generator:
            # Apply overlay if provided
            if overlay_video is not None and frame_id < len(overlay_video):
                frame = alpha * frame.astype(np.float32) + (1.0 - alpha) * overlay_video[frame_id].astype(np.float32)  # noqa: PLW2901
                frame = np.clip(frame, 0, 255).astype(np.uint8)  # noqa: PLW2901

            # Add to buffer
            chunk_buffer.append(frame)

            # Check if we should start a new chunk
            if len(chunk_buffer) == chunk_size:
                # Write chunk
                output_file = output_dir / "hdmap" / camera_folder / f"{clip_id}_{current_chunk}.mp4"
                writer = imageio.get_writer(
                    str(output_file),
                    fps=fps,
                    codec="libx264",
                    macro_block_size=None,
                    ffmpeg_params=["-crf", "18", "-preset", "slow"],
                )

                for chunk_frame in chunk_buffer:
                    writer.append_data(chunk_frame)
                writer.close()

                output_files.append(output_file)
                logger.info(f"Saved chunk {current_chunk}: {output_file}")

                # Prepare for next chunk with overlap
                if overlap > 0:
                    chunk_buffer = chunk_buffer[-overlap:]
                else:
                    chunk_buffer = []

                current_chunk += 1

                # Check max chunks
                if max_chunks > 0 and current_chunk >= max_chunks:
                    break

        # Write remaining frames if any
        if chunk_buffer and (max_chunks < 0 or current_chunk < max_chunks):
            output_file = output_dir / "hdmap" / camera_folder / f"{clip_id}_{current_chunk}.mp4"
            writer = imageio.get_writer(
                str(output_file),
                fps=fps,
                codec="libx264",
                macro_block_size=None,
                ffmpeg_params=["-crf", "18", "-preset", "slow"],
            )

            for chunk_frame in chunk_buffer:
                writer.append_data(chunk_frame)
            writer.close()

            output_files.append(output_file)
            logger.info(f"Saved final chunk {current_chunk}: {output_file}")

    except Exception as e:
        logger.error(f"Error writing video: {e}")
        if writer is not None:
            writer.close()
        raise

    return output_files


def convert_scene_data_for_rendering(
    scene_data: SceneData,
    camera_names: list[str],
    resize_hw: Optional[Tuple[int, int]] = None,
) -> tuple[Dict[str, FThetaCamera], Dict[str, np.ndarray]]:
    """
    Convert SceneData to the format needed by the rendering pipeline.

    Args:
        scene_data: Unified scene data representation
        camera_names: List of camera names to process
        resize_hw: Optional resize resolution (height, width)

    Returns:
        Tuple of (camera_models, camera_poses)
    """
    logger.debug("Converting scene data for rendering")

    # Create camera models
    all_camera_models = {}
    for camera_name in camera_names:
        if camera_name not in scene_data.camera_models:
            logger.warning(f"Camera {camera_name} not found in scene data")
            continue

        camera_model = scene_data.camera_models[camera_name]

        camera = camera_model
        if resize_hw:
            resize_h, resize_w = resize_hw
            if camera.height != resize_h or camera.width != resize_w:
                # Create a copy before resizing to avoid mutating the stored model
                camera = FThetaCamera.from_numpy(camera_model.intrinsics.copy())
                scale_h = resize_h / camera_model.height
                scale_w = resize_w / camera_model.width
                if abs(scale_h - scale_w) > 0.02:
                    logger.warning(f"Non-uniform scaling for {camera_name}: {scale_h:.3f} vs {scale_w:.3f}")
                camera.rescale(ratio_h=scale_h, ratio_w=scale_w)

        all_camera_models[camera_name] = camera

    # Create camera poses
    all_camera_poses = {}
    for camera_name in camera_names:
        if camera_name not in scene_data.camera_extrinsics or camera_name not in scene_data.camera_models:
            continue

        camera_to_ego = scene_data.camera_extrinsics[camera_name]

        # Build camera poses for all frames
        camera_poses = []
        for ego_pose in scene_data.ego_poses:
            ego_to_world = ego_pose.transformation_matrix
            camera_to_world = ego_to_world @ camera_to_ego
            camera_poses.append(camera_to_world)

        all_camera_poses[camera_name] = np.array(camera_poses)

    logger.debug(f"Converted {len(all_camera_models)} cameras, {scene_data.num_frames} frames")

    return all_camera_models, all_camera_poses


def override_camera_poses_with_tar(
    all_camera_poses: Dict[str, np.ndarray],
    camera_names: list[str],
    novel_pose_tar: str,
) -> Dict[str, np.ndarray]:
    """Override camera poses using a tar containing per-frame pose .npy files.

    Expected tar entries are named like "000000.pose.<camera_underscore_name>.npy" where the
    camera underscore name is derived from the rig sensor name by replacing ":" with "_".
    """

    def canon(name: str) -> str:
        return name.replace(":", "_")

    poses_from_tar: dict[str, np.ndarray] = {}
    with tarfile.open(novel_pose_tar, "r") as tfh:
        for member in tfh.getmembers():
            if not member.isfile():
                continue
            base = Path(member.name).name
            # Expect files like 000000.pose.camera_front_wide_120fov.npy
            if base.endswith(".npy") and ".pose." in base:
                f = tfh.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                arr = np.load(io.BytesIO(data))
                poses_from_tar[base] = arr

    frame_key_regex = re.compile(r"(\d{6})\.pose\.")
    for cam in camera_names:
        cam_key = canon(cam)
        keyed: list[tuple[str, int]] = []
        for k in poses_from_tar.keys():
            if not k.endswith(f"pose.{cam_key}.npy"):
                continue
            m = frame_key_regex.search(k)
            if not m:
                raise ValueError(f"Novel pose filename missing 6-digit frame segment before 'pose.': {k}")
            keyed.append((k, int(m.group(1))))
        if not keyed:
            # Collect candidate keys to help debugging
            candidates = [k for k in poses_from_tar.keys() if f"pose.{cam_key}.npy" in k]
            raise ValueError(
                f"Novel pose tar has no entries for requested camera '{cam}' (expected suffix 'pose.{cam_key}.npy'). "
                f"Found {len(candidates)} candidate keys for this camera: {candidates[:5]}{' ...' if len(candidates) > 5 else ''}"
            )
        keyed.sort(key=lambda x: x[1])
        pose_list = [poses_from_tar[k] for k, _ in keyed]
        all_camera_poses[cam] = np.stack(pose_list, axis=0)

    logger.info("Using novel poses from tar for rendering")
    return all_camera_poses


def read_video_simple(input_file: str, height: int, width: int) -> np.ndarray:
    """
    Read entire video into numpy array with specified dimensions.

    Args:
        input_file: Path to video file
        height: Output height
        width: Output width

    Returns:
        numpy array of shape (num_frames, height, width, 3)
    """
    cmd = [
        "ffmpeg",
        "-i",
        input_file,
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-vcodec",
        "rawvideo",
        "-loglevel",
        "error",
        "-",
    ]

    result = subprocess.run(cmd, capture_output=True, check=True)
    frames = np.frombuffer(result.stdout, dtype=np.uint8)

    # Reshape to 4D array
    return frames.reshape((-1, height, width, 3))


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.option(
    "--camera-names",
    default=["camera_front_wide_120fov"],
    multiple=True,
    help="Camera sensor name (use underscores, e.g., camera_front_wide_120fov)",
)
@click.option("--output-dir", default="output_final", help="Output directory for rendered videos")
@click.option("--max-frames", default=-1, help="Maximum number of frames to render (-1 for all)")
@click.option(
    "--chunk-output/--no-chunk-output",
    is_flag=True,
    default=False,
    help="Save as single video instead of chunked videos",
)
@click.option(
    "--overlay-camera/--no-overlay-camera",
    default=False,
    is_flag=True,
    help="Overlay camera view on the HD map",
)
@click.option(
    "--alpha",
    default=0.5,
    help="Alpha value for camera overlay",
)
@click.option(
    "--use-persistent-vbos/--no-persistent-vbos",
    default=True,
    is_flag=True,
    help="Use persistent VBOs for static geometry",
)
@click.option(
    "--multi-sample",
    default=4,
    type=int,
    help="Number of samples for multisampling anti-aliasing (MSAA). Use 1 to disable, 4 for 4x MSAA (default).",
)
@click.option(
    "--novel-pose-tar",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Optional tar with novel poses (keys like 000000.pose.camera_front_wide_120fov.npy)",
)
@click.option(
    "--clip-id",
    default=None,
    type=str,
    help="Clip ID to render (relevant for RDS-HQ)",
)
def main(
    data_dir: str,
    camera_names: list[str],
    output_dir: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    use_persistent_vbos: bool,
    multi_sample: int,
    novel_pose_tar: Optional[str],
    clip_id: Optional[str],
) -> None:
    """Render HD map for a given data directory (ClipGT or RDS-HQ format)."""

    logger.info(f"Loading data from: {data_dir}")
    data_path = Path(data_dir)

    # Load scene data using the new loader system (auto-detects ClipGT or RDS-HQ)
    # First load with None to get all available cameras
    scene_data = load_scene(
        data_path,
        camera_names=None,  # Load all available cameras first
        max_frames=max_frames,
        input_pose_fps=SETTINGS["INPUT_POSE_FPS"],
        resize_resolution_hw=SETTINGS["RESIZE_RESOLUTION"],
        clip_id=clip_id,
    )

    # Parse camera names after loading to see what's available
    if "all" in camera_names:
        camera_names = list(scene_data.camera_models.keys())

    logger.info(f"Loaded scene {scene_data.scene_id}:")
    logger.info(f"  - Frames: {scene_data.num_frames}")
    logger.info(f"  - Dynamic objects: {len(scene_data.dynamic_objects)}")
    logger.info("  - Map elements loaded")
    logger.info(f"  - Cameras: {', '.join(camera_names)}")

    # Convert to rendering format
    all_camera_models, all_camera_poses = convert_scene_data_for_rendering(
        scene_data,
        camera_names,
        SETTINGS["RESIZE_RESOLUTION"],
    )

    # Optionally override camera poses with a novel trajectory from tar
    if novel_pose_tar is not None:
        all_camera_poses = override_camera_poses_with_tar(all_camera_poses, camera_names, novel_pose_tar)

    # Always use tiled multi-camera renderer (works for single camera too)
    logger.info(f"Using tiled multi-camera renderer for {len(camera_names)} camera(s)")
    render_multi_camera_tiled(
        all_camera_models,
        all_camera_poses,
        scene_data,
        camera_names,
        output_dir,
        scene_data.scene_id,
        max_frames,
        chunk_output,
        overlay_camera,
        alpha,
        data_path,
        use_persistent_vbos,
        multi_sample,
    )


def render_multi_camera_tiled(
    all_camera_models: Dict[str, FThetaCamera],
    all_camera_poses: Dict[str, np.ndarray],
    scene_data: SceneData,
    camera_names: list[str],
    output_dir: str,
    clip_id: str,
    max_frames: int,
    chunk_output: bool,
    overlay_camera: bool,
    alpha: float,
    clipgt_path: Path,
    use_persistent_vbos: bool,
    multi_sample: int,
    simplified_output: bool = False,
) -> None:
    """Render using tiled multi-camera renderer for maximum performance.

    Args:
        simplified_output: If True, use simplified output structure (scene_id/scene_id_camera.mp4).
                          If False, use full structure (hdmap/ftheta_camera/clip_id.mp4).
    """

    # Create tiled multi-camera renderer
    logger.info("Creating tiled multi-camera renderer...")

    # Filter camera models to only requested cameras
    selected_camera_models = {name: all_camera_models[name] for name in camera_names}

    tiled_renderer = TiledMultiCameraRenderer(
        camera_models=selected_camera_models,
        scene_data=scene_data,
        hdmap_color_version="v3",
        bbox_color_version="v3",
        enable_height_filter=False,
        use_persistent_vbos=use_persistent_vbos,
        multi_sample=multi_sample,
    )

    # Determine frames to render
    num_frames = len(all_camera_poses[camera_names[0]])
    if max_frames > 0:
        num_frames = min(num_frames, max_frames)

    # Load camera videos if overlaying
    video_arrays = {}
    if overlay_camera:
        for camera_name in camera_names:
            mapped_name = camera_name.replace(":", "_")
            video_path = clipgt_path / f"{clip_id}.{mapped_name}.mp4"
            if video_path.exists():
                camera_model = all_camera_models[camera_name]
                h, w = camera_model.height, camera_model.width
                video_arrays[camera_name] = read_video_simple(video_path.as_posix(), h, w)
                logger.info(f"Loaded video for {camera_name} with {len(video_arrays[camera_name])} frames")

    # Prepare output paths and writers
    output_path = Path(output_dir)
    writers = {}
    output_files = {camera_name: [] for camera_name in camera_names}
    chunk_buffers = {camera_name: [] for camera_name in camera_names}
    current_chunks = {camera_name: 0 for camera_name in camera_names}

    # Create output directories based on output structure
    if simplified_output:
        # Simplified: output_dir/scene_id/
        clip_save_root = output_path / clip_id
        clip_save_root.mkdir(parents=True, exist_ok=True)
    else:
        # Full: output_dir/hdmap/ftheta_camera/
        for camera_name in camera_names:
            mapped_name = camera_name.replace(":", "_")
            camera_folder = f"ftheta_{mapped_name}"
            (output_path / "hdmap" / camera_folder).mkdir(parents=True, exist_ok=True)

    # Process frames
    logger.info(f"Starting tiled multi-camera render for {num_frames} frames...")

    for frame_id in range(num_frames):
        # Prepare camera poses for this frame
        camera_poses_frame = {camera_name: all_camera_poses[camera_name][frame_id] for camera_name in camera_names}

        # Render all cameras in a single OpenGL pass
        rendered_frames = tiled_renderer.render_all_cameras(camera_poses_frame, frame_id)

        # Process each rendered frame
        for camera_name, rendered_frame in rendered_frames.items():
            # Apply overlay if needed
            frame = rendered_frame
            if overlay_camera and camera_name in video_arrays and frame_id < len(video_arrays[camera_name]):
                overlay_frame = alpha * frame.astype(np.float32) + (1.0 - alpha) * video_arrays[camera_name][
                    frame_id
                ].astype(np.float32)
                frame = np.clip(overlay_frame, 0, 255).astype(np.uint8)

            if chunk_output:
                # Handle chunked output
                chunk_buffers[camera_name].append(frame)

                if len(chunk_buffers[camera_name]) == SETTINGS["TARGET_CHUNK_FRAME"]:
                    # Write chunk
                    mapped_name = camera_name.replace(":", "_")
                    if simplified_output:
                        output_file = clip_save_root / f"{clip_id}.{mapped_name}_{current_chunks[camera_name]}.mp4"
                    else:
                        camera_folder = f"ftheta_{mapped_name}"
                        output_file = (
                            output_path / "hdmap" / camera_folder / f"{clip_id}_{current_chunks[camera_name]}.mp4"
                        )

                    writer = imageio.get_writer(
                        str(output_file),
                        fps=SETTINGS["TARGET_RENDER_FPS"],
                        codec="libx264",
                        macro_block_size=None,
                        ffmpeg_params=["-crf", "18", "-preset", "slow"],
                    )

                    for chunk_frame in chunk_buffers[camera_name]:
                        writer.append_data(chunk_frame)
                    writer.close()

                    output_files[camera_name].append(output_file)
                    logger.info(f"Saved chunk {current_chunks[camera_name]} for {camera_name}")

                    # Prepare for next chunk with overlap
                    if SETTINGS["OVERLAP_FRAME"] > 0:
                        chunk_buffers[camera_name] = chunk_buffers[camera_name][-SETTINGS["OVERLAP_FRAME"] :]
                    else:
                        chunk_buffers[camera_name] = []

                    current_chunks[camera_name] += 1

                    # Check max chunks
                    if SETTINGS.get("MAX_CHUNK", -1) > 0 and current_chunks[camera_name] >= SETTINGS["MAX_CHUNK"]:
                        break
            else:
                # Single video output - initialize writer if needed
                if camera_name not in writers:
                    mapped_name = camera_name.replace(":", "_")
                    if simplified_output:
                        output_file = clip_save_root / f"{clip_id}.{mapped_name}.mp4"
                    else:
                        camera_folder = f"ftheta_{mapped_name}"
                        output_file = output_path / "hdmap" / camera_folder / f"{clip_id}.mp4"
                    writers[camera_name] = imageio.get_writer(
                        str(output_file),
                        fps=SETTINGS["TARGET_RENDER_FPS"],
                        codec="libx264",
                        macro_block_size=None,
                        ffmpeg_params=["-crf", "18", "-preset", "slow"],
                    )
                    output_files[camera_name] = [output_file]

                writers[camera_name].append_data(frame)

        # Log progress
        if (frame_id + 1) % 100 == 0:
            logger.info(f"Processed {frame_id + 1}/{num_frames} frames")

    # Write remaining frames and cleanup
    if chunk_output:
        # Write remaining chunks
        for camera_name in camera_names:
            if chunk_buffers[camera_name]:
                mapped_name = camera_name.replace(":", "_")
                if simplified_output:
                    output_file = clip_save_root / f"{clip_id}.{mapped_name}_{current_chunks[camera_name]}.mp4"
                else:
                    camera_folder = f"ftheta_{mapped_name}"
                    output_file = output_path / "hdmap" / camera_folder / f"{clip_id}_{current_chunks[camera_name]}.mp4"

                writer = imageio.get_writer(
                    str(output_file),
                    fps=SETTINGS["TARGET_RENDER_FPS"],
                    codec="libx264",
                    macro_block_size=None,
                    ffmpeg_params=["-crf", "18", "-preset", "slow"],
                )

                for chunk_frame in chunk_buffers[camera_name]:
                    writer.append_data(chunk_frame)
                writer.close()

                output_files[camera_name].append(output_file)
                logger.info(f"Saved final chunk for {camera_name}")
    else:
        # Close single video writers
        for _, writer in writers.items():
            writer.close()

    # Cleanup renderer
    tiled_renderer.cleanup()

    # Report results
    for camera_name in camera_names:
        for output_file in output_files[camera_name]:
            logger.info(f"Saved: {output_file}")

    if simplified_output:
        logger.info(f"Processing completed successfully! Files saved to: {clip_save_root}")
    else:
        logger.info(f"Processing completed successfully! Files saved to: {output_path / 'hdmap'}")


if __name__ == "__main__":
    main()
