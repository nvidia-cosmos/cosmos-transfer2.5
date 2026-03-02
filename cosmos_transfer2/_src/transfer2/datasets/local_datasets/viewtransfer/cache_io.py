"""Cache IO helpers (locks + atomic writes) for view-transfer datasets.

Conventions:
- Rendered RGB/gray video: H.264 in MP4 (small, viewable)
- Depth: FFV1 in MKV, gray16le uint16 millimeters (0 = invalid)
- Binary masks: FFV1 in MKV, gray uint8 (0 or 255), threshold on load
"""

from __future__ import annotations

import os
import subprocess
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Literal, Tuple

import numpy as np
import torch
from torchcodec.decoders import VideoDecoder

# ----------------------------
# Filesystem utilities
# ----------------------------


def ensure_parent_dir(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _tmp_sibling(path: Path, suffix: str) -> Path:
    return path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp{suffix}")


@contextmanager
def atomic_write_path(final_path: str | Path, *, tmp_suffix: str) -> Iterator[Path]:
    """Yield a temporary sibling path; atomically replace on success."""
    final_path = Path(final_path)
    ensure_parent_dir(final_path)
    tmp = _tmp_sibling(final_path, tmp_suffix)
    try:
        yield tmp
        os.replace(tmp, final_path)
    finally:
        # Best-effort cleanup if something failed before replace.
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


@contextmanager
def file_lock(lock_path: str | Path, *, timeout_sec: float = 60.0, poll_sec: float = 0.1) -> Iterator[None]:
    """Simple lockfile using atomic creation."""
    lock_path = Path(lock_path)
    ensure_parent_dir(lock_path)
    deadline = time.time() + float(timeout_sec)

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()} {time.time()}\n")
            break
        except FileExistsError:
            if time.time() > deadline:
                raise TimeoutError(f"Timeout acquiring lock: {lock_path}")
            time.sleep(float(poll_sec))

    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def atomic_save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    with atomic_write_path(path, tmp_suffix=".npz") as tmp:
        np.savez(tmp, **arrays)


# ----------------------------
# FFMPEG rawvideo helpers
# ----------------------------

RawPixFmt = Literal["rgb24", "gray", "gray16le"]


def _run_ffmpeg_pipe_write(cmd: list[str], payload: bytes, *, out_path: Path) -> None:
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    assert proc.stdin is not None
    try:
        proc.stdin.write(payload)
        proc.stdin.close()
        stderr = proc.stderr.read() if proc.stderr is not None else b""
        rc = proc.wait()
    except Exception as e:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.kill()
        _, stderr = proc.communicate()
        msg = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed while writing {out_path}: {msg}") from e

    if rc != 0:
        msg = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"ffmpeg failed with return code {rc} for {out_path}: {msg}")


def _ffmpeg_write_rawvideo(
    *,
    out_path: Path,
    fps: float,
    width: int,
    height: int,
    in_pix_fmt: RawPixFmt,
    frames_bytes: bytes,
    video_codec_args: list[str],
) -> None:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        in_pix_fmt,
        "-s:v",
        f"{width}x{height}",
        "-r",
        str(float(fps)),
        "-i",
        "-",
        "-an",
        *video_codec_args,
        str(out_path),
    ]
    _run_ffmpeg_pipe_write(cmd, frames_bytes, out_path=out_path)


def _ffmpeg_read_rawvideo(*, path: Path, out_pix_fmt: RawPixFmt) -> bytes:
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        out_pix_fmt,
        "-",
    ]
    return subprocess.check_output(cmd)


# ----------------------------
# Rendered video (H.264 MP4)
# ----------------------------


def _to_uint8_image(frame: np.ndarray) -> np.ndarray:
    """Accept uint8, int in [0,255], or float in [0,1] / [0,255]."""
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    if np.issubdtype(frame.dtype, np.floating):
        if not np.isfinite(frame).all():
            raise ValueError("Float image contains NaN/Inf values.")
        mn = float(frame.min())
        mx = float(frame.max())
        if mn < 0:
            raise ValueError(f"Float image has negative values: min={mn}, max={mx}")
        if mx <= 1.0:
            return np.clip(frame * 255.0, 0.0, 255.0).round().astype(np.uint8)
        if mx <= 255.0:
            return np.clip(frame, 0.0, 255.0).round().astype(np.uint8)
        raise ValueError(f"Float image out of expected range [0,1] or [0,255]: min={mn}, max={mx}")
    if np.issubdtype(frame.dtype, np.integer):
        mn = int(frame.min())
        mx = int(frame.max())
        if mn < 0 or mx > 255:
            raise ValueError(f"Integer image out of uint8 range [0,255]: min={mn}, max={mx}")
        return frame.astype(np.uint8)
    raise ValueError(f"Unsupported dtype {frame.dtype}")


def _normalize_render_frame(frame: np.ndarray) -> Tuple[np.ndarray, RawPixFmt]:
    """
    Returns (frame_uint8, pix_fmt) where pix_fmt is rgb24 or gray.
    Accepts (H,W), (H,W,1), (H,W,3).
    """
    f = _to_uint8_image(frame)
    if f.ndim == 2:
        return f, "gray"
    if f.ndim == 3 and f.shape[2] == 1:
        return f[..., 0], "gray"
    if f.ndim == 3 and f.shape[2] == 3:
        return f, "rgb24"
    raise ValueError(f"Expected (H,W), (H,W,1), or (H,W,3), got {f.shape}")


def save_render_mp4_h264(
    *,
    path: str | Path,
    frames: Iterable[np.ndarray] | list[np.ndarray],
    fps: float,
    crf: int = 18,
    preset: str = "veryfast",
) -> None:
    """Save rendered RGB/gray frames as H.264 MP4 (viewable, small)."""
    path = Path(path).with_suffix(".mp4")

    frame_iter = iter(frames)
    try:
        first = next(frame_iter)
    except StopIteration as e:
        raise ValueError(f"No frames provided for video write: {path}") from e

    first_u8, pix = _normalize_render_frame(first)
    h, w = first_u8.shape[:2]

    # Stream bytes: first + rest
    chunks = [np.ascontiguousarray(first_u8).tobytes()]
    for fr in frame_iter:
        u8, pix2 = _normalize_render_frame(fr)
        if pix2 != pix:
            raise ValueError(f"Mixed color modes in render stream: {pix} then {pix2}")
        if u8.shape[0] != h or u8.shape[1] != w:
            raise RuntimeError(f"Frame size mismatch: got {u8.shape[1]}x{u8.shape[0]}, expected {w}x{h}")
        chunks.append(np.ascontiguousarray(u8).tobytes())
    payload = b"".join(chunks)

    # H.264 output settings: yuv420p for maximal compatibility
    codec_args = [
        "-c:v",
        "libx264",
        "-preset",
        str(preset),
        "-crf",
        str(int(crf)),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    with atomic_write_path(path, tmp_suffix=".mp4") as tmp:
        _ffmpeg_write_rawvideo(
            out_path=tmp,
            fps=float(fps),
            width=w,
            height=h,
            in_pix_fmt=pix,
            frames_bytes=payload,
            video_codec_args=codec_args,
        )


# ----------------------------
# Depth (FFV1 MKV, gray16le)
# ----------------------------


def save_depth_mkv_ffv1_mm_u16(*, depth_mm: np.ndarray, path: str | Path, fps: float) -> None:
    """
    Save depth as lossless MKV:
      - codec: FFV1
      - pix_fmt: gray16le
      - data: uint16 millimeters (0 = invalid)
    Accepts (H,W) or (T,H,W).
    """
    path = Path(path).with_suffix(".mkv")
    depth_mm = np.asarray(depth_mm)

    if depth_mm.dtype != np.uint16:
        raise ValueError(f"Expected uint16 depth millimeters, got {depth_mm.dtype}")
    if depth_mm.ndim == 2:
        depth_mm = depth_mm[None, ...]
    if depth_mm.ndim != 3:
        raise ValueError(f"Expected (H,W) or (T,H,W), got {depth_mm.shape}")

    _, h, w = depth_mm.shape
    payload = np.ascontiguousarray(depth_mm).tobytes()

    codec_args = [
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "gray16le",
    ]

    with atomic_write_path(path, tmp_suffix=".mkv") as tmp:
        _ffmpeg_write_rawvideo(
            out_path=tmp,
            fps=float(fps),
            width=w,
            height=h,
            in_pix_fmt="gray16le",
            frames_bytes=payload,
            video_codec_args=codec_args,
        )


def load_depth_mkv_ffv1_mm_u16(
    *,
    path: str | Path,
    width: int,
    height: int,
    invalid_value: int = 0,
    return_float64: bool = False,
) -> np.ndarray:
    """
    Load depth MKV written by save_depth_mkv_ffv1_mm_u16.

    Returns (T,H,W) float meters with NaN for invalid (depth==invalid_value).
    """
    path = Path(path)
    raw = _ffmpeg_read_rawvideo(path=path, out_pix_fmt="gray16le")

    frame_bytes = int(width) * int(height) * 2
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(f"Unexpected raw size {len(raw)} not divisible by frame size {frame_bytes}")

    t = len(raw) // frame_bytes
    depth_mm = np.frombuffer(raw, dtype=np.uint16).reshape(t, int(height), int(width))

    dtype = np.float64 if return_float64 else np.float32
    depth_m = depth_mm.astype(dtype) * 0.001
    depth_m[depth_mm == np.uint16(invalid_value)] = np.nan
    return depth_m


# ----------------------------
# Binary masks (FFV1 MKV, gray8)
# ----------------------------


def save_mask_mkv_ffv1(*, masks: np.ndarray, path: str | Path, fps: float) -> None:
    """
    Save binary masks as lossless MKV:
      - codec: FFV1
      - pix_fmt: gray (8-bit)
      - stored frames: uint8 0/255
    Accepts (T,H,W) or (H,W).
    """
    path = Path(path).with_suffix(".mkv")
    masks = np.asarray(masks)

    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (H,W), got {masks.shape}")

    # Normalize to uint8 0/255
    if masks.dtype == np.bool_:
        frames_u8 = masks.astype(np.uint8) * 255
    else:
        frames_u8 = (masks > 0).astype(np.uint8) * 255

    _, h, w = frames_u8.shape
    payload = np.ascontiguousarray(frames_u8).tobytes()

    codec_args = [
        "-c:v",
        "ffv1",
        "-level",
        "3",
        "-g",
        "1",
        "-pix_fmt",
        "gray",
    ]

    with atomic_write_path(path, tmp_suffix=".mkv") as tmp:
        _ffmpeg_write_rawvideo(
            out_path=tmp,
            fps=float(fps),
            width=w,
            height=h,
            in_pix_fmt="gray",
            frames_bytes=payload,
            video_codec_args=codec_args,
        )


def load_mask_mkv_ffv1(*, path: str | Path, width: int, height: int) -> np.ndarray:
    """Load masks saved by save_mask_mkv_ffv1. Returns (T,H,W) bool."""
    path = Path(path)
    raw = _ffmpeg_read_rawvideo(path=path, out_pix_fmt="gray")

    frame_bytes = int(width) * int(height)
    if len(raw) % frame_bytes != 0:
        raise RuntimeError(f"Unexpected raw size {len(raw)} not divisible by frame size {frame_bytes}")

    t = len(raw) // frame_bytes
    frames_u8 = np.frombuffer(raw, dtype=np.uint8).reshape(t, int(height), int(width))
    return frames_u8  # robust threshold


# ----------------------------
# VideoDecoder helpers
# ----------------------------


def load_full_video_frames(
    video_path: Path, decoder_device: str, dimension_order: Literal["NCHW", "NHWC"] = "NCHW"
) -> torch.Tensor:
    """Return frames as (N,C,H,W) uint8."""
    decoder = VideoDecoder(str(video_path), device=decoder_device, dimension_order=dimension_order)
    return decoder.get_frames_at(list(range(len(decoder)))).data  # N C H W uint8


def get_video_fps(path: str | Path) -> float:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video not found: {path}")

    decoder = VideoDecoder(str(path), device="cpu")
    fps = decoder.metadata.average_fps
    if fps is None:
        raise RuntimeError(f"Decoder returned None fps for: {path}")
    fps = float(fps)
    if fps <= 0:
        raise RuntimeError(f"Invalid fps={fps} for: {path}")
    return fps
