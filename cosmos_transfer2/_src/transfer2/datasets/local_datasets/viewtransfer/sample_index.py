"""Indexing helpers for Agibot single-view view-transfer dataset."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from tqdm import tqdm

from cosmos_transfer2._src.transfer2.datasets.local_datasets.viewtransfer.path_templates import raw_video_path


class PairMode(str, Enum):
    ALL_ORDERED_NONSELF = "all_ordered_nonself"
    ONE_DIRECTION_FROM_ANCHOR = "one_direction_from_anchor"
    EXPLICIT_LIST = "explicit_list"


class AnchorDirection(str, Enum):
    SOURCE_TO_OTHERS = "source_to_others"
    OTHERS_TO_SOURCE = "others_to_source"


@dataclass(frozen=True)
class ViewTransferPairSample:
    """A single source->target training sample descriptor."""

    task: str
    episode: str
    source_clip: str
    target_clip: str

    def __str__(self) -> str:
        return f"{self.task}_{self.episode}_{self.source_clip}_to_{self.target_clip}"


def build_clip_pairs(
    *,
    clip_names: tuple[str, ...],
    pair_mode: PairMode,
    anchor_clip: str | None = None,
    anchor_direction: AnchorDirection | None = AnchorDirection.SOURCE_TO_OTHERS,
    explicit_pairs: tuple[tuple[str, str], ...] | None = None,
) -> list[tuple[str, str]]:
    """Build clip pairs for one episode according to pairing strategy."""
    assert isinstance(pair_mode, PairMode), f"Invalid pair_mode: {pair_mode!r}"
    if len(clip_names) < 2:
        raise ValueError("clip_names must contain at least 2 clips.")

    clip_set = set(clip_names)
    if len(clip_set) != len(clip_names):
        raise ValueError(f"clip_names contains duplicates: {clip_names}")

    if pair_mode is PairMode.ALL_ORDERED_NONSELF:
        return [(src, tgt) for src in clip_names for tgt in clip_names if src != tgt]

    if pair_mode is PairMode.ONE_DIRECTION_FROM_ANCHOR:
        if anchor_clip is None:
            raise ValueError("anchor_clip is required when pair_mode='one_direction_from_anchor'.")
        if anchor_clip not in clip_set:
            raise ValueError(f"anchor_clip={anchor_clip!r} must be present in clip_names={clip_names}.")
        assert isinstance(anchor_direction, AnchorDirection), f"Invalid anchor_direction: {anchor_direction!r}"

        others = [clip for clip in clip_names if clip != anchor_clip]
        if anchor_direction is AnchorDirection.SOURCE_TO_OTHERS:
            return [(anchor_clip, other) for other in others]
        return [(other, anchor_clip) for other in others]

    # PairMode.EXPLICIT_LIST
    if not explicit_pairs:
        raise ValueError("explicit_pairs must be provided when pair_mode='explicit_list'.")

    out: list[tuple[str, str]] = []
    for src, tgt in explicit_pairs:
        if src not in clip_set or tgt not in clip_set:
            raise ValueError(
                f"Invalid explicit pair ({src!r}, {tgt!r}): both clips must be in clip_names={clip_names}."
            )
        out.append((src, tgt))
    return out


def scan_task_episodes(dataset_dir: str | Path) -> list[tuple[str, str]]:
    """Discover (task, episode) from dataset_dir/observations/<task>/<episode>/videos."""
    root = Path(dataset_dir)
    observations_dir = root / "observations"
    if not observations_dir.exists():
        raise FileNotFoundError(f"Expected observations root at {observations_dir}")

    episodes: list[tuple[str, str]] = []
    for task_dir in tqdm(
        sorted([p for p in observations_dir.iterdir() if p.is_dir()]), desc="Scanning dataset direcory..."
    ):
        task = task_dir.name
        for episode_dir in sorted([p for p in task_dir.iterdir() if p.is_dir()]):
            videos_dir = episode_dir / "videos"
            if videos_dir.exists():
                episodes.append((task, episode_dir.name))
    return episodes


def _episode_has_clips(
    *,
    dataset_dir: str | Path,
    task: str,
    episode: str,
    clip_names: Iterable[str],
) -> bool:
    for clip_name in clip_names:
        if not raw_video_path(dataset_dir, task, episode, clip_name).exists():
            return False
    return True


def build_samples(
    *,
    dataset_dir: str | Path,
    clip_names: tuple[str, ...],
    pair_mode: PairMode,
    anchor_clip: str | None = None,
    anchor_direction: AnchorDirection | None = AnchorDirection.SOURCE_TO_OTHERS,
    explicit_pairs: tuple[tuple[str, str], ...] | None = None,
) -> list[ViewTransferPairSample]:
    """Build per-episode source/target samples from dataset root scan."""
    pairs = build_clip_pairs(
        clip_names=clip_names,
        pair_mode=pair_mode,
        anchor_clip=anchor_clip,
        anchor_direction=anchor_direction,
        explicit_pairs=explicit_pairs,
    )

    out: list[ViewTransferPairSample] = []
    tasks_episodes = scan_task_episodes(dataset_dir)
    pbar = tqdm(tasks_episodes, desc="Building samples...", total=len(tasks_episodes) * len(pairs))
    for task, episode in pbar:
        if not _episode_has_clips(dataset_dir=dataset_dir, task=task, episode=episode, clip_names=clip_names):
            pbar.update(len(pairs))
            continue
        for src, tgt in pairs:
            out.append(
                ViewTransferPairSample(
                    task=task,
                    episode=episode,
                    source_clip=src,
                    target_clip=tgt,
                )
            )
            pbar.update(1)
    pbar.close()
    return out
