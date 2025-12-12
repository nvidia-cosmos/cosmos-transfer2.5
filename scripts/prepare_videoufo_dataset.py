#!/usr/bin/env python3
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

"""
Script to download and prepare VideoUFO dataset for Cosmos Transfer2 singleview post-training.

VideoUFO Dataset: https://huggingface.co/datasets/WenhaoWang/VideoUFO
Paper: https://openreview.net/forum?id=wwlwRuKle7 (NeurIPS 2025)
License: CC BY 4.0

This script:
1. Downloads VideoUFO metadata CSV from HuggingFace
2. Downloads VideoUFO tar files from HuggingFace
3. Extracts videos from tar files
4. Organizes into the required structure for transfer2 training:
   datasets/your_dataset/
   ├── videos/
   │   └── *.mp4
   └── captions/
       └── *.json
5. Filters to keep only the specified number of videos
6. Creates corresponding JSON caption files

Required dataset structure for transfer2 post-training:
- Each video in videos/ must have a corresponding JSON file in captions/
- Caption JSON format: {"caption": "Your description here"}
- File names must match (e.g., video1.mp4 → video1.json)

"""

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare VideoUFO dataset for Cosmos Transfer2 singleview post-training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download first tar file and keep 8 videos (default)
  python prepare_videoufo_dataset.py --storage_dir datasets/videoufo --num_videos 8

  # Download first 3 tar files and keep 100 videos
  python prepare_videoufo_dataset.py --storage_dir datasets/videoufo --num_videos 100 --num_tars 3

  # Download all 200 tar files (WARNING: ~900GB!)
  python prepare_videoufo_dataset.py --storage_dir datasets/videoufo --num_videos 10000 --num_tars 200 --download_all
        """,
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        required=True,
        help="Directory where the dataset will be stored (e.g., datasets/videoufo)",
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=8,
        help="Number of videos to keep in the final dataset (default: 8)",
    )
    parser.add_argument(
        "--num_tars",
        type=int,
        default=1,
        help="Number of tar files to download (1-200). Default: 1 (break after first iteration)",
    )
    parser.add_argument(
        "--download_all",
        action="store_true",
        help="Download all 200 tar files (~800GB). Requires explicit confirmation.",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading tar files (assumes they're already in storage_dir/tars)",
    )
    parser.add_argument(
        "--skip_metadata",
        action="store_true",
        help="Skip downloading metadata (assumes it's already in storage_dir/VideoUFO.csv)",
    )
    parser.add_argument(
        "--use_brief_caption",
        action="store_true",
        help="Use brief captions instead of detailed captions",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output for troubleshooting",
    )
    parser.add_argument(
        "--no_concat_brief",
        action="store_true",
        help="Don't concatenate brief caption with detailed caption (by default, brief + detailed)",
    )
    parser.add_argument(
        "--min_frames",
        type=int,
        default=93,
        help="Minimum number of frames required for a video (default: 93, matching training requirements)",
    )
    return parser.parse_args()


def confirm_download_all():
    """Show warning and get user confirmation for downloading the entire dataset."""
    print("\n" + "=" * 80)
    print("⚠️  WARNING: LARGE DOWNLOAD AHEAD ⚠️")
    print("=" * 80)
    print("\nYou are about to download the ENTIRE VideoUFO dataset:")
    print("  • Total size: ~800 GB")
    print("  • Number of tar files: 200")
    print("  • Total videos: 1,091,712")
    print("\nThis will:")
    print("  1. Take a VERY long time to download")
    print("  2. Require significant storage space")
    print("  3. Use substantial bandwidth")
    print("\nMake sure you have:")
    print("  ✓ At least 1 TB of free storage space")
    print("  ✓ Stable internet connection")
    print("  ✓ Sufficient time for the download to complete")
    print("\n" + "=" * 80)

    response = input("\nAre you sure you want to proceed? (yes/no): ").strip().lower()
    return response in ["yes", "y"]


def download_file(url: str, dest_path: Path, description: str = "Downloading"):
    """Download a file with progress indication."""
    print(f"{description}...")
    try:
        with urllib.request.urlopen(url) as response:
            total_size = int(response.headers.get("content-length", 0))

            with open(dest_path, "wb") as f:
                downloaded = 0
                chunk_size = 8192

                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    # Simple progress indicator
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_downloaded = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(
                            f"\r  Progress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True
                        )

                print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n✗ Failed: {e}")
        return False


def download_metadata(storage_dir: Path, skip_metadata: bool = False):
    """Download VideoUFO metadata CSV directly from HuggingFace."""
    metadata_path = storage_dir / "VideoUFO.csv"

    if metadata_path.exists() and skip_metadata:
        print(f"✓ Metadata already exists at {metadata_path}, skipping download")
        return metadata_path

    print("\n" + "=" * 80)
    print("Downloading VideoUFO Metadata")
    print("=" * 80)

    url = "https://huggingface.co/datasets/WenhaoWang/VideoUFO/resolve/main/VideoUFO.csv"

    if download_file(url, metadata_path, "Downloading VideoUFO.csv"):
        # Verify it's a valid CSV and count lines
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                # Read first line to check if it's a CSV
                first_line = f.readline()
                if not first_line.startswith("ID,") and "ID" not in first_line:
                    print(f"✗ Downloaded file doesn't look like a CSV. First line: {first_line[:100]}")
                    sys.exit(1)

                # Count total lines
                f.seek(0)
                line_count = sum(1 for _ in f) - 1  # Subtract header

            print(f"✓ Downloaded metadata: {line_count} entries")
            print(f"✓ Saved to: {metadata_path}")
            return metadata_path
        except Exception as e:
            print(f"✗ Error reading downloaded CSV: {e}")
            sys.exit(1)
    else:
        print("✗ Failed to download metadata")
        sys.exit(1)


def download_tar_files(storage_dir: Path, num_tars: int, skip_download: bool = False):
    """Download VideoUFO tar files from HuggingFace."""
    tar_dir = storage_dir / "tars"
    tar_dir.mkdir(parents=True, exist_ok=True)

    if skip_download:
        print(f"✓ Skipping tar download, using existing files in {tar_dir}")
        return tar_dir

    print("\n" + "=" * 80)
    print(f"Downloading VideoUFO Tar Files (1-{num_tars})")
    print("=" * 80)

    base_url = "https://huggingface.co/datasets/WenhaoWang/VideoUFO/resolve/main/VideoUFO_tar"

    for i in range(1, num_tars + 1):
        tar_filename = f"VideoUFO_{i}.tar"
        tar_path = tar_dir / tar_filename

        if tar_path.exists():
            print(f"✓ [{i}/{num_tars}] {tar_filename} already exists, skipping")
            continue

        print(f"\n[{i}/{num_tars}] Downloading {tar_filename}...")
        url = f"{base_url}/{tar_filename}"

        if not download_file(url, tar_path, f"  Downloading {tar_filename}"):
            if i == 1:
                print("Failed to download the first tar file. Exiting.")
                sys.exit(1)
            else:
                print(f"Continuing with {i - 1} tar files...")
                break

        print(f"✓ Downloaded: {tar_filename}")

    return tar_dir


def extract_videos(tar_dir: Path, temp_extract_dir: Path, num_videos: int, min_frames: int = 93):
    """Extract videos from tar files until we have enough valid ones (with >= min_frames)."""
    print("\n" + "=" * 80)
    print(f"Extracting Videos (target: {num_videos} valid videos with >= {min_frames} frames)")
    print("=" * 80)

    temp_extract_dir.mkdir(parents=True, exist_ok=True)

    tar_files = sorted(tar_dir.glob("VideoUFO_*.tar"))
    if not tar_files:
        print(f"✗ No tar files found in {tar_dir}")
        sys.exit(1)

    extracted_videos = []
    valid_count = 0
    total_extracted = 0
    skipped_too_short = 0

    # Check if ffprobe is available
    ffprobe_available = True
    try:
        subprocess.run(["ffprobe", "-version"], capture_output=True, check=True, timeout=5)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("  ⚠ ffprobe not found - extracting without frame count filtering")
        print("    Videos may be filtered later, which could result in fewer than requested videos")
        ffprobe_available = False

    for tar_path in tar_files:
        if valid_count >= num_videos:
            break

        print(f"\nExtracting from {tar_path.name}...")

        try:
            with tarfile.open(tar_path, "r") as tar:
                members = tar.getmembers()
                video_members = [m for m in members if m.name.endswith(".mp4") and m.isfile()]

                print(f"  Found {len(video_members)} videos in {tar_path.name}")

                if len(video_members) == 0:
                    print(f"  ⚠ No video files found in {tar_path.name}")
                    continue

                # Extract and validate videos
                for idx, member in enumerate(video_members):
                    if valid_count >= num_videos:
                        break

                    # Extract video
                    tar.extract(member, path=temp_extract_dir)
                    extracted_path = temp_extract_dir / member.name
                    total_extracted += 1

                    # Verify the file exists
                    if not (extracted_path.exists() and extracted_path.is_file()):
                        print(f"  ⚠ Failed to extract {member.name}")
                        continue

                    # Check frame count if ffprobe is available
                    if ffprobe_available:
                        frame_count = get_video_frame_count(extracted_path)
                        if frame_count is not None and frame_count < min_frames:
                            skipped_too_short += 1
                            # Delete the video since it doesn't meet requirements
                            extracted_path.unlink()
                            continue

                    # Video is valid
                    extracted_videos.append(extracted_path)
                    valid_count += 1

                    # Simple progress
                    if total_extracted % 50 == 0 or valid_count >= num_videos:
                        print(
                            f"  Progress: {valid_count}/{num_videos} valid videos (extracted {total_extracted}, skipped {skipped_too_short})",
                            end="\r",
                            flush=True,
                        )

                print()  # New line after progress

        except Exception as e:
            print(f"✗ Failed to extract {tar_path.name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print(
        f"\n✓ Extracted {valid_count} valid videos (total extracted: {total_extracted}, skipped: {skipped_too_short})"
    )
    return extracted_videos


def get_video_frame_count(video_path: Path):
    """Get the number of frames in a video using ffprobe."""
    try:
        # Try ffprobe first (more accurate)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())

        # Fallback: use nb_frames if nb_read_frames is not available
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        pass

    return None


def load_metadata_csv(
    metadata_path: Path, use_brief_caption: bool = False, debug: bool = False, concat_brief: bool = True
):
    """Load metadata CSV and return ID to caption mapping."""
    id_to_caption = {}

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            # Check available columns
            if reader.fieldnames:
                print(f"  CSV columns found: {len(reader.fieldnames)} total")
                if debug:
                    print(f"  First 10 columns: {reader.fieldnames[:10]}")

                # Check if our required columns exist
                if "ID" not in reader.fieldnames:
                    print(f"  ✗ 'ID' column not found in CSV!")
                    print(f"  Available columns: {reader.fieldnames}")
                    return {}

                # Check for both Brief and Detailed Caption columns
                if "Detailed Caption" not in reader.fieldnames:
                    print(f"  ✗ 'Detailed Caption' column not found in CSV!")
                    print(f"  Available columns: {reader.fieldnames}")
                    return {}

                if concat_brief and "Brief Caption" not in reader.fieldnames:
                    print(f"  ⚠ 'Brief Caption' column not found, will not concatenate")
                    concat_brief = False

            for idx, row in enumerate(reader):
                video_id = row.get("ID")
                detailed_caption = row.get("Detailed Caption", "")
                brief_caption = row.get("Brief Caption", "")

                if video_id and detailed_caption and detailed_caption.strip():
                    # Concatenate brief caption with detailed caption
                    if concat_brief and brief_caption and brief_caption.strip():
                        # Brief caption first, then detailed caption
                        combined_caption = f"{brief_caption.strip()} {detailed_caption.strip()}"
                    elif use_brief_caption and brief_caption and brief_caption.strip():
                        # If user wants brief only
                        combined_caption = brief_caption.strip()
                    else:
                        # Detailed only
                        combined_caption = detailed_caption.strip()

                    id_to_caption[video_id] = combined_caption

                    # Debug first few rows
                    if idx < 3:
                        print(f"  Row {idx}: ID='{video_id}'")
                        print(f"    Brief: '{brief_caption[:60]}...'" if brief_caption else "    Brief: [missing]")
                        print(
                            f"    Detailed: '{detailed_caption[:60]}...'"
                            if detailed_caption
                            else "    Detailed: [missing]"
                        )
                        print(f"    Combined: '{combined_caption[:100]}...'")

                # Stop early if debug and we have some data
                if debug and idx > 100 and len(id_to_caption) > 10:
                    # Continue loading the rest without debug output
                    for row in reader:
                        video_id = row.get("ID")
                        detailed_caption = row.get("Detailed Caption", "")
                        brief_caption = row.get("Brief Caption", "")
                        if video_id and detailed_caption and detailed_caption.strip():
                            if concat_brief and brief_caption and brief_caption.strip():
                                combined_caption = f"{brief_caption.strip()} {detailed_caption.strip()}"
                            elif use_brief_caption and brief_caption and brief_caption.strip():
                                combined_caption = brief_caption.strip()
                            else:
                                combined_caption = detailed_caption.strip()
                            id_to_caption[video_id] = combined_caption
                    break

        return id_to_caption

    except Exception as e:
        print(f"  ✗ Error parsing CSV: {e}")
        import traceback

        traceback.print_exc()
        return {}


def create_dataset_structure(
    extracted_videos: list,
    metadata_path: Path,
    storage_dir: Path,
    use_brief_caption: bool = False,
    debug: bool = False,
    concat_brief: bool = True,
    min_frames: int = 93,
):
    """Create the required dataset structure for transfer2 training."""
    print("\n" + "=" * 80)
    print("Creating Dataset Structure")
    print("=" * 80)

    # Create output directories
    videos_dir = storage_dir / "videos"
    captions_dir = storage_dir / "captions"
    videos_dir.mkdir(parents=True, exist_ok=True)
    captions_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    print("Loading metadata...")
    id_to_caption = load_metadata_csv(
        metadata_path, use_brief_caption=use_brief_caption, debug=debug, concat_brief=concat_brief
    )

    if concat_brief:
        print(f"Concatenating Brief Caption + Detailed Caption")
    elif use_brief_caption:
        print(f"Using Brief Caption only")
    else:
        print(f"Using Detailed Caption only")
    print(f"Loaded {len(id_to_caption)} caption entries from metadata")

    if len(extracted_videos) == 0:
        print("✗ No videos to process!")
        return 0

    # Process each video
    processed_count = 0
    skipped_count = 0

    print(f"\nProcessing {len(extracted_videos)} extracted videos...")
    print(f"Note: Videos have already been filtered for >= {min_frames} frames during extraction")

    for idx, video_path in enumerate(extracted_videos):
        # Extract video ID from filename (e.g., "---MaV1RQGE.0.mp4" -> "---MaV1RQGE.0")
        # Handle both direct files and files in subdirectories
        video_filename = video_path.name
        video_id = Path(video_filename).stem

        # Debug: print first few to help diagnose
        if idx < 3:
            print(f"  Processing: {video_path} -> ID: {video_id}")

        # Get caption from metadata
        caption = id_to_caption.get(video_id)

        if not caption:
            if idx < 3:
                print(f"  ⚠ No caption found for ID: {video_id}")
                # Show what IDs are available (first few)
                sample_ids = list(id_to_caption.keys())[:5]
                print(f"  Sample metadata IDs: {sample_ids}")
            skipped_count += 1
            continue

        # Copy video to videos directory
        video_dest = videos_dir / f"{video_id}.mp4"
        try:
            shutil.copy2(video_path, video_dest)
        except Exception as e:
            print(f"  ✗ Failed to copy {video_id}: {e}")
            skipped_count += 1
            continue

        # Create caption JSON
        caption_dest = captions_dir / f"{video_id}.json"
        caption_data = {"caption": caption}
        try:
            with open(caption_dest, "w", encoding="utf-8") as f:
                json.dump(caption_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  ✗ Failed to write caption for {video_id}: {e}")
            continue

        processed_count += 1

        # Simple progress
        if (idx + 1) % 10 == 0 or (idx + 1) == len(extracted_videos):
            print(f"  Processed {processed_count}/{len(extracted_videos)} videos", end="\r", flush=True)

    print()  # New line after progress
    print(f"\n✓ Processed {processed_count} videos")
    if skipped_count > 0:
        print(f"  ⚠ Skipped {skipped_count} videos (missing captions or errors)")

    print(f"\n✓ Dataset structure created:")
    print(f"  Videos: {videos_dir} ({len(list(videos_dir.glob('*.mp4')))} files)")
    print(f"  Captions: {captions_dir} ({len(list(captions_dir.glob('*.json')))} files)")

    return processed_count


def cleanup_temp_files(temp_extract_dir: Path):
    """Clean up temporary extraction directory."""
    print("\n" + "=" * 80)
    print("Cleanup")
    print("=" * 80)

    if temp_extract_dir.exists():
        print(f"Removing temporary extraction directory: {temp_extract_dir}")
        shutil.rmtree(temp_extract_dir)
        print("✓ Cleanup complete")

    print("\nNote: Keeping tar files and cache for future use")
    print("To save space, you can manually delete:")
    print(f"  - {temp_extract_dir.parent / 'tars'}")


def verify_dataset(storage_dir: Path):
    """Verify the dataset structure and contents."""
    print("\n" + "=" * 80)
    print("Dataset Verification")
    print("=" * 80)

    videos_dir = storage_dir / "videos"
    captions_dir = storage_dir / "captions"

    # Count files
    video_files = list(videos_dir.glob("*.mp4"))
    caption_files = list(captions_dir.glob("*.json"))

    print(f"Videos: {len(video_files)}")
    print(f"Captions: {len(caption_files)}")

    # Check matching
    video_ids = {v.stem for v in video_files}
    caption_ids = {c.stem for c in caption_files}

    if video_ids == caption_ids:
        print("✓ All videos have matching captions")
    else:
        missing_captions = video_ids - caption_ids
        missing_videos = caption_ids - video_ids

        if missing_captions:
            print(f"✗ {len(missing_captions)} videos missing captions")
        if missing_videos:
            print(f"✗ {len(missing_videos)} captions missing videos")

    # Sample a few captions
    print("\nSample captions:")
    for caption_file in list(caption_files)[:3]:
        with open(caption_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            caption_text = data["caption"]
            # Truncate long captions
            if len(caption_text) > 100:
                caption_text = caption_text[:100] + "..."
            print(f"  {caption_file.stem}: {caption_text}")

    print("\n" + "=" * 80)
    print("✓ Dataset Preparation Complete!")
    print("=" * 80)
    print(f"\nDataset location: {storage_dir}")
    print("\nYou can now use this dataset for training with:")
    print(f"  dataloader_train.dataset.dataset_dir={storage_dir}")


def main():
    args = parse_args()

    # Validate arguments
    if args.num_tars < 1 or args.num_tars > 200:
        print("✗ num_tars must be between 1 and 200")
        sys.exit(1)

    if args.download_all:
        if not confirm_download_all():
            print("Download cancelled by user.")
            sys.exit(0)
        args.num_tars = 200

    # Create storage directory
    storage_dir = Path(args.storage_dir).resolve()
    storage_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("VideoUFO Dataset Preparation for Cosmos Transfer2")
    print("=" * 80)
    print(f"Storage directory: {storage_dir}")
    print(f"Target videos: {args.num_videos}")
    print(f"Tar files to download: {args.num_tars}")
    print(
        f"Caption format: {'Brief only' if args.use_brief_caption else 'Brief + Detailed' if not args.no_concat_brief else 'Detailed only'}"
    )
    print(f"Minimum frames: {args.min_frames} (videos with fewer frames will be skipped)")
    print("=" * 80)

    # Step 1: Download metadata
    metadata_path = download_metadata(storage_dir, args.skip_metadata)

    # Step 2: Download tar files
    tar_dir = download_tar_files(storage_dir, args.num_tars, args.skip_download)

    # Step 3: Extract videos (with frame count filtering)
    temp_extract_dir = storage_dir / "temp_extract"
    extracted_videos = extract_videos(tar_dir, temp_extract_dir, args.num_videos, args.min_frames)

    if len(extracted_videos) < args.num_videos:
        print(
            f"\n⚠ Warning: Only extracted {len(extracted_videos)} valid videos, less than requested {args.num_videos}"
        )
        print(f"   Reason: Not enough videos with >= {args.min_frames} frames in the downloaded tar files")
        print(f"   Solution: Download more tar files using --num_tars (e.g., --num_tars 2)")

    # Step 4: Create dataset structure
    processed_count = create_dataset_structure(
        extracted_videos,
        metadata_path,
        storage_dir,
        args.use_brief_caption,
        debug=args.debug,
        concat_brief=not args.no_concat_brief,
        min_frames=args.min_frames,
    )

    # Step 5: Cleanup
    cleanup_temp_files(temp_extract_dir)

    # Step 6: Verify
    verify_dataset(storage_dir)

    print("\nNext steps:")
    print("1. Verify video quality and captions")
    print("2. Run training with:")
    print(f"\n   torchrun --nproc_per_node=8 -m scripts.train \\")
    print(f"       --config=cosmos_transfer2/singleview_config.py \\")
    print(f"       -- experiment=transfer2_singleview_posttrain_edge_example \\")
    print(f"       dataloader_train.dataset.dataset_dir={storage_dir} \\")
    print(f"       'dataloader_train.sampler.dataset=${{dataloader_train.dataset}}' \\")
    print(f"       job.wandb_mode=disabled")


if __name__ == "__main__":
    main()
