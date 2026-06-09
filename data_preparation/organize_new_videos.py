"""
Organize Videos into Gesture Folders

This script takes loose .mp4 files and organizes them into folders
where each video's name becomes the gesture/folder name.

Before:
  video train/
    A adormi.mp4
    A atipi.mp4
    abandon/
      video1.mp4

After:
  video train/
    A adormi/
      A adormi.mp4
    A atipi/
      A atipi.mp4
    abandon/
      video1.mp4
"""

import os
import sys
import shutil
from pathlib import Path

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def organize_videos(video_dir: str, dry_run: bool = False):
    """
    Organize loose video files into gesture folders.

    Args:
        video_dir: Path to video directory
        dry_run: If True, only print what would happen without moving files
    """
    video_dir = Path(video_dir)
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv'}

    # Find all loose video files (not in subfolders)
    loose_videos = []
    for item in video_dir.iterdir():
        if item.is_file() and item.suffix.lower() in video_extensions:
            loose_videos.append(item)

    print(f"Found {len(loose_videos)} loose video files to organize\n")

    if len(loose_videos) == 0:
        print("No loose videos found. All videos are already in folders.")
        return

    # Organize each video into its own folder
    organized_count = 0
    for video_file in loose_videos:
        # Use video name (without extension) as folder name
        gesture_name = video_file.stem

        # Clean up folder name (remove problematic characters)
        folder_name = gesture_name.strip()

        # Create folder path
        folder_path = video_dir / folder_name

        if dry_run:
            print(f"[DRY RUN] Would create: {folder_path}/")
            print(f"[DRY RUN] Would move: {video_file.name} -> {folder_name}/{video_file.name}")
        else:
            # Create folder if it doesn't exist
            folder_path.mkdir(exist_ok=True)

            # Move video into folder
            destination = folder_path / video_file.name
            if not destination.exists():
                shutil.move(str(video_file), str(destination))
                print(f"[OK] {video_file.name} -> {folder_name}/")
                organized_count += 1
            else:
                print(f"[SKIP] {destination} already exists")

    print(f"\nOrganized {organized_count} videos into folders")

    # Count total gesture folders
    folders = [f for f in video_dir.iterdir() if f.is_dir()]
    print(f"Total gesture classes: {len(folders)}")


def main():
    VIDEO_DIR = str(Path(__file__).parent.parent / "alphabet")

    print("=" * 60)
    print("ORGANIZE VIDEOS INTO GESTURE FOLDERS")
    print("=" * 60)

    # Organize directly
    print("\n--- ORGANIZING ---\n")
    organize_videos(VIDEO_DIR, dry_run=False)
    print("\n[DONE] Videos organized!")
    print("\nNext step: Run extract_augmented_data.py")


if __name__ == "__main__":
    main()
