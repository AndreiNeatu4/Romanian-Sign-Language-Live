"""
Organize video files into folders by gesture name
This prepares videos for the augmented data extraction script
"""
import os
import shutil
from pathlib import Path
import re

def extract_gesture_name(filename):
    """Extract the base gesture name from a video filename"""
    # Remove .mp4 extension
    name = filename.replace('.mp4', '')
    name = name.replace('.avi', '')
    name = name.replace('.mov', '')
    name = name.replace('.mkv', '')

    # Handle numbered variants like "abandon-2" -> "abandon"
    # or "adapta-1" -> "adapta"
    name = re.sub(r'-\d+$', '', name)

    return name

def organize_videos(video_dir):
    """Organize videos into gesture-named folders"""
    video_dir = Path(video_dir)

    # Find all video files
    video_extensions = ['*.mp4', '*.avi', '*.mov', '*.mkv']
    video_files = []
    for ext in video_extensions:
        video_files.extend(video_dir.glob(ext))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"Found {len(video_files)} video files")
    print("\nProposed organization:")
    print("=" * 60)

    # Group videos by gesture name
    gesture_groups = {}
    for video_file in video_files:
        gesture_name = extract_gesture_name(video_file.name)
        if gesture_name not in gesture_groups:
            gesture_groups[gesture_name] = []
        gesture_groups[gesture_name].append(video_file)

    # Show what will be done
    for gesture_name, files in sorted(gesture_groups.items()):
        print(f"\n{gesture_name}/")
        for f in files:
            print(f"  - {f.name}")

    print("\n" + "=" * 60)
    print(f"This will create {len(gesture_groups)} folders")
    print(f"and organize {len(video_files)} videos")
    print("=" * 60)

    response = input(f"\nProceed with organization? (y/yes): ").strip().lower()

    if response not in ['yes', 'y']:
        print("Operation cancelled.")
        return

    # Create folders and move files
    moved_count = 0
    for gesture_name, files in gesture_groups.items():
        # Create gesture folder
        gesture_folder = video_dir / gesture_name
        gesture_folder.mkdir(exist_ok=True)

        # Move videos into folder
        for video_file in files:
            dest = gesture_folder / video_file.name
            if dest.exists():
                print(f"Warning: {dest} already exists, skipping...")
                continue

            shutil.move(str(video_file), str(dest))
            moved_count += 1
            print(f"Moved: {video_file.name} -> {gesture_name}/")

    print(f"\n[OK] Successfully organized {moved_count} videos into {len(gesture_groups)} folders!")
    print(f"\nFolder structure:")
    print(f"{video_dir}/")
    for gesture_name in sorted(gesture_groups.keys()):
        print(f"  {gesture_name}/")
        for f in gesture_groups[gesture_name]:
            print(f"    {f.name}")

    print(f"\n[OK] Ready for augmented data extraction!")
    print(f"\nNext step:")
    print(f"  python extract_augmented_data.py")

if __name__ == '__main__':
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))
        import config
        VIDEO_DIR = config.VIDEO_DIR
    except:
        VIDEO_DIR = r"C:\Users\ASUS\Desktop\video train"

    organize_videos(VIDEO_DIR)
