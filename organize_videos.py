import os
import shutil
from pathlib import Path

_BASE = Path(__file__).parent
source_dir = str(_BASE / "videos")   # drop unorganized videos here
dest_base  = str(_BASE / "alphabet") # organized per-letter destination

moved = 0
created = 0
added_to_existing = 0

# Walk through all subfolders
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        # Find mp4 files in this folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.mp4'):
                word_name = os.path.splitext(filename)[0]

                # Check if folder exists in destination
                dest_folder = os.path.join(dest_base, word_name)
                if os.path.exists(dest_folder):
                    added_to_existing += 1
                else:
                    os.makedirs(dest_folder)
                    created += 1

                # Move the file
                src_path = os.path.join(folder_path, filename)
                dst_path = os.path.join(dest_folder, filename)
                shutil.move(src_path, dst_path)
                moved += 1

print(f"Moved {moved} videos")
print(f"  - {added_to_existing} added to existing folders")
print(f"  - {created} new folders created")
