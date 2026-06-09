"""
Detect hands in an image (or folder of images) using MediaPipe and save cropped regions.
Crops are saved in the same folder as the original, named: originalname_001.jpg, originalname_002.jpg, ...

Usage:
    python crop_hand.py photo.jpg                   # saves photo_001.jpg next to original
    python crop_hand.py ./photos/                   # processes every image in the folder
    python crop_hand.py ./alphabet/                 # goes into A/, B/, C/... and crops in each
    python crop_hand.py ./photos/ --padding 50      # extra pixels around the hand (default: 30)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import cv2
import mediapipe as mp


def imread_unicode(path: str):
    """Read an image from a path that may contain Unicode characters."""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def imwrite_unicode(path: str, image):
    """Write an image to a path that may contain Unicode characters."""
    ext = Path(path).suffix
    success, buf = cv2.imencode(ext, image)
    if success:
        buf.tofile(path)
    return success


def detect_and_crop_hands(image_path: str, padding: int = 30):
    """Detect hands in an image and return cropped regions."""
    image = imread_unicode(image_path)
    if image is None:
        print(f"Error: Could not read image '{image_path}'")
        return []

    h, w, _ = image.shape
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5,
    ) as hands:
        results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        print(f"  No hands detected in {Path(image_path).name}")
        return []

    crops = []
    for hand_landmarks in results.multi_hand_landmarks:
        x_coords = [lm.x * w for lm in hand_landmarks.landmark]
        y_coords = [lm.y * h for lm in hand_landmarks.landmark]

        x_min = max(0, int(min(x_coords)) - padding)
        y_min = max(0, int(min(y_coords)) - padding)
        x_max = min(w, int(max(x_coords)) + padding)
        y_max = min(h, int(max(y_coords)) + padding)

        crops.append(image[y_min:y_max, x_min:x_max])

    return crops


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def process_single_image(image_path: Path, padding: int):
    """Process one image: detect hands and save crops in the same folder."""
    crops = detect_and_crop_hands(str(image_path), padding)
    if not crops:
        return 0

    stem = image_path.stem
    ext = image_path.suffix or ".jpg"
    folder = image_path.parent

    for i, crop in enumerate(crops):
        out_path = str(folder / f"{stem}_{i + 1:03d}{ext}")
        imwrite_unicode(out_path, crop)
        print(f"  Saved -> {out_path}")

    return len(crops)


def main():
    parser = argparse.ArgumentParser(description="Crop hands from an image or folder using MediaPipe.")
    parser.add_argument("input", help="Path to an image or a folder of images")
    parser.add_argument("--padding", type=int, default=30, help="Padding around hand (default: 30)")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        # Check for subfolders (e.g. alphabet/A/, alphabet/B/, ...)
        subfolders = sorted(f for f in input_path.iterdir() if f.is_dir())

        if subfolders:
            # Recursive mode: go through each subfolder
            total_crops = 0
            total_images = 0
            for subfolder in subfolders:
                images = sorted(f for f in subfolder.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
                if not images:
                    continue
                print(f"\n[{subfolder.name}] — {len(images)} image(s)")
                for img in images:
                    print(f"  Processing {img.name}...")
                    total_crops += process_single_image(img, args.padding)
                    total_images += 1

            if total_images == 0:
                print(f"No images found in subfolders of '{input_path}'")
                sys.exit(1)

            print(f"\nDone — {total_crops} hand(s) cropped from {total_images} image(s) across {len(subfolders)} folder(s)")
        else:
            # Flat mode: process images directly in this folder
            images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            if not images:
                print(f"No images found in '{input_path}'")
                sys.exit(1)

            total_crops = 0
            for img in images:
                print(f"Processing {img.name}...")
                total_crops += process_single_image(img, args.padding)

            print(f"\nDone — {total_crops} hand(s) cropped from {len(images)} image(s)")

    else:
        if not input_path.exists():
            print(f"Error: '{input_path}' not found")
            sys.exit(1)

        print(f"Processing {input_path.name}...")
        if not process_single_image(input_path, args.padding):
            sys.exit(1)


if __name__ == "__main__":
    main()
