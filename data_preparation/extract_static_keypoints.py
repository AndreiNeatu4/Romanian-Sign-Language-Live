"""
Static Alphabet Keypoint Extraction
====================================
Processes JPG images (with MP4 fallback for letters missing images) from the
alphabet folder. Uses MediaPipe static_image_mode=True for accurate single-frame
hand detection.

Output saved to: data/alphabet_static/
  X_train/val/test.npy       -- (N, 126)   flat hand features  → use for MLP
  X_seq_train/val/test.npy   -- (N, 30, 126) tiled sequences   → use for CNN-LSTM
  y_train/val/test.npy       -- integer labels
  class_labels.json          -- class names and index map
  extraction_report.json     -- per-letter detection stats
"""

import cv2
import mediapipe as mp
import numpy as np
import json
import random
import sys
from pathlib import Path
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ALPHABET_DIR = Path(__file__).parent.parent / "alphabet"
OUTPUT_DIR   = Path(__file__).parent.parent / "data" / "alphabet_static"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEQUENCE_LENGTH    = 30    # how many times to tile one frame (for CNN-LSTM)
MIN_DETECT_CONF    = 0.5
FRAMES_FROM_VIDEO  = 40    # frames to sample when a letter has no JPGs
TRAIN_SPLIT        = 0.70
VAL_SPLIT          = 0.15  # test gets the rest

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Feature extraction  (126-dim: 2 hands × 21 landmarks × 3 coords)
# ---------------------------------------------------------------------------
def extract_hand_features(image_rgb: np.ndarray, hands) -> tuple:
    """
    Returns (feature_vector, hand_detected).
    Each hand is wrist-centred and scale-normalised before flattening.
    """
    results = hands.process(image_rgb)
    features = np.zeros(126, dtype=np.float32)

    if not results.multi_hand_landmarks:
        return features, False

    for i, hand_lm in enumerate(results.multi_hand_landmarks[:2]):
        lms = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_lm.landmark], dtype=np.float32
        )
        wrist = lms[0].copy()
        lms  -= wrist
        scale = np.max(np.abs(lms)) + 1e-7
        lms  /= scale
        features[i * 63 : (i + 1) * 63] = lms.flatten()

    return features, True


# ---------------------------------------------------------------------------
# Video frame sampler (fallback for letters with no JPGs)
# ---------------------------------------------------------------------------
def sample_frames_from_video(video_path: Path, n: int = FRAMES_FROM_VIDEO) -> list:
    """Return up to n evenly-spaced BGR frames from a video file."""
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, min(n, total), dtype=int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # MediaPipe with static_image_mode=True — best accuracy for single frames
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=MIN_DETECT_CONF,
    )

    all_features: list = []
    all_labels:   list = []
    report:       dict = {}

    letter_dirs = sorted([d for d in ALPHABET_DIR.iterdir() if d.is_dir()])
    class_names = [d.name for d in letter_dirs]
    label_map   = {name: idx for idx, name in enumerate(class_names)}

    print(f"Classes ({len(class_names)}): {class_names}\n")

    for letter_dir in letter_dirs:
        letter    = letter_dir.name
        label_idx = label_map[letter]

        # Collect image files
        img_paths = (
            list(letter_dir.glob("*.jpg"))
            + list(letter_dir.glob("*.jpeg"))
            + list(letter_dir.glob("*.png"))
        )

        if img_paths:
            source = "images"
            items  = img_paths          # list of Path objects
        else:
            mp4s = list(letter_dir.glob("*.mp4"))
            if not mp4s:
                print(f"  {letter}: no images or video — SKIPPED")
                report[letter] = {"source": "none", "total": 0, "detected": 0}
                continue
            print(f"  {letter}: no JPGs — sampling {FRAMES_FROM_VIDEO} frames "
                  f"from {mp4s[0].name}")
            source = "video"
            items  = sample_frames_from_video(mp4s[0])  # list of BGR arrays

        total    = len(items)
        detected = 0

        for item in tqdm(items, desc=f"  {letter:3s}", leave=False, ncols=72):
            if isinstance(item, Path):
                bgr = cv2.imread(str(item))
            else:
                bgr = item  # numpy array from video

            if bgr is None:
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            feats, found = extract_hand_features(rgb, hands)

            if found:
                all_features.append(feats)
                all_labels.append(label_idx)
                detected += 1

        report[letter] = {"source": source, "total": total, "detected": detected}
        print(f"  {letter:3s}  {detected:3d}/{total:3d} detected  ({source})")

    hands.close()

    if not all_features:
        print("\nNo features extracted. Make sure images contain visible hands.")
        return

    X = np.array(all_features, dtype=np.float32)   # (N, 126)
    y = np.array(all_labels,   dtype=np.int64)      # (N,)

    print(f"\nTotal samples: {len(X)}")

    # ── Stratified split ────────────────────────────────────────────────────
    idx_train, idx_val, idx_test = [], [], []
    for cls_i in range(len(class_names)):
        pool = np.where(y == cls_i)[0].tolist()
        random.shuffle(pool)
        n       = len(pool)
        n_train = max(1, int(round(n * TRAIN_SPLIT)))
        n_val   = max(0, int(round(n * VAL_SPLIT)))
        idx_train.extend(pool[:n_train])
        idx_val.extend(pool[n_train : n_train + n_val])
        idx_test.extend(pool[n_train + n_val :])

    for lst in (idx_train, idx_val, idx_test):
        random.shuffle(lst)

    # ── Tiled sequences for CNN-LSTM ─────────────────────────────────────────
    # Shape: (N, 30, 126) — each frame repeated 30 times
    X_seq = np.tile(X[:, np.newaxis, :], (1, SEQUENCE_LENGTH, 1))

    # ── Save ─────────────────────────────────────────────────────────────────
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    for name, idx in splits.items():
        if not idx:
            print(f"  WARNING: {name} split is empty")
            continue
        np.save(OUTPUT_DIR / f"X_{name}.npy",     X[idx])
        np.save(OUTPUT_DIR / f"y_{name}.npy",     y[idx])
        np.save(OUTPUT_DIR / f"X_seq_{name}.npy", X_seq[idx])

    with open(OUTPUT_DIR / "class_labels.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "classes":         class_names,
                "label_to_idx":    label_map,
                "num_classes":     len(class_names),
                "feature_size":    126,
                "sequence_length": SEQUENCE_LENGTH,
            },
            f, ensure_ascii=False, indent=2,
        )

    with open(OUTPUT_DIR / "extraction_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'─'*52}")
    print(f"  Train : {len(idx_train)} samples")
    print(f"  Val   : {len(idx_val)} samples")
    print(f"  Test  : {len(idx_test)} samples")
    print(f"\nPer-class sample counts:")
    for i, name in enumerate(class_names):
        count = int(np.sum(y == i))
        bar   = "█" * (count // 3)
        print(f"  {name:3s}  {count:4d}  {bar}")

    print(f"\nSaved to: {OUTPUT_DIR}")
    print("  X_*.npy       — (N, 126)    flat features  → MLP classifier")
    print("  X_seq_*.npy   — (N, 30, 126) tiled frames  → CNN-LSTM")
    print("  class_labels.json")
    print("  extraction_report.json")
    print("\nNext: run train_static_model.py")


if __name__ == "__main__":
    main()
