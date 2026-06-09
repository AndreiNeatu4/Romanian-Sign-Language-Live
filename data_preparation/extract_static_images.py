"""
Static-image extractor (JPG/PNG -> tiled 216-dim sequences)
============================================================

Walks alphabet/<LETTER>/ for image files and produces .npy windows in the
SAME folder layout as extract_augmented_fast.py:

    data/alphabet_augmented/<LETTER>/img_<imgname>/aug_<idx>.npy

so prepare_augmented_dataset.py picks them up alongside the video sequences
without any modification.

Each JPG becomes its own "video group". For each image we generate N visual
augmentations, run MediaPipe in static_image_mode (more accurate per-frame
detection than tracking mode), and tile the resulting 216-dim feature vector
into a SEQUENCE_LENGTH-frame window. The wrist velocity / acceleration
channels are zero for static frames - which is the truth: these letters
don't move.

Directional letters (J / Z / Q / S / Ș / Ț) are skipped entirely. They are
defined by motion; tiling a snapshot of one would teach the model "no motion
also means J", which is wrong.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from datetime import datetime
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Set

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.feature_extractor import (
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    TOTAL_FEATURES_SIZE,
)


IMAGE_EXTS = ('*.jpg', '*.jpeg', '*.png')


def _build_pipelines(n: int) -> List[A.Compose]:
    """Same visual-only augmentations as the video extractor."""
    pipelines = [
        A.Compose([]),  # original
        A.Compose([A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0)]),
        A.Compose([A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=15, val_shift_limit=15, p=1.0)]),
        A.Compose([A.CLAHE(p=1.0)]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=10, p=1.0),
        ]),
        A.Compose([A.GaussNoise(std_range=(0.01, 0.06), p=1.0)]),
        A.Compose([A.MotionBlur(blur_limit=5, p=1.0)]),
        A.Compose([A.RandomGamma(gamma_limit=(85, 115), p=1.0)]),
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.GaussNoise(std_range=(0.01, 0.04), p=0.5),
        ]),
        A.Compose([
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=20, val_shift_limit=10, p=1.0),
            A.MotionBlur(blur_limit=3, p=0.5),
        ]),
    ]
    return pipelines[:n]


def _process_one_image(args) -> Dict:
    """Worker: one (image_path, label, output_dir, config) task."""
    image_path, gesture_label, output_dir, config = args

    sequence_length = config['sequence_length']
    spatial_augs = config['spatial_augmentations']
    face_anchors = config.get('face_mesh_anchors', DEFAULT_FACE_ANCHORS)
    use_face = config.get('extract_face_mesh', True)
    use_pose = config.get('extract_pose', True)
    bridge_frames = config.get('hand_dropout_bridge_frames', 3)

    image_path = Path(image_path)
    # cv2.imread cannot open paths that contain non-ASCII characters on
    # Windows (e.g. alphabet/Â/, alphabet/Ă/). Read the file via numpy and
    # decode in memory so any path is supported.
    try:
        raw = np.fromfile(str(image_path), dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size > 0 else None
    except OSError as e:
        return {'image': str(image_path), 'sequences': 0, 'error': f'fromfile failed: {e}'}
    if bgr is None:
        return {'image': str(image_path), 'sequences': 0, 'error': 'imdecode returned None'}

    pipelines = _build_pipelines(spatial_augs)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=config.get('min_detection_confidence', 0.5),
    )
    mp_face = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=config.get('min_detection_confidence', 0.5),
    ) if use_face else None
    mp_pose = mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=1,
        min_detection_confidence=config.get('min_detection_confidence', 0.5),
    ) if use_pose else None

    img_dir = Path(output_dir) / gesture_label / f"img_{image_path.stem}"
    img_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped_no_hand = 0
    for idx, pipeline in enumerate(pipelines):
        aug_bgr = bgr if idx == 0 else pipeline(image=bgr)['image']
        rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB)

        hand_res = mp_hands.process(rgb)
        face_res = mp_face.process(rgb) if mp_face is not None else None
        pose_res = mp_pose.process(rgb) if mp_pose is not None else None

        if not (hand_res and hand_res.multi_hand_landmarks):
            skipped_no_hand += 1
            continue

        ffe = FrameFeatureExtractor(
            face_anchors=face_anchors,
            bridge_frames=bridge_frames,
            use_face_mesh=use_face,
        )
        face_lm = face_res.multi_face_landmarks[0] if (face_res and face_res.multi_face_landmarks) else None
        pose_lm = pose_res.pose_landmarks if (pose_res and pose_res.pose_landmarks) else None
        feat = ffe.process_frame(
            hand_landmarks_list=hand_res.multi_hand_landmarks,
            handedness_list=hand_res.multi_handedness,
            face_landmarks=face_lm,
            pose_landmarks=pose_lm,
        )

        # Tile the single frame into a SEQUENCE_LENGTH-frame static window.
        # Velocity / acceleration are zero (correct for a static sign).
        tiled = np.tile(feat[np.newaxis, :], (sequence_length, 1)).astype(np.float32)
        seq = tiled[np.newaxis, :, :]  # (1, SEQUENCE_LENGTH, TOTAL_FEATURES_SIZE)
        np.save(img_dir / f"aug_{idx}.npy", seq)
        saved += 1

    mp_hands.close()
    if mp_face is not None:
        mp_face.close()
    if mp_pose is not None:
        mp_pose.close()

    return {
        'image': str(image_path),
        'gesture': gesture_label,
        'sequences': saved,
        'skipped_no_hand': skipped_no_hand,
    }


def _load_runtime_config():
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        import config as cfg
        runtime = {
            'sequence_length': cfg.SEQUENCE_LENGTH,
            'spatial_augmentations': 10,
            'extract_face_mesh': cfg.EXTRACT_FACE_MESH,
            'extract_pose': cfg.EXTRACT_POSE,
            'face_mesh_anchors': cfg.FACE_MESH_ANCHORS,
            'min_detection_confidence': cfg.MIN_DETECTION_CONFIDENCE,
            'hand_dropout_bridge_frames': cfg.HAND_DROPOUT_BRIDGE_FRAMES,
            'directional_letters': set(cfg.DIRECTIONAL_LETTERS),
            'directional_with_static_form': set(
                getattr(cfg, 'DIRECTIONAL_LETTERS_WITH_STATIC_FORM', set())
            ),
        }
        video_dir = cfg.VIDEO_DIR
        output_dir = str(Path(cfg.OUTPUT_DIR).parent / 'alphabet_augmented')
        return runtime, video_dir, output_dir
    except Exception:
        runtime = {
            'sequence_length': 45,
            'spatial_augmentations': 10,
            'extract_face_mesh': True,
            'extract_pose': True,
            'face_mesh_anchors': DEFAULT_FACE_ANCHORS,
            'min_detection_confidence': 0.5,
            'hand_dropout_bridge_frames': 3,
            'directional_letters': {'J', 'Z', 'Q', 'Ș', 'Ț', 'S'},
            'directional_with_static_form': {'Z'},
        }
        return runtime, str(Path(__file__).parent.parent / 'alphabet'), str(Path(__file__).parent.parent / 'data' / 'alphabet_augmented')


def main():
    runtime_cfg, video_dir, output_dir = _load_runtime_config()
    directional: Set[str] = {x.upper() for x in runtime_cfg['directional_letters']}
    directional_with_static: Set[str] = {x.upper() for x in runtime_cfg['directional_with_static_form']}
    # Letters skipped entirely (directional, no static form).
    skip_letters: Set[str] = directional - directional_with_static

    VIDEO_DIR = Path(video_dir)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('STATIC IMAGE EXTRACTION (jpg/png -> tiled sequences)')
    print('=' * 60)
    print(f'Sequence length: {runtime_cfg["sequence_length"]}')
    print(f'Per-frame features: {TOTAL_FEATURES_SIZE}')
    print(f'Visual augmentations per image: {runtime_cfg["spatial_augmentations"]}')
    print(f'Skipping directional letters: {sorted(skip_letters)}')
    if directional_with_static:
        print(f'Directional letters WITH valid static form (included): {sorted(directional_with_static)}')
    print('=' * 60)

    gesture_folders = sorted([f for f in VIDEO_DIR.iterdir() if f.is_dir()])

    tasks = []
    for folder in gesture_folders:
        if folder.name.upper() in skip_letters:
            continue
        for ext in IMAGE_EXTS:
            for img_file in folder.glob(ext):
                tasks.append((str(img_file), folder.name, str(OUTPUT_DIR), runtime_cfg))

    print(f'\nFound {len(tasks)} static images to process across '
          f'{len(set(t[1] for t in tasks))} non-directional classes')

    if not tasks:
        print('Nothing to do.')
        return

    NUM_WORKERS = min(4, cpu_count())
    print(f'Workers: {NUM_WORKERS}\n')

    completed = 0
    total_seqs = 0
    skipped = 0

    with Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(_process_one_image, tasks):
            completed += 1
            seqs = result.get('sequences', 0)
            total_seqs += seqs
            skipped += result.get('skipped_no_hand', 0)
            if 'error' in result:
                print(f"[{completed}/{len(tasks)}] ERROR ({Path(result['image']).name}): {result['error']}")
            elif completed % 50 == 0 or completed == len(tasks):
                print(f"[{completed}/{len(tasks)}] running total: {total_seqs} seqs, {skipped} skipped no-hand")

    info_path = OUTPUT_DIR / 'static_extraction_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'images_processed': len(tasks),
            'sequences_saved': total_seqs,
            'skipped_no_hand': skipped,
            'sequence_length': runtime_cfg['sequence_length'],
            'feature_size': TOTAL_FEATURES_SIZE,
            'directional_letters_skipped': sorted(directional),
            'created_at': datetime.now().isoformat(),
        }, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('STATIC EXTRACTION COMPLETE')
    print('=' * 60)
    print(f'Images processed:  {len(tasks)}')
    print(f'Sequences saved:   {total_seqs}')
    print(f'Skipped (no hand): {skipped}')
    print(f'Output:            {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
