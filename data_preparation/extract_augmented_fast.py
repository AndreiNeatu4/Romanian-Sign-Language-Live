"""
FAST Augmented Gesture Data Extraction Script
==============================================

Each video is processed as a TEMPORAL SEQUENCE: every frame is run through
MediaPipe in order, the per-frame feature vectors are stacked, and the stream
is sliced into overlapping windows of `sequence_length` frames. The model
later trains on *whole windows*, never on isolated frames.

Per-frame features are produced by `feature_extractor.FrameFeatureExtractor`
which is also used by `application/realtime_recognition.py` so training and
inference see identical feature spaces.

Improvements vs. the old version:
  - Body-anchored global wrist / fingertip trajectory (the signal that
    distinguishes J / Z / Q / S / Ș / Ț from static letters)
  - Velocity & acceleration channels
  - MediaPipe handedness for stable left/right slot assignment
  - Brief MediaPipe dropouts (<= 3 frames) are bridged with last-known
    landmarks instead of zero-padding
  - Source videos are resampled to a fixed target FPS so windows have a
    consistent wall-clock duration
  - Sequences with too few hand-detected frames are dropped at extraction
  - `directional_letters` (J, Z, Q, S, Ș, Ț) skip time-reverse / mirror
    augmentation, which would invert the sign's meaning
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
from datetime import datetime
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Set

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np

# Make the shared feature extractor importable when running this file directly.
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.feature_extractor import (
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    TOTAL_FEATURES_SIZE,
    hand_presence_ratio,
    resample_frame_indices,
)


def process_single_video(args):
    """Worker entry point - one process handles one (video, label) task."""
    video_path, gesture_label, output_dir, config = args
    try:
        extractor = VideoExtractor(config)
        return extractor.process_video(video_path, gesture_label, output_dir)
    except Exception as e:
        return {
            'video_path': str(video_path),
            'error': str(e),
            'total_sequences': 0,
        }


class VideoExtractor:
    """Single video extractor - one instance per worker process."""

    def __init__(self, config: dict):
        self.sequence_length: int = config['sequence_length']
        self.target_fps: float = config.get('target_fps', 30)
        self.spatial_augmentations: int = config['spatial_augmentations']
        self.speed_variations: List[float] = config['speed_variations']
        self.use_reverse: bool = config['use_reverse']
        self.overlapping_sequences: bool = config['overlapping_sequences']
        self.overlap_ratio: float = config['overlap_ratio']

        self.extract_face_mesh: bool = config.get('extract_face_mesh', True)
        self.extract_pose: bool = config.get('extract_pose', True)
        self.face_mesh_anchors: dict = config.get('face_mesh_anchors', DEFAULT_FACE_ANCHORS)

        self.bridge_frames: int = config.get('hand_dropout_bridge_frames', 3)
        self.min_hand_presence: float = config.get('min_hand_presence_ratio', 0.5)
        self.directional_letters: Set[str] = set(config.get('directional_letters', set()))

        # MediaPipe Hands.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.get('min_detection_confidence', 0.5),
            min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            model_complexity=1,  # 1 gives noticeably better handedness reliability
        )

        # MediaPipe Face Mesh.
        self.face_mesh = None
        if self.extract_face_mesh:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            )

        # MediaPipe Pose.
        self.pose = None
        if self.extract_pose:
            self.mp_pose = mp.solutions.pose
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=config.get('min_detection_confidence', 0.5),
                min_tracking_confidence=config.get('min_tracking_confidence', 0.5),
            )

        self.augmentation_pipelines = self._create_augmentation_pipelines()

    # ------------------------------------------------------------------
    # Augmentation pipelines (visual, never spatial-flipping)
    # ------------------------------------------------------------------

    def _create_augmentation_pipelines(self) -> List[A.Compose]:
        """No horizontal flip - left/right hand matters in signs."""
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
        return pipelines[: self.spatial_augmentations]

    def _apply_spatial_augmentation(self, frames, pipeline_idx: int):
        """Returns a list of augmented frames (uint8 ndarrays).

        Albumentations 2.x's batch `images=` API is not consistently supported
        across every transform we use, so we apply the pipeline per-frame.
        Slightly slower, but correct on every version >= 1.4.
        """
        if pipeline_idx >= len(self.augmentation_pipelines):
            return list(frames)
        pipeline = self.augmentation_pipelines[pipeline_idx]
        if pipeline_idx == 0:
            return list(frames)
        return [pipeline(image=f)['image'] for f in frames]

    @staticmethod
    def _apply_temporal_augmentation(
        frames: List[np.ndarray],
        speed: float,
        reverse: bool,
        min_length: int,
    ) -> List[np.ndarray]:
        if reverse:
            frames = frames[::-1]
        if abs(speed - 1.0) < 1e-6:
            return frames

        original_length = len(frames)
        target_length = int(original_length / speed)
        if target_length < min_length:
            target_length = min_length
        indices = np.linspace(0, original_length - 1, target_length).astype(int)
        return [frames[i] for i in indices]

    # ------------------------------------------------------------------
    # Main per-video processing
    # ------------------------------------------------------------------

    def process_video(self, video_path: str, gesture_label: str, output_dir: Path) -> Dict:
        video_name = Path(video_path).stem
        video_output_dir = output_dir / gesture_label / video_name
        video_output_dir.mkdir(parents=True, exist_ok=True)

        # Read all frames + source FPS.
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return {'video_path': str(video_path), 'total_sequences': 0, 'error': 'No frames'}

        # Resample to TARGET_FPS so window length = same wall-clock duration
        # regardless of source frame rate.
        idx = resample_frame_indices(len(frames), source_fps, self.target_fps)
        frames = [frames[i] for i in idx]

        # Letters whose meaning depends on direction must NOT be reversed.
        is_directional = gesture_label.upper() in {x.upper() for x in self.directional_letters}
        local_use_reverse = self.use_reverse and not is_directional

        total_sequences = 0
        augmentation_stats: List[Dict] = []
        skipped_low_quality = 0

        for spatial_idx in range(self.spatial_augmentations):
            for speed in self.speed_variations:
                # Time-reverse only for non-directional letters.
                reverse_settings = [False, True] if local_use_reverse else [False]

                for reverse in reverse_settings:
                    aug_name = f"s{spatial_idx}_sp{speed:.2f}_r{int(reverse)}"

                    aug_frames = self._apply_temporal_augmentation(
                        frames.copy(), speed, reverse, min_length=self.sequence_length
                    )

                    aug_frames_list = self._apply_spatial_augmentation(aug_frames, spatial_idx)

                    sequences, dropped = self._extract_sequences_from_frames(aug_frames_list)
                    skipped_low_quality += dropped

                    if len(sequences) > 0:
                        save_path = video_output_dir / f"{aug_name}.npy"
                        np.save(save_path, sequences)
                        total_sequences += len(sequences)
                        augmentation_stats.append({
                            'augmentation': aug_name,
                            'num_sequences': int(len(sequences)),
                            'dropped_low_quality': int(dropped),
                        })

        metadata = {
            'video_path': str(video_path),
            'video_name': video_name,
            'gesture_label': gesture_label,
            'is_directional': is_directional,
            'source_fps': source_fps,
            'target_fps': self.target_fps,
            'frames_after_resample': len(frames),
            'total_sequences': total_sequences,
            'skipped_low_quality': skipped_low_quality,
            'augmentations': augmentation_stats,
            'sequence_length': self.sequence_length,
            'feature_size': TOTAL_FEATURES_SIZE,
            'created_at': datetime.now().isoformat(),
        }

        with open(video_output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        # Close MediaPipe handles.
        self.hands.close()
        if self.face_mesh is not None:
            self.face_mesh.close()
        if self.pose is not None:
            self.pose.close()

        return metadata

    # ------------------------------------------------------------------
    # Frame loop (the heart of the extractor)
    # ------------------------------------------------------------------

    def _extract_sequences_from_frames(self, frames):
        """
        Run MediaPipe over the full frame stream IN ORDER, build per-frame
        feature vectors via the shared FrameFeatureExtractor, then slice into
        windows. Returns (sequences_array, num_sequences_dropped_low_quality).
        """
        ffe = FrameFeatureExtractor(
            face_anchors=self.face_mesh_anchors,
            bridge_frames=self.bridge_frames,
            use_face_mesh=self.extract_face_mesh,
        )

        per_frame: List[np.ndarray] = []
        for frame in frames:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            hand_results = self.hands.process(frame_rgb)
            face_results = self.face_mesh.process(frame_rgb) if self.face_mesh is not None else None
            pose_results = self.pose.process(frame_rgb) if self.pose is not None else None

            face_lm = face_results.multi_face_landmarks[0] if (face_results and face_results.multi_face_landmarks) else None
            pose_lm = pose_results.pose_landmarks if (pose_results and pose_results.pose_landmarks) else None

            feat = ffe.process_frame(
                hand_landmarks_list=hand_results.multi_hand_landmarks if hand_results else None,
                handedness_list=hand_results.multi_handedness if hand_results else None,
                face_landmarks=face_lm,
                pose_landmarks=pose_lm,
            )
            per_frame.append(feat)

        if not per_frame:
            return np.array([]), 0

        # Window slicing.
        if self.overlapping_sequences:
            step_size = max(1, int(self.sequence_length * (1 - self.overlap_ratio)))
        else:
            step_size = self.sequence_length

        sequences: List[np.ndarray] = []
        dropped = 0
        for i in range(0, len(per_frame) - self.sequence_length + 1, step_size):
            seq = np.asarray(per_frame[i:i + self.sequence_length], dtype=np.float32)
            if len(seq) != self.sequence_length:
                continue
            if hand_presence_ratio(seq) < self.min_hand_presence:
                dropped += 1
                continue
            sequences.append(seq)

        return (np.asarray(sequences, dtype=np.float32) if sequences else np.array([])), dropped


def _load_runtime_config():
    """Pull runtime knobs from config.py with safe fallbacks."""
    try:
        sys.path.append(str(Path(__file__).parent.parent))
        import config as cfg
        return {
            'sequence_length': cfg.SEQUENCE_LENGTH,
            'target_fps': getattr(cfg, 'TARGET_FPS', 30),
            'spatial_augmentations': 10,
            # Tighter speed range for moving signs - large speed changes
            # destroy the trajectory shape we now rely on.
            'speed_variations': [0.9, 1.0, 1.1],
            'use_reverse': False,
            'overlapping_sequences': True,
            'overlap_ratio': 0.5,
            'extract_face_mesh': cfg.EXTRACT_FACE_MESH,
            'extract_pose': cfg.EXTRACT_POSE,
            'face_mesh_anchors': cfg.FACE_MESH_ANCHORS,
            'min_detection_confidence': cfg.MIN_DETECTION_CONFIDENCE,
            'min_tracking_confidence': cfg.MIN_TRACKING_CONFIDENCE,
            'hand_dropout_bridge_frames': cfg.HAND_DROPOUT_BRIDGE_FRAMES,
            'min_hand_presence_ratio': cfg.MIN_HAND_PRESENCE_RATIO,
            'directional_letters': cfg.DIRECTIONAL_LETTERS,
        }, cfg.VIDEO_DIR, str(Path(cfg.OUTPUT_DIR).parent / 'alphabet_augmented')
    except Exception:
        return {
            'sequence_length': 45,
            'target_fps': 30,
            'spatial_augmentations': 5,
            'speed_variations': [0.9, 1.0, 1.1],
            'use_reverse': False,
            'overlapping_sequences': True,
            'overlap_ratio': 0.5,
            'extract_face_mesh': True,
            'extract_pose': True,
            'face_mesh_anchors': DEFAULT_FACE_ANCHORS,
            'min_detection_confidence': 0.5,
            'min_tracking_confidence': 0.5,
            'hand_dropout_bridge_frames': 3,
            'min_hand_presence_ratio': 0.5,
            'directional_letters': {'J', 'Z', 'Q', 'Ș', 'Ț', 'S'},
        }, str(Path(__file__).parent.parent / 'alphabet'), str(Path(__file__).parent.parent / 'data' / 'alphabet_augmented')


def main():
    runtime_cfg, video_dir, output_dir = _load_runtime_config()

    VIDEO_DIR = Path(video_dir)
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    multiplier = runtime_cfg['spatial_augmentations'] * len(runtime_cfg['speed_variations'])
    if runtime_cfg['use_reverse']:
        multiplier *= 2
    if runtime_cfg['overlapping_sequences']:
        multiplier = int(multiplier * 1.5)  # rough estimate

    NUM_WORKERS = min(4, cpu_count())

    print('=' * 60)
    print('FAST AUGMENTED DATA EXTRACTION (v2)')
    print('=' * 60)
    print(f'Workers: {NUM_WORKERS}')
    print(f'Sequence length:  {runtime_cfg["sequence_length"]} frames @ {runtime_cfg["target_fps"]}fps')
    print(f'Per-frame features: {TOTAL_FEATURES_SIZE}')
    print(f'Spatial augs: {runtime_cfg["spatial_augmentations"]}')
    print(f'Speed variations: {runtime_cfg["speed_variations"]}')
    print(f'Reverse augmentation: {runtime_cfg["use_reverse"]} (skipped for {sorted(runtime_cfg["directional_letters"])})')
    print(f'Approx multiplier: ~{multiplier}x')
    print('=' * 60)

    # Collect tasks.
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    gesture_folders = sorted([f for f in VIDEO_DIR.iterdir() if f.is_dir()])
    print(f'\nFound {len(gesture_folders)} gesture categories')

    tasks = []
    for folder in gesture_folders:
        for ext in video_extensions:
            for video_file in folder.glob(f'*{ext}'):
                tasks.append((str(video_file), folder.name, OUTPUT_DIR, runtime_cfg))

    print(f'Total videos to process: {len(tasks)}')
    if not tasks:
        print('Nothing to do.')
        return

    print('\nProcessing...\n')

    total_sequences = 0
    completed = 0
    errors = 0

    with Pool(NUM_WORKERS) as pool:
        for result in pool.imap_unordered(process_single_video, tasks):
            completed += 1
            seqs = result.get('total_sequences', 0)
            if 'error' in result and seqs == 0:
                errors += 1
                print(f"[{completed}/{len(tasks)}] ERROR ({Path(result['video_path']).name}): {result['error']}")
            else:
                total_sequences += seqs
                video_name = Path(result.get('video_path', '')).stem
                dropped = result.get('skipped_low_quality', 0)
                print(f"[{completed}/{len(tasks)}] {video_name}: {seqs} sequences (dropped {dropped} low-quality)")

    dataset_info = {
        'total_videos': len(tasks),
        'total_sequences': total_sequences,
        'errors': errors,
        'config': {k: (sorted(v) if isinstance(v, set) else v) for k, v in runtime_cfg.items()},
        'feature_size': TOTAL_FEATURES_SIZE,
        'created_at': datetime.now().isoformat(),
    }

    with open(OUTPUT_DIR / 'dataset_info.json', 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)

    print('\n' + '=' * 60)
    print('EXTRACTION COMPLETE')
    print('=' * 60)
    print(f'Videos processed: {len(tasks)}')
    print(f'Total sequences: {total_sequences}')
    print(f'Errors: {errors}')
    print(f'Output: {OUTPUT_DIR}')
    print('\nNext: python data_preparation/prepare_augmented_dataset.py')


if __name__ == '__main__':
    main()
