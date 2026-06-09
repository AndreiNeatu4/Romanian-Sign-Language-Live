"""
Gesture Data Extraction (single-pass, no augmentation).

This is the lightweight alternative to extract_augmented_fast.py. It uses the
SAME shared FrameFeatureExtractor so the per-frame feature space is identical
to what the live recognizer sees.

The result is still TEMPORAL SEQUENCES of `sequence_length` frames - one
video produces many overlapping windows, never one image per sign.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# Make the shared feature extractor importable.
sys.path.append(str(Path(__file__).parent.parent))

from data_preparation.feature_extractor import (  # noqa: E402
    DEFAULT_FACE_ANCHORS,
    FrameFeatureExtractor,
    TOTAL_FEATURES_SIZE,
    hand_presence_ratio,
    resample_frame_indices,
)


class GestureDataExtractor:
    def __init__(
        self,
        sequence_length: int = 45,
        target_fps: float = 30.0,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        extract_both_hands: bool = True,
        extract_pose: bool = True,
        extract_face_mesh: bool = True,
        face_mesh_anchors: Optional[Dict[str, int]] = None,
        normalize: bool = True,
        extract_features: bool = True,
        hand_dropout_bridge_frames: int = 3,
        min_hand_presence_ratio: float = 0.5,
        directional_letters: Optional[set] = None,
    ):
        self.sequence_length = sequence_length
        self.target_fps = target_fps
        self.normalize = normalize
        self.extract_features_flag = extract_features
        self.bridge_frames = hand_dropout_bridge_frames
        self.min_hand_presence = min_hand_presence_ratio
        self.directional_letters = set(directional_letters or set())
        self.face_mesh_anchors = face_mesh_anchors or DEFAULT_FACE_ANCHORS
        self.use_face_mesh = extract_face_mesh
        self.use_pose = extract_pose

        # MediaPipe.
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2 if extract_both_hands else 1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            model_complexity=1,
        )

        self.face_mesh = None
        if self.use_face_mesh:
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

        self.pose = None
        if self.use_pose:
            self.pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    # ------------------------------------------------------------------
    # Per-video processing
    # ------------------------------------------------------------------

    def process_video(
        self,
        video_path: str,
        gesture_label: str,
        sample_rate: int = 1,
        visualize: bool = False,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        source_fps = cap.get(cv2.CAP_PROP_FPS) or self.target_fps
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Read all frames first so we can resample to target FPS.
        all_frames: List[np.ndarray] = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        cap.release()

        # Resample.
        idx = resample_frame_indices(len(all_frames), source_fps, self.target_fps)
        frames = [all_frames[i] for i in idx]

        if sample_rate > 1:
            frames = frames[::sample_rate]

        ffe = FrameFeatureExtractor(
            face_anchors=self.face_mesh_anchors,
            bridge_frames=self.bridge_frames,
            use_face_mesh=self.use_face_mesh,
        )

        per_frame: List[np.ndarray] = []
        pbar = tqdm(total=len(frames), desc=f"Processing {Path(video_path).name}")
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

            if visualize and hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )
                cv2.imshow('Gesture Extraction', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            pbar.update(1)
        pbar.close()
        if visualize:
            cv2.destroyAllWindows()

        # Slice into overlapping windows.
        sequences: List[np.ndarray] = []
        feature_sequences: List[np.ndarray] = []  # legacy slot - now empty
        step = max(1, self.sequence_length // 2)
        for i in range(0, len(per_frame) - self.sequence_length + 1, step):
            seq = np.asarray(per_frame[i:i + self.sequence_length], dtype=np.float32)
            if len(seq) != self.sequence_length:
                continue
            if hand_presence_ratio(seq) < self.min_hand_presence:
                continue
            sequences.append(seq)

        metadata = {
            'video_path': video_path,
            'gesture_label': gesture_label,
            'source_fps': source_fps,
            'target_fps': self.target_fps,
            'total_frames': total_frames,
            'frames_after_resample': len(frames),
            'num_sequences': len(sequences),
            'sequence_length': self.sequence_length,
            'feature_size': TOTAL_FEATURES_SIZE,
            'timestamp': datetime.now().isoformat(),
        }
        return sequences, feature_sequences, metadata

    # ------------------------------------------------------------------
    # Dataset processing
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        video_dir: str,
        output_dir: str,
        gesture_mapping: Optional[Dict[str, str]] = None,
        sample_rate: int = 1,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        visualize: bool = False,
    ):
        video_dir_p = Path(video_dir)
        output_dir_p = Path(output_dir)
        output_dir_p.mkdir(parents=True, exist_ok=True)

        all_sequences: List[np.ndarray] = []
        all_labels: List[str] = []
        all_metadata: List[Dict] = []

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        gesture_folders = [f for f in video_dir_p.iterdir() if f.is_dir()]
        print(f"\nFound {len(gesture_folders)} gesture categories")

        for gesture_folder in gesture_folders:
            gesture_name = gesture_folder.name
            gesture_label = (gesture_mapping or {}).get(gesture_name, gesture_name)
            print(f"\n{'='*60}\nProcessing gesture: {gesture_label}\n{'='*60}")

            video_files = []
            for ext in video_extensions:
                video_files.extend(gesture_folder.glob(f'*{ext}'))
            print(f"Found {len(video_files)} videos")

            for video_file in video_files:
                try:
                    sequences, _, metadata = self.process_video(
                        str(video_file), gesture_label, sample_rate=sample_rate, visualize=visualize
                    )
                    all_sequences.extend(sequences)
                    all_labels.extend([gesture_label] * len(sequences))
                    all_metadata.append(metadata)
                    print(f"  OK {video_file.name}: {len(sequences)} sequences")
                except Exception as e:
                    print(f"  ERROR processing {video_file.name}: {e}")

        if not all_sequences:
            print("No sequences extracted - aborting.")
            return None

        X = np.asarray(all_sequences, dtype=np.float32)

        unique_labels = sorted(set(all_labels))
        label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
        y = np.array([label_to_idx[label] for label in all_labels])

        print(f"\nDataset shape: {X.shape}, classes: {len(unique_labels)}")

        # Stratified-ish random split (legacy script - the augmented pipeline
        # does proper stratified split via prepare_augmented_dataset.py).
        rng = np.random.default_rng(42)
        indices = rng.permutation(len(X))
        train_size = int(len(X) * train_split)
        val_size = int(len(X) * val_split)
        train_idx = indices[:train_size]
        val_idx = indices[train_size:train_size + val_size]
        test_idx = indices[train_size + val_size:]

        np.save(output_dir_p / 'X_train.npy', X[train_idx])
        np.save(output_dir_p / 'y_train.npy', y[train_idx])
        np.save(output_dir_p / 'X_val.npy', X[val_idx])
        np.save(output_dir_p / 'y_val.npy', y[val_idx])
        np.save(output_dir_p / 'X_test.npy', X[test_idx])
        np.save(output_dir_p / 'y_test.npy', y[test_idx])

        dataset_info = {
            'num_classes': len(unique_labels),
            'classes': unique_labels,
            'label_to_idx': label_to_idx,
            'sequence_length': self.sequence_length,
            'feature_size': TOTAL_FEATURES_SIZE,
            'total_sequences': len(X),
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'test_size': len(test_idx),
            'video_metadata': all_metadata,
            'created_at': datetime.now().isoformat(),
        }
        with open(output_dir_p / 'dataset_info.json', 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, indent=2, ensure_ascii=False)

        with open(output_dir_p / 'dataset.pkl', 'wb') as f:
            pickle.dump({
                'X_train': X[train_idx],
                'y_train': y[train_idx],
                'X_val': X[val_idx],
                'y_val': y[val_idx],
                'X_test': X[test_idx],
                'y_test': y[test_idx],
                'info': dataset_info,
            }, f)

        print(f"\nDataset saved to {output_dir_p}")
        return dataset_info


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        import config as cfg
    except ImportError:
        print("config.py not found - using defaults")
        cfg = None

    if cfg is None:
        VIDEO_DIR = 'training_videos'
        OUTPUT_DIR = 'processed_data'
        cfg_dict = {}
        SAMPLE_RATE = 1
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = 0.7, 0.15, 0.15
        VISUALIZE = False
        GESTURE_MAPPING = None
    else:
        if not cfg.validate_config():
            return
        VIDEO_DIR = cfg.VIDEO_DIR
        OUTPUT_DIR = cfg.OUTPUT_DIR
        cfg_dict = cfg.get_extraction_config()
        SAMPLE_RATE = cfg.SAMPLE_RATE
        TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = cfg.TRAIN_SPLIT, cfg.VAL_SPLIT, cfg.TEST_SPLIT
        VISUALIZE = cfg.VISUALIZE_EXTRACTION
        GESTURE_MAPPING = cfg.GESTURE_MAPPING or None

    extractor = GestureDataExtractor(**cfg_dict)
    extractor.process_dataset(
        video_dir=VIDEO_DIR,
        output_dir=OUTPUT_DIR,
        gesture_mapping=GESTURE_MAPPING,
        sample_rate=SAMPLE_RATE,
        train_split=TRAIN_SPLIT,
        val_split=VAL_SPLIT,
        test_split=TEST_SPLIT,
        visualize=VISUALIZE,
    )


if __name__ == '__main__':
    main()
