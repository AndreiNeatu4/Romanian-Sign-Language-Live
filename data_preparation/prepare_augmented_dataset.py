"""
Prepare Augmented Dataset for Training
======================================

Loads all augmented .npy windows produced by extract_augmented_fast.py,
splits them into train/val/test, and writes dataset.pkl + class_labels.json.

Key change vs. the previous version: split happens at the *source-video*
level, not at the window level. With overlapping sliding windows, splitting
per window leaks information across train/val/test (the same gesture
fragment appears in both splits) and inflates accuracy. Splitting by source
video is the honest setup.

Also computes class weights (inverse frequency) so the trainer can give the
under-represented moving letters (J, Z, Q, Ș, Ț) full gradient signal.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _load_video_groups(augmented_dir: Path):
    """
    Returns:
        groups: list of dicts {label_idx, gesture_name, video_name, sequences (np.ndarray)}
        class_names: sorted gesture names
        feature_size: int
    """
    gesture_folders = sorted([f for f in augmented_dir.iterdir() if f.is_dir()])
    class_names = [f.name for f in gesture_folders]

    groups = []
    feature_size = None

    print(f"Found {len(class_names)} gesture classes:")
    for label_idx, gesture_folder in enumerate(gesture_folders):
        gesture_name = gesture_folder.name
        video_dirs = [f for f in gesture_folder.iterdir() if f.is_dir()]

        per_class_sequences = 0
        for video_dir in video_dirs:
            npy_files = sorted(video_dir.glob('*.npy'))
            video_seqs: List[np.ndarray] = []

            for npy_file in npy_files:
                try:
                    arr = np.load(npy_file)
                except Exception as e:
                    print(f"  WARN could not load {npy_file}: {e}")
                    continue
                if arr.ndim != 3 or len(arr) == 0:
                    continue
                if feature_size is None:
                    feature_size = arr.shape[-1]
                elif arr.shape[-1] != feature_size:
                    print(
                        f"  WARN feature-size mismatch in {npy_file.name}: "
                        f"got {arr.shape[-1]}, expected {feature_size}. SKIPPING."
                    )
                    continue
                video_seqs.append(arr)

            if not video_seqs:
                continue
            video_arr = np.concatenate(video_seqs, axis=0).astype(np.float32)
            groups.append({
                'label_idx': label_idx,
                'gesture_name': gesture_name,
                'video_name': video_dir.name,
                'sequences': video_arr,
            })
            per_class_sequences += len(video_arr)

        print(f"  [{label_idx}] {gesture_name}: {per_class_sequences} sequences from {len(video_dirs)} videos")

    return groups, class_names, feature_size


def _split_by_video(groups, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    """
    Split source videos per class into train/val/test, then concat their
    sequences. Each class gets at least one video in train; if it has 2+
    videos one goes to val, if 3+ one goes to test.
    """
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for g in groups:
        by_class[g['label_idx']].append(g)

    train, val, test = [], [], []
    for label_idx, items in by_class.items():
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            # Only one video for this class - it must go to train; we'll
            # also add a tiny slice of its windows to val/test to keep
            # metrics meaningful.
            seqs = items[0]['sequences']
            if len(seqs) >= 5:
                idx = rng.permutation(len(seqs))
                n_val = max(1, int(len(seqs) * val_ratio))
                n_test = max(1, int(len(seqs) * test_ratio))
                val_idx = idx[:n_val]
                test_idx = idx[n_val:n_val + n_test]
                train_idx = idx[n_val + n_test:]
                train.append({'label_idx': label_idx, 'sequences': seqs[train_idx]})
                val.append({'label_idx': label_idx, 'sequences': seqs[val_idx]})
                test.append({'label_idx': label_idx, 'sequences': seqs[test_idx]})
            else:
                train.append({'label_idx': label_idx, 'sequences': seqs})
            continue

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_val + n_test >= n:
            n_val = 1
            n_test = 1 if n >= 3 else 0
        n_train = n - n_val - n_test
        train_items = items[:n_train]
        val_items = items[n_train:n_train + n_val]
        test_items = items[n_train + n_val:]

        if train_items:
            train.append({
                'label_idx': label_idx,
                'sequences': np.concatenate([g['sequences'] for g in train_items], axis=0),
            })
        if val_items:
            val.append({
                'label_idx': label_idx,
                'sequences': np.concatenate([g['sequences'] for g in val_items], axis=0),
            })
        if test_items:
            test.append({
                'label_idx': label_idx,
                'sequences': np.concatenate([g['sequences'] for g in test_items], axis=0),
            })

    def _stack(parts):
        if not parts:
            return np.zeros((0, 0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
        Xs, ys = [], []
        for p in parts:
            Xs.append(p['sequences'])
            ys.append(np.full(len(p['sequences']), p['label_idx'], dtype=np.int64))
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_train, y_train = _stack(train)
    X_val, y_val = _stack(val)
    X_test, y_test = _stack(test)

    # Shuffle within each split.
    for X, y in ((X_train, y_train), (X_val, y_val), (X_test, y_test)):
        if len(X) > 0:
            idx = rng.permutation(len(X))
            X[:] = X[idx]
            y[:] = y[idx]

    return X_train, y_train, X_val, y_val, X_test, y_test


def _compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    """Inverse-frequency weights, normalized so mean(weights) == 1."""
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float64)
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights * (num_classes / weights.sum())
    return weights.astype(np.float32)


def create_dataset(
    augmented_dir: str,
    output_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    print('=' * 60)
    print('PREPARING AUGMENTED DATASET FOR TRAINING')
    print('=' * 60)
    augmented_dir = Path(augmented_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups, class_names, feature_size = _load_video_groups(augmented_dir)
    if not groups:
        print('No data found - did you run extract_augmented_fast.py first?')
        return None

    print(f"\nFeature size per frame: {feature_size}")

    X_train, y_train, X_val, y_val, X_test, y_test = _split_by_video(
        groups, train_ratio, val_ratio, test_ratio, random_seed
    )

    print('\nSplit by SOURCE VIDEO (no leakage between splits):')
    print(f'  Training:   {len(X_train)} sequences')
    print(f'  Validation: {len(X_val)} sequences')
    print(f'  Test:       {len(X_test)} sequences')

    print('\nClass distribution in training set:')
    counts = Counter(y_train.tolist())
    for idx, name in enumerate(class_names):
        print(f'  {name}: {counts.get(idx, 0)}')

    class_weights = _compute_class_weights(y_train, num_classes=len(class_names))

    sequence_length = X_train.shape[1] if len(X_train) > 0 else 0
    landmarks_per_frame = X_train.shape[2] if len(X_train) > 0 else 0

    dataset = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'class_weights': class_weights,
        'info': {
            'sequence_length': int(sequence_length),
            'landmarks_per_frame': int(landmarks_per_frame),
            'num_classes': len(class_names),
            'classes': class_names,
            'class_weights': class_weights.tolist(),
            'created_at': datetime.now().isoformat(),
            'source': 'augmented_data',
            'split_by': 'source_video',
        },
    }

    dataset_path = output_dir / 'dataset.pkl'
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    print(f'\nDataset saved to: {dataset_path}')

    labels_path = output_dir / 'class_labels.json'
    with open(labels_path, 'w', encoding='utf-8') as f:
        json.dump({
            'classes': class_names,
            'label_to_name': {i: n for i, n in enumerate(class_names)},
            'name_to_label': {n: i for i, n in enumerate(class_names)},
            'class_weights': class_weights.tolist(),
        }, f, indent=2, ensure_ascii=False)
    print(f'Class labels saved to: {labels_path}')

    print('\n' + '=' * 60)
    print('DATASET PREPARATION COMPLETE')
    print('=' * 60)
    print('\nNext step: python training/train_model.py')
    return dataset


def main():
    sys.path.append(str(Path(__file__).parent.parent))
    try:
        import config
        AUGMENTED_DIR = str(Path(config.OUTPUT_DIR).parent / 'alphabet_augmented')
        OUTPUT_DIR = config.OUTPUT_DIR
        train_ratio, val_ratio, test_ratio = config.TRAIN_SPLIT, config.VAL_SPLIT, config.TEST_SPLIT
    except Exception:
        AUGMENTED_DIR = str(Path(__file__).parent.parent / 'data' / 'alphabet_augmented')
        OUTPUT_DIR = str(Path(__file__).parent.parent / 'data' / 'alphabet_processed')
        train_ratio, val_ratio, test_ratio = 0.7, 0.15, 0.15

    create_dataset(
        augmented_dir=AUGMENTED_DIR,
        output_dir=OUTPUT_DIR,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        random_seed=42,
    )


if __name__ == '__main__':
    main()
