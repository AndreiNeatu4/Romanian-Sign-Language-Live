"""
Visualize extracted gesture data.
Use this to verify your data extraction worked correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path


def load_data(data_dir="processed_data"):
    """Load the processed dataset."""
    data_path = Path(data_dir) / 'dataset.pkl'

    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {data_path}\n"
            "Run extract_gesture_data.py first!"
        )

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    return data


def plot_dataset_overview(data):
    """Plot overview of the dataset."""
    info = data['info']

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Class distribution
    ax = axes[0, 0]
    y_train = data['y_train']
    unique, counts = np.unique(y_train, return_counts=True)
    class_names = [info['classes'][i] for i in unique]

    ax.bar(class_names, counts, color='skyblue', edgecolor='black')
    ax.set_title('Training Data Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Gesture Class')
    ax.set_ylabel('Number of Sequences')
    ax.grid(axis='y', alpha=0.3)

    for i, (name, count) in enumerate(zip(class_names, counts)):
        ax.text(i, count, str(count), ha='center', va='bottom')

    # 2. Train/Val/Test split
    ax = axes[0, 1]
    splits = ['Train', 'Validation', 'Test']
    sizes = [info['train_size'], info['val_size'], info['test_size']]
    colors = ['#66b3ff', '#99ff99', '#ffcc99']

    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=splits,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90
    )

    for autotext in autotexts:
        autotext.set_color('black')
        autotext.set_fontweight('bold')

    ax.set_title('Dataset Split', fontsize=14, fontweight='bold')

    # 3. Sequence length distribution
    ax = axes[1, 0]
    X_train = data['X_train']
    sequence_lengths = [X_train.shape[1]]

    ax.bar(['Sequence Length'], sequence_lengths, color='lightcoral', edgecolor='black')
    ax.set_title('Sequence Configuration', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Frames')
    ax.grid(axis='y', alpha=0.3)
    ax.text(0, sequence_lengths[0], str(sequence_lengths[0]), ha='center', va='bottom')

    # 4. Feature dimensions
    ax = axes[1, 1]
    feature_info = [
        ('Landmarks\nper frame', info['landmarks_per_frame']),
        ('Features\nper frame', info.get('features_per_frame', 0)),
        ('Total\nclasses', info['num_classes'])
    ]

    labels = [x[0] for x in feature_info]
    values = [x[1] for x in feature_info]
    colors_bars = ['#ff9999', '#66b3ff', '#99ff99']

    bars = ax.bar(labels, values, color=colors_bars, edgecolor='black')
    ax.set_title('Dataset Dimensions', fontsize=14, fontweight='bold')
    ax.set_ylabel('Count')
    ax.grid(axis='y', alpha=0.3)

    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(value)}', ha='center', va='bottom')

    plt.tight_layout()
    return fig


def plot_sample_sequences(data, num_samples=3):
    """Plot sample sequences from each class."""
    info = data['info']
    X_train = data['X_train']
    y_train = data['y_train']
    classes = info['classes']
    num_classes = len(classes)

    fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 4*num_classes))

    if num_classes == 1:
        axes = axes.reshape(1, -1)

    for class_idx, class_name in enumerate(classes):
        # Find samples of this class
        class_samples = X_train[y_train == class_idx]

        if len(class_samples) == 0:
            continue

        # Select random samples
        sample_indices = np.random.choice(
            len(class_samples),
            min(num_samples, len(class_samples)),
            replace=False
        )

        for i, sample_idx in enumerate(sample_indices):
            ax = axes[class_idx, i] if num_classes > 1 else axes[i]

            sequence = class_samples[sample_idx]

            # Reshape to (sequence_length, num_landmarks, 3)
            # Assuming 2 hands * 21 landmarks * 3 coords = 126 per frame
            num_coords = sequence.shape[-1]

            # Plot each coordinate dimension
            ax.plot(sequence[:, :min(num_coords, 63)], alpha=0.3, linewidth=0.5)

            ax.set_title(f'{class_name} - Sample {i+1}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Coordinate Value')
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_landmark_variance(data):
    """Plot variance of each landmark across all sequences."""
    X_train = data['X_train']

    # Calculate variance across all sequences and frames
    # Shape: (num_sequences, sequence_length, num_features)
    variance = np.var(X_train, axis=(0, 1))

    fig, ax = plt.subplots(figsize=(15, 6))

    ax.bar(range(len(variance)), variance, color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_title('Feature Variance (High variance = More informative)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Variance')
    ax.grid(axis='y', alpha=0.3)

    # Highlight top 10 most variant features
    top_indices = np.argsort(variance)[-10:]
    ax.bar(top_indices, variance[top_indices], color='coral', edgecolor='black', alpha=0.9)

    plt.tight_layout()
    return fig


def print_dataset_info(data):
    """Print detailed dataset information."""
    info = data['info']

    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)

    print(f"\nDataset Size:")
    print(f"   Total sequences: {info['total_sequences']}")
    print(f"   Training:        {info['train_size']} ({info['train_size']/info['total_sequences']*100:.1f}%)")
    print(f"   Validation:      {info['val_size']} ({info['val_size']/info['total_sequences']*100:.1f}%)")
    print(f"   Test:            {info['test_size']} ({info['test_size']/info['total_sequences']*100:.1f}%)")

    print(f"\nGesture Classes ({info['num_classes']}):")
    for i, class_name in enumerate(info['classes']):
        num_samples = np.sum(data['y_train'] == i)
        print(f"   {i}: {class_name} ({num_samples} training sequences)")

    print(f"\nSequence Dimensions:")
    print(f"   Sequence length:       {info['sequence_length']} frames")
    print(f"   Landmarks per frame:   {info['landmarks_per_frame']}")
    print(f"   Features per frame:    {info.get('features_per_frame', 0)}")

    print(f"\nProcessing Settings:")
    print(f"   Normalized:            {info['normalized']}")
    print(f"   Extracted features:    {info['extracted_features']}")

    print(f"\nCreated: {info['created_at']}")

    print("\n" + "="*60)


def main():
    """Main visualization function."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize gesture dataset')
    parser.add_argument(
        '--data-dir',
        type=str,
        default='processed_data',
        help='Directory containing processed data'
    )
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save plots to files instead of displaying'
    )

    args = parser.parse_args()

    print("Loading dataset...")
    data = load_data(args.data_dir)

    # Print info
    print_dataset_info(data)

    # Create visualizations
    print("\nGenerating visualizations...")

    fig1 = plot_dataset_overview(data)
    fig2 = plot_sample_sequences(data, num_samples=3)
    fig3 = plot_landmark_variance(data)

    if args.save:
        output_dir = Path(args.data_dir) / 'visualizations'
        output_dir.mkdir(exist_ok=True)

        fig1.savefig(output_dir / 'dataset_overview.png', dpi=300, bbox_inches='tight')
        fig2.savefig(output_dir / 'sample_sequences.png', dpi=300, bbox_inches='tight')
        fig3.savefig(output_dir / 'feature_variance.png', dpi=300, bbox_inches='tight')

        print(f"\nVisualizations saved to {output_dir}")
    else:
        print("\nDisplaying visualizations (close windows to exit)")
        print("   Run with --save flag to save plots to files")
        plt.show()


if __name__ == "__main__":
    main()
