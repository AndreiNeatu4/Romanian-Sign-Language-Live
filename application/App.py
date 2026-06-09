"""
Gesture Recognition System - Main Launcher
Author: AI Assistant
Optimized for NVIDIA RTX 4080

This is the main entry point for the gesture recognition system.
Choose from the menu below to run different parts of the pipeline.
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))


def print_banner():
    """Print welcome banner."""
    print("\n" + "="*70)
    print("  GESTURE RECOGNITION SYSTEM")
    print("  Powered by MediaPipe & Deep Learning")
    print("  Optimized for NVIDIA RTX 4080")
    print("="*70)


def print_menu():
    """Print main menu."""
    print("\nMAIN MENU:\n")
    print("  1. Test single video (verify setup)")
    print("  2. Extract data from all videos")
    print("  3. Visualize extracted data")
    print("  4. Train model")
    print("  5. View configuration")
    print("  6. Quick start guide")
    print("  7. Exit")
    print()


def test_single_video():
    """Run single video test."""
    print("\n" + "="*70)
    print("TEST SINGLE VIDEO")
    print("="*70)

    video_path = input("\nEnter path to video file: ").strip()

    if not Path(video_path).exists():
        print(f"Error: File not found: {video_path}")
        return

    import subprocess
    subprocess.run([sys.executable, "test_single_video.py", video_path])


def extract_data():
    """Run data extraction."""
    print("\n" + "="*70)
    print("EXTRACT GESTURE DATA")
    print("="*70)

    response = input("\nThis will process all videos in your VIDEO_DIR. Continue? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    import subprocess
    subprocess.run([sys.executable, "extract_gesture_data.py"])


def visualize_data():
    """Visualize extracted data."""
    print("\n" + "="*70)
    print("VISUALIZE DATA")
    print("="*70)

    if not Path("processed_data/dataset.pkl").exists():
        print("\nNo data found. Run 'Extract data' first!")
        return

    import subprocess
    subprocess.run([sys.executable, "visualize_data.py"])


def train_model():
    """Train the model."""
    print("\n" + "="*70)
    print("TRAIN MODEL")
    print("="*70)

    if not Path("processed_data/dataset.pkl").exists():
        print("\nNo data found. Run 'Extract data' first!")
        return

    response = input("\nThis will start model training. Continue? (y/n): ")

    if response.lower() != 'y':
        print("Cancelled.")
        return

    import subprocess
    subprocess.run([sys.executable, "train_model.py"])


def view_config():
    """View current configuration."""
    print("\n" + "="*70)
    print("CURRENT CONFIGURATION")
    print("="*70)

    try:
        import config
        config.validate_config()

        print("\nDirectories:")
        print(f"   Video directory:  {config.VIDEO_DIR}")
        print(f"   Output directory: {config.OUTPUT_DIR}")
        print(f"   Model directory:  {config.MODEL_DIR}")

        print("\nData Extraction:")
        print(f"   Sequence length:   {config.SEQUENCE_LENGTH}")
        print(f"   Both hands:        {config.EXTRACT_BOTH_HANDS}")
        print(f"   Normalize:         {config.NORMALIZE_COORDINATES}")
        print(f"   Extract features:  {config.EXTRACT_FEATURES}")

        print("\nModel Training:")
        print(f"   Model type:        {config.MODEL_TYPE}")
        print(f"   Epochs:            {config.EPOCHS}")
        print(f"   Batch size:        {config.BATCH_SIZE}")
        print(f"   Learning rate:     {config.LEARNING_RATE}")

        print("\nEdit config.py to change these settings")

    except ImportError:
        print("\nconfig.py not found!")
        print("   Copy config.py from the provided template")


def show_quickstart():
    """Show quick start guide."""
    quickstart_path = Path("QUICKSTART.md")

    if quickstart_path.exists():
        print("\n" + "="*70)
        print("QUICK START GUIDE")
        print("="*70)
        print()

        with open(quickstart_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Print first 50 lines
            lines = content.split('\n')[:50]
            print('\n'.join(lines))

        print("\n\nSee QUICKSTART.md for the full guide")
    else:
        print("\nQUICKSTART.md not found")


def check_setup():
    """Check if setup is complete."""
    issues = []

    # Check for config file
    if not Path("config.py").exists():
        issues.append("WARNING: config.py not found (will use defaults)")

    # Check for video directory
    try:
        import config
        if not Path(config.VIDEO_DIR).exists():
            issues.append(f"WARNING: Video directory not found: {config.VIDEO_DIR}")
    except:
        if not Path("training_videos").exists():
            issues.append("WARNING: training_videos directory not found")

    # Check for required packages
    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
        import torch  # noqa: F401
        import numpy  # noqa: F401
    except ImportError as e:
        issues.append(f"ERROR: Missing package: {e.name}")
        issues.append("   Run: pip install -r requirements.txt")

    # Check GPU
    try:
        import torch
        gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if len(gpus) == 0:
            issues.append("INFO: No GPU detected (training will be slower)")
    except:
        pass

    if issues:
        print("\nSetup Issues:")
        for issue in issues:
            print(f"   {issue}")
        print()
    else:
        print("\nSetup looks good!")


def main():
    """Main application loop."""
    print_banner()
    check_setup()

    while True:
        print_menu()

        try:
            choice = input("Select option (1-7): ").strip()

            if choice == '1':
                test_single_video()
            elif choice == '2':
                extract_data()
            elif choice == '3':
                visualize_data()
            elif choice == '4':
                train_model()
            elif choice == '5':
                view_config()
            elif choice == '6':
                show_quickstart()
            elif choice == '7':
                print("\nGoodbye!\n")
                break
            else:
                print("\nInvalid choice. Please select 1-7.")

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\nError: {e}")

        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
