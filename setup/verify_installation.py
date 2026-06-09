"""
Installation Verification Script
Run this to verify all packages are installed correctly.
"""

import sys


def check_package(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name.replace('-', '_')

    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'unknown')
        return True, version
    except ImportError as e:
        return False, str(e)


def main():
    print("="*70)
    print("  INSTALLATION VERIFICATION")
    print("="*70)
    print()

    packages = [
        ('opencv-python', 'cv2', 'OpenCV'),
        ('mediapipe', 'mediapipe', 'MediaPipe'),
        ('numpy', 'numpy', 'NumPy'),
        ('torch', 'torch', 'PyTorch'),
        ('tqdm', 'tqdm', 'tqdm'),
        ('scikit-learn', 'sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn', 'seaborn'),
        ('albumentations', 'albumentations', 'albumentations'),
        ('onnx', 'onnx', 'ONNX'),
    ]

    print("Checking packages...")
    print()

    all_ok = True
    for package, import_name, display_name in packages:
        success, version = check_package(package, import_name)

        if success:
            status = f"OK   ({version})"
            symbol = "[OK]"
        else:
            status = "FAILED"
            symbol = "[X]"
            all_ok = False

        print(f"  {symbol} {display_name:20s} {status}")

    print()
    print("-"*70)
    print()

    # Check GPU
    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"GPU Status: {gpu_count} GPU(s) detected")
            for i in range(gpu_count):
                print(f"  - {torch.cuda.get_device_name(i)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            gpu_ok = True
        else:
            print("GPU Status: No GPU detected (CPU-only mode)")
            print("  Note: GPU is optional. Training will work but be slower.")
            gpu_ok = False
    except Exception as e:
        print(f"GPU Status: Could not check GPU ({e})")
        gpu_ok = False

    print()
    print("="*70)

    if all_ok:
        print("STATUS: All packages installed successfully!")
        print()
        print("Next steps:")
        print("  1. Organize your training videos")
        print("  2. Run: python App.py")
        print("  3. Or read: QUICKSTART.md")

        if not gpu_ok:
            print()
            print("Optional: Enable GPU for faster training")
            print("  See INSTALLATION_STATUS.md for GPU setup instructions")
    else:
        print("STATUS: Some packages failed to install")
        print()
        print("Please run:")
        print("  pip install -r requirements.txt")

    print("="*70)
    print()


if __name__ == "__main__":
    main()
