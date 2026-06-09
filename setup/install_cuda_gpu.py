"""
CUDA Installation Helper for GPU Support
Guides you through installing CUDA Toolkit for TensorFlow GPU acceleration
"""

import os
import sys
import subprocess
from pathlib import Path
import urllib.request


def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70 + "\n")


def check_nvidia_driver():
    """Check if NVIDIA driver is installed."""
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("[OK] NVIDIA Driver detected")
            # Parse driver version
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Driver Version' in line:
                    print(f"     {line.strip()}")
                    break
            return True
        return False
    except FileNotFoundError:
        print("[X] NVIDIA Driver not found")
        return False


def check_cuda_installed():
    """Check if CUDA is already installed."""
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA")

    if cuda_path.exists():
        versions = [d.name for d in cuda_path.iterdir() if d.is_dir()]
        if versions:
            print(f"[INFO] Found CUDA installations: {', '.join(versions)}")
            return True, versions

    print("[INFO] No CUDA installation found")
    return False, []


def download_file(url, filename):
    """Download a file with progress."""
    print(f"\nDownloading {filename}...")
    print(f"URL: {url}")
    print("\nThis will open in your browser. Please download and run the installer.")

    # Open in browser
    import webbrowser
    webbrowser.open(url)

    return True


def main():
    print_header("CUDA GPU SETUP FOR TENSORFLOW")

    print("This script will help you enable GPU acceleration for TensorFlow.")
    print("Your RTX 4080 will provide significant speedup for model training.\n")

    # Step 1: Check NVIDIA driver
    print_header("STEP 1: Check NVIDIA Driver")
    if not check_nvidia_driver():
        print("\n[ERROR] NVIDIA Driver not found!")
        print("Please install the latest NVIDIA driver first:")
        print("https://www.nvidia.com/download/index.aspx")
        return

    # Step 2: Check existing CUDA
    print_header("STEP 2: Check CUDA Installation")
    cuda_installed, versions = check_cuda_installed()

    # Step 3: Install CUDA Toolkit 11.8
    print_header("STEP 3: Install CUDA Toolkit 11.8")

    if cuda_installed and 'v11.8' in versions:
        print("[OK] CUDA 11.8 is already installed!")
        cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")
    else:
        print("TensorFlow 2.15 requires CUDA Toolkit 11.8")
        print("\nOptions:")
        print("  A) Download CUDA 11.8 (Recommended)")
        print("  B) Skip (I'll install manually later)")
        print("  C) Cancel")

        choice = input("\nSelect option (A/B/C): ").strip().upper()

        if choice == 'A':
            cuda_url = "https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows&target_arch=x86_64"
            print("\n" + "-"*70)
            print("CUDA 11.8 INSTALLATION INSTRUCTIONS:")
            print("-"*70)
            print("1. Download CUDA Toolkit 11.8 from NVIDIA")
            print("2. Run the installer (requires admin rights)")
            print("3. Choose 'Express Installation' (recommended)")
            print("4. Wait for installation to complete (takes 5-10 minutes)")
            print("5. Restart your terminal when done")
            print("-"*70)

            download_file(cuda_url, "CUDA Toolkit 11.8")

            input("\nPress Enter after you've installed CUDA...")
        elif choice == 'B':
            print("\nYou can install CUDA manually later from:")
            print("https://developer.nvidia.com/cuda-11-8-0-download-archive")
        else:
            print("\nInstallation cancelled.")
            return

    # Step 4: Install cuDNN
    print_header("STEP 4: Install cuDNN 8.6")

    print("cuDNN is required for TensorFlow GPU acceleration")
    print("\nOptions:")
    print("  A) Download cuDNN 8.6 (Recommended)")
    print("  B) Skip (I'll install manually later)")
    print("  C) Cancel")

    choice = input("\nSelect option (A/B/C): ").strip().upper()

    if choice == 'A':
        cudnn_url = "https://developer.nvidia.com/cudnn-downloads"
        print("\n" + "-"*70)
        print("cuDNN INSTALLATION INSTRUCTIONS:")
        print("-"*70)
        print("1. Sign in to NVIDIA Developer (create free account if needed)")
        print("2. Download cuDNN 8.6 for CUDA 11.x (Windows)")
        print("3. Extract the zip file")
        print("4. Copy files to CUDA installation:")
        print("   - bin/*.dll     -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin\\")
        print("   - include/*.h   -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include\\")
        print("   - lib/x64/*.lib -> C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\lib\\x64\\")
        print("5. Add to System PATH (if not already):")
        print("   C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")
        print("-"*70)

        download_file(cudnn_url, "cuDNN 8.6")

        input("\nPress Enter after you've installed cuDNN...")
    elif choice == 'B':
        print("\nYou can install cuDNN manually later from:")
        print("https://developer.nvidia.com/cudnn")
    else:
        print("\nInstallation cancelled.")
        return

    # Step 5: Verify installation
    print_header("STEP 5: Verify GPU Setup")

    print("Testing TensorFlow GPU detection...")
    print("\nPlease restart your terminal and run:")
    print("  python -c \"import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))\"")
    print("\nIf GPU is detected, you're all set!")
    print("If not, check the PATH environment variable includes:")
    print("  C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin")

    print_header("ALTERNATIVE: Use PyTorch Instead")
    print("If CUDA installation is difficult, you can use PyTorch which has")
    print("better Windows GPU support:")
    print("\n  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
    print("\nThen modify train_model.py to use PyTorch instead of TensorFlow.")

    print_header("DONE")
    print("After installing CUDA and cuDNN:")
    print("1. Restart your terminal")
    print("2. Run: python verify_installation.py")
    print("3. GPU should be detected!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInstallation cancelled by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
