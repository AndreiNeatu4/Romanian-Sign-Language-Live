"""
Simple cuDNN installer for CUDA 11.8
Run this script to automatically install cuDNN
"""

import os
import shutil
from pathlib import Path
import zipfile


def find_cudnn_zip():
    """Find cuDNN zip in Downloads folder."""
    # Check multiple possible download locations
    possible_paths = [
        Path("D:/Downloads"),
        Path.home() / "Downloads",
        Path("C:/Users") / Path.home().name / "Downloads"
    ]

    cudnn_files = []
    downloads = None

    for path in possible_paths:
        if path.exists():
            files = list(path.glob("cudnn*.zip"))
            if files:
                cudnn_files = files
                downloads = path
                break

    if not cudnn_files:
        print("[X] No cuDNN zip file found in Downloads folder")
        print("\nChecked locations:")
        for path in possible_paths:
            print(f"  - {path}")
        print("\nPlease:")
        print("1. Download cuDNN from: https://developer.nvidia.com/cudnn-downloads")
        print("2. Choose: cuDNN for CUDA 11.x (Windows)")
        print("3. Save to your Downloads folder")
        print("4. Run this script again")
        return None

    # Use the most recent file
    cudnn_zip = max(cudnn_files, key=lambda p: p.stat().st_mtime)
    print(f"[OK] Found cuDNN: {cudnn_zip.name}")
    print(f"[OK] Location: {cudnn_zip.parent}")
    return cudnn_zip


def extract_cudnn(zip_path):
    """Extract cuDNN zip file."""
    extract_path = zip_path.parent / "cudnn_extracted"

    if extract_path.exists():
        shutil.rmtree(extract_path)

    print(f"\nExtracting {zip_path.name}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Find the actual cudnn folder (might be nested)
    cudnn_folder = None
    for item in extract_path.rglob("bin"):
        if item.is_dir() and (item / "cudnn64_8.dll").exists():
            cudnn_folder = item.parent
            break

    if not cudnn_folder:
        # Try direct structure
        if (extract_path / "bin").exists():
            cudnn_folder = extract_path

    if not cudnn_folder:
        print("[X] Could not find cuDNN files in extracted archive")
        return None

    print(f"[OK] Extracted to: {cudnn_folder}")
    return cudnn_folder


def install_cudnn(cudnn_folder):
    """Copy cuDNN files to CUDA installation."""
    cuda_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8")

    if not cuda_path.exists():
        print(f"[X] CUDA 11.8 not found at: {cuda_path}")
        return False

    print(f"\nInstalling cuDNN to CUDA 11.8...")

    # Copy bin files (DLLs)
    src_bin = cudnn_folder / "bin"
    dst_bin = cuda_path / "bin"
    if src_bin.exists():
        print("  Copying DLL files...")
        for dll in src_bin.glob("*.dll"):
            try:
                shutil.copy2(dll, dst_bin)
                print(f"    [OK] {dll.name}")
            except PermissionError:
                print(f"    [X] {dll.name} (Run as Administrator)")
                return False

    # Copy include files (headers)
    src_include = cudnn_folder / "include"
    dst_include = cuda_path / "include"
    if src_include.exists():
        print("  Copying header files...")
        for header in src_include.glob("*.h"):
            try:
                shutil.copy2(header, dst_include)
                print(f"    [OK] {header.name}")
            except PermissionError:
                print(f"    [X] {header.name} (Run as Administrator)")
                return False

    # Copy lib files
    src_lib = cudnn_folder / "lib" / "x64"
    if not src_lib.exists():
        src_lib = cudnn_folder / "lib"

    dst_lib = cuda_path / "lib" / "x64"
    if src_lib.exists():
        print("  Copying library files...")
        for lib in src_lib.glob("*.lib"):
            try:
                shutil.copy2(lib, dst_lib)
                print(f"    [OK] {lib.name}")
            except PermissionError:
                print(f"    [X] {lib.name} (Run as Administrator)")
                return False

    return True


def verify_installation():
    """Verify cuDNN was installed correctly."""
    cuda_bin = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin")

    cudnn_dlls = list(cuda_bin.glob("cudnn*.dll"))

    if cudnn_dlls:
        print(f"\n[OK] cuDNN installed successfully!")
        print(f"  Found {len(cudnn_dlls)} cuDNN DLL files")
        return True
    else:
        print("\n[X] cuDNN installation failed")
        return False


def main():
    print("="*70)
    print("  cuDNN Installation Script")
    print("="*70)
    print()

    # Check for admin rights
    try:
        test_path = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\test.tmp")
        test_path.touch()
        test_path.unlink()
        print("[OK] Running with Administrator privileges")
    except:
        print("[WARNING] Not running as Administrator")
        print("   Some files may fail to copy")
        print("   If installation fails, right-click and 'Run as Administrator'")
        print()

    # Find cuDNN zip
    cudnn_zip = find_cudnn_zip()
    if not cudnn_zip:
        return

    # Extract
    cudnn_folder = extract_cudnn(cudnn_zip)
    if not cudnn_folder:
        return

    # Install
    if install_cudnn(cudnn_folder):
        # Verify
        if verify_installation():
            print("\n" + "="*70)
            print("  Installation Complete!")
            print("="*70)
            print()
            print("Next steps:")
            print("1. Restart your terminal")
            print("2. Run: python verify_installation.py")
            print("3. GPU should now be detected!")
            print()
        else:
            print("\nPlease run this script as Administrator")
    else:
        print("\n[X] Installation failed")
        print("Please run this script as Administrator:")
        print("  Right-click install_cudnn_simple.py -> Run as Administrator")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()

    input("\nPress Enter to exit...")
