"""Colab environment setup helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from constants import (
    COLAB_CODE_DIR,
    COLAB_DRIVE_MOUNT_POINT,
    DRIVE_PROJECT_DIR,
    HF_CACHE_DIRNAME,
    HF_HUB_CACHE_DIRNAME,
    HF_XDG_CACHE_DIRNAME,
    LOCAL_MODELS_CACHE_DIR,
    MASKS_DIR,
    OUTPUTS_DIR,
    RAW_IMAGES_DIR,
    SAM_WEIGHTS_FILENAME,
)


def print_colab_code_directory() -> None:
    """Print current Code/ directory details after the notebook changes cwd."""
    print(f"Working directory: {os.getcwd()}")
    print(f"Configured Colab code directory: {COLAB_CODE_DIR}")
    print("Code directory contents:")
    for item in sorted(os.listdir(".")):
        suffix = "/" if os.path.isdir(item) else ""
        print(f"  {item}{suffix}")


def install_requirements(requirements_path: str = "requirements.txt") -> None:
    """Install notebook dependencies and print GPU availability."""
    print(f"Installing packages from {requirements_path}.")
    process = subprocess.Popen(
        [sys.executable, "-m", "pip", "install", "-r", requirements_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout is not None:
        for output_line in process.stdout:
            print(output_line, end="", flush=True)
    process.wait()

    if process.returncode != 0:
        raise RuntimeError("Dependency installation failed. Review the pip output above.")

    import torch

    print("Dependency installation completed.")
    print(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        gpu_properties = torch.cuda.get_device_properties(0)
        print(f"GPU model: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {gpu_properties.total_memory / 1e9:.1f} GB")
    else:
        print("No GPU detected. Change the Colab runtime to a T4 GPU before running model cells.")


def create_data_directories() -> list[Path]:
    """Create project data directories needed by the pipeline."""
    required_directories = [
        RAW_IMAGES_DIR,
        MASKS_DIR,
        OUTPUTS_DIR,
        LOCAL_MODELS_CACHE_DIR,
    ]
    for directory in required_directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    print("Data directory structure is ready:")
    for directory in required_directories:
        print(f"  {directory}/")
    return [Path(directory) for directory in required_directories]


def configure_drive_cache() -> dict[str, str]:
    """Mount Google Drive and configure persistent model/cache locations."""
    from google.colab import drive

    print("Mounting Google Drive.")
    drive.mount(COLAB_DRIVE_MOUNT_POINT)
    print("Google Drive mounted.")

    drive_project_dir = Path(os.environ.get("CV2_DRIVE_WEIGHTS", DRIVE_PROJECT_DIR))
    drive_hf_home = drive_project_dir / HF_CACHE_DIRNAME
    drive_hub_cache = drive_hf_home / HF_HUB_CACHE_DIRNAME
    drive_xdg_cache = drive_hf_home / HF_XDG_CACHE_DIRNAME

    for directory in [drive_project_dir, drive_hf_home, drive_hub_cache, drive_xdg_cache]:
        directory.mkdir(parents=True, exist_ok=True)

    cache_environment = {
        "CV2_DRIVE_WEIGHTS": str(drive_project_dir),
        "HF_HOME": str(drive_hf_home),
        "HF_HUB_CACHE": str(drive_hub_cache),
        "TRANSFORMERS_CACHE": str(drive_hub_cache),
        "DIFFUSERS_CACHE": str(drive_hub_cache),
        "XDG_CACHE_HOME": str(drive_xdg_cache),
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TQDM_DISABLE": "1",
    }
    os.environ.update(cache_environment)

    print("Model cache environment:")
    for key in [
        "CV2_DRIVE_WEIGHTS",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
        "XDG_CACHE_HOME",
    ]:
        print(f"  {key}: {os.environ[key]}")

    sam_local_path = Path(SAM_WEIGHTS_FILENAME)
    sam_drive_path = drive_project_dir / SAM_WEIGHTS_FILENAME
    os.environ["SAM_CHECKPOINT_PATH"] = str(sam_local_path)

    print(f"Checking SAM weights: {SAM_WEIGHTS_FILENAME}")
    if sam_local_path.exists():
        print(f"  Local checkpoint found ({sam_local_path.stat().st_size / 1e9:.2f} GB).")
    elif sam_drive_path.exists():
        import shutil

        print(f"  Drive checkpoint found ({sam_drive_path.stat().st_size / 1e9:.2f} GB). Copying to runtime.")
        shutil.copy2(sam_drive_path, sam_local_path)
        print("  SAM checkpoint restored from Drive.")
    else:
        print("  SAM checkpoint not found. Cell 2.1 will download it once and persist it to Drive.")

    cache_size = subprocess.run(["du", "-sh", str(drive_hf_home)], capture_output=True, text=True)
    cache_size_text = cache_size.stdout.split()[0] if cache_size.stdout else "0B"
    if any(drive_hub_cache.iterdir()):
        print(f"Existing HuggingFace cache found in Drive ({cache_size_text}).")
    else:
        print(f"HuggingFace Drive cache is currently empty ({cache_size_text}).")

    return cache_environment
