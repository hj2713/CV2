"""Module-level validation helpers for the Colab notebook."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

from constants import (
    DRIVE_PROJECT_DIR,
    DPT_CACHE_DIRNAME,
    HF_CACHE_DIRNAME,
    HF_HUB_CACHE_DIRNAME,
    HF_XDG_CACHE_DIRNAME,
    SAM_WEIGHTS_FILENAME,
    SAM_WEIGHTS_URL,
)
from notebook_helpers.scripts import run_python_script


def validate_sam_segmentation() -> None:
    """Download/restore SAM weights, persist them to Drive, and run Module 1."""
    drive_project_dir = Path(os.environ.get("CV2_DRIVE_WEIGHTS", DRIVE_PROJECT_DIR))
    sam_local_path = Path(SAM_WEIGHTS_FILENAME)
    sam_drive_path = drive_project_dir / SAM_WEIGHTS_FILENAME

    if not drive_project_dir.exists():
        raise RuntimeError("Drive cache is not configured. Run Cell 1.5 first.")

    if sam_local_path.exists():
        print(f"Local SAM checkpoint found ({sam_local_path.stat().st_size / 1e9:.2f} GB).")
    elif sam_drive_path.exists():
        print(f"Drive SAM checkpoint found ({sam_drive_path.stat().st_size / 1e9:.2f} GB). Copying to runtime.")
        shutil.copy2(sam_drive_path, sam_local_path)
    else:
        print("Downloading SAM checkpoint once.")
        subprocess.run(["wget", "-O", str(sam_local_path), SAM_WEIGHTS_URL], check=True)
        print(f"Download complete ({sam_local_path.stat().st_size / 1e9:.2f} GB).")

    print("Persisting SAM checkpoint to Drive.")
    shutil.copy2(sam_local_path, sam_drive_path)
    if sam_local_path.stat().st_size != sam_drive_path.stat().st_size:
        raise RuntimeError("SAM checkpoint verification failed: local and Drive file sizes differ.")

    os.environ["SAM_CHECKPOINT_PATH"] = str(sam_local_path)
    print("SAM checkpoint is ready. Running segmentation test.")
    run_python_script("core/1_segmentation.py")


def validate_illumination_estimators() -> None:
    """Validate Sobel-shading and DPT-Large estimators with Drive cache paths."""
    drive_project_dir = Path(os.environ.get("CV2_DRIVE_WEIGHTS", DRIVE_PROJECT_DIR))
    drive_hf_home = drive_project_dir / HF_CACHE_DIRNAME
    drive_hub_cache = drive_hf_home / HF_HUB_CACHE_DIRNAME
    drive_xdg_cache = drive_hf_home / HF_XDG_CACHE_DIRNAME

    if not drive_project_dir.exists():
        raise RuntimeError("Drive cache is not configured. Run Cell 1.5 first.")

    for key, value in {
        "CV2_DRIVE_WEIGHTS": drive_project_dir,
        "HF_HOME": drive_hf_home,
        "HF_HUB_CACHE": drive_hub_cache,
        "TRANSFORMERS_CACHE": drive_hub_cache,
        "DIFFUSERS_CACHE": drive_hub_cache,
        "XDG_CACHE_HOME": drive_xdg_cache,
    }.items():
        os.environ[key] = str(value)
        if key != "CV2_DRIVE_WEIGHTS":
            Path(value).mkdir(parents=True, exist_ok=True)

    print("Cache configuration before DPT-Large validation:")
    for key in ["HF_HOME", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "DIFFUSERS_CACHE", "XDG_CACHE_HOME"]:
        print(f"  {key}: {os.environ[key]}")

    cache_size_before = subprocess.run(["du", "-sh", str(drive_hf_home)], capture_output=True, text=True)
    print(f"Cache size before run: {(cache_size_before.stdout.split()[0] if cache_size_before.stdout else '?')}")

    run_python_script("core/2_illumination.py")

    cache_size_after = subprocess.run(["du", "-sh", str(drive_hf_home)], capture_output=True, text=True)
    print(f"Cache size after run: {(cache_size_after.stdout.split()[0] if cache_size_after.stdout else '?')}")
    print(f"Expected DPT cache directory: {drive_hub_cache / DPT_CACHE_DIRNAME}")
