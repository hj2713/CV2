"""Batch-run helpers for Colab."""

from __future__ import annotations

import os
from pathlib import Path

from constants import IMAGE_EXTENSIONS, OUTPUTS_DIR, OUTPUTS_PER_IMAGE, RAW_IMAGES_DIR, TEST_IMAGE_NAME
from notebook_helpers.scripts import run_python_script


def prepare_run_id() -> tuple[int, Path, Path]:
    """Create the next versioned run directory and CSV path."""
    outputs_dir = Path(OUTPUTS_DIR)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    existing_run_ids = []
    for csv_path in outputs_dir.glob("results_*.csv"):
        try:
            existing_run_ids.append(int(csv_path.stem.replace("results_", "")))
        except ValueError:
            continue

    run_id = max(existing_run_ids) + 1 if existing_run_ids else 1
    run_outdir = outputs_dir / f"run_{run_id:03d}"
    run_csv = outputs_dir / f"results_{run_id:03d}.csv"
    run_outdir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in Path(RAW_IMAGES_DIR).iterdir()
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.name != TEST_IMAGE_NAME
    )

    print(f"Existing run IDs: {sorted(existing_run_ids) if existing_run_ids else 'none'}")
    print(f"Current run ID: {run_id:03d}")
    print(f"Output directory: {run_outdir}")
    print(f"Results CSV: {run_csv}")
    print(f"Images to process: {len(image_paths)}")
    print(f"Expected generated outputs: {len(image_paths) * OUTPUTS_PER_IMAGE}")
    return run_id, run_outdir, run_csv


def run_batch_pipeline(run_id: int) -> None:
    """Run main_pipeline.py with notebook cache validation."""
    if "HF_HUB_CACHE" not in os.environ or not os.path.exists(os.environ["HF_HUB_CACHE"]):
        raise RuntimeError("Drive/HuggingFace cache is not configured. Run Cell 1.5 first.")

    print(f"Starting batch run {run_id:03d}.")
    print(f"Drive cache: {os.environ['HF_HOME']}")
    print("Detailed paths are already validated in Cell 1.5.")
    run_python_script("main_pipeline.py", "--run-id", str(run_id))


def summarize_run_results(run_id: int, run_csv: Path) -> None:
    """Print compact tables and averages for a completed run."""
    import pandas as pd

    if not Path(run_csv).exists():
        raise FileNotFoundError(f"Results CSV not found: {run_csv}")

    results_df = pd.read_csv(run_csv)
    display_columns = [
        "image",
        "theta_sobel",
        "theta_dl",
        "sdcs_naive",
        "sdcs_sobel",
        "sdcs_dl",
        "lpips_naive",
        "lpips_sobel",
        "lpips_dl",
        "sobel_time_s",
        "dl_time_s",
        "status",
    ]
    display_columns = [column for column in display_columns if column in results_df.columns]

    print(f"Run {run_id:03d} results ({len(results_df)} image(s)):")
    print(results_df[display_columns].to_string(index=False))

    print("\nAverage SDCS (higher is better):")
    for column, label in [
        ("sdcs_naive", "Naive"),
        ("sdcs_sobel", "Sobel"),
        ("sdcs_dl", "DPT-Large"),
    ]:
        if column in results_df.columns:
            print(f"  {label}: {results_df[column].dropna().mean():.4f}")

    print("\nAverage LPIPS (lower is more perceptually similar to original):")
    for column, label in [
        ("lpips_naive", "Naive"),
        ("lpips_sobel", "Sobel"),
        ("lpips_dl", "DPT-Large"),
    ]:
        if column in results_df.columns:
            print(f"  {label}: {results_df[column].dropna().mean():.4f}")

    print("\nAverage illumination-estimation time:")
    for column, label in [("sobel_time_s", "Sobel"), ("dl_time_s", "DPT-Large")]:
        if column in results_df.columns:
            print(f"  {label}: {results_df[column].dropna().mean():.3f} seconds/image")

    if {"sdcs_sobel", "sdcs_naive"}.issubset(results_df.columns):
        improvement = results_df["sdcs_sobel"].dropna().mean() - results_df["sdcs_naive"].dropna().mean()
        print(f"\nSDCS improvement, Sobel versus naive: {improvement:+.4f}")
