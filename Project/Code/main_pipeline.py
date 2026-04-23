"""
main_pipeline.py - Batch processing script for the full benchmark.

Run on Colab with:
    !python main_pipeline.py               # Auto-increments run ID
    !python main_pipeline.py --run-id 3   # Forces run ID = 3

What this does:
  1. Reads all images from data/raw_images/
  2. For each image: SAM segmentation, Sobel + DL light estimation, 3 composites
  3. Saves all outputs to data/outputs/run_NNN/
  4. Computes SDCS on all outputs
  5. Saves results to data/outputs/results_NNN.csv (never overwrites old runs)
"""
import os
import sys
import csv
import glob
import time
import argparse
import importlib.util
import contextlib
from io import StringIO

import cv2
import numpy as np
import torch
from PIL import Image

from constants import (
    DEFAULT_BACKGROUND_STYLE,
    GENERATED_VARIANTS,
    IMAGE_EXTENSIONS,
    LPIPS_BACKBONE,
    LPIPS_IMAGE_SIZE,
    LPIPS_NORMALIZATION_OFFSET,
    LPIPS_NORMALIZATION_SCALE,
    MASKS_DIR,
    OUTPUTS_DIR,
    RAW_IMAGES_DIR,
    TEST_IMAGE_NAME,
)

_lpips_model = None
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("CV2_SHOW_GENERATION_PROMPTS", "0")

# -- Load modules (importlib handles filenames starting with digits) -----------

def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join("core", filename)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


print("Loading pipeline modules.")
seg_mod  = _load_module("1_segmentation.py", "segmentation")
illu_mod = _load_module("2_illumination.py", "illumination")
gen_mod  = _load_module("3_generation.py",   "generation")
eval_mod = _load_module("4_evaluation.py",   "evaluation")

segment_product                = seg_mod.segment_product
estimate_light_direction_sobel = illu_mod.estimate_light_direction_sobel
estimate_light_direction_dl    = illu_mod.estimate_light_direction_dl
generate_background            = gen_mod.generate_background
build_generation_prompt        = gen_mod.build_generation_prompt
shadow_vector_from_light_angle = gen_mod.shadow_vector_from_light_angle
compute_sdcs                   = eval_mod.compute_sdcs


# -- Configuration -------------------------------------------------------------

RAW_IMAGES_DIR = str(RAW_IMAGES_DIR)
MASKS_DIR = str(MASKS_DIR)
BASE_OUTPUT = str(OUTPUTS_DIR)

BG_STYLE = DEFAULT_BACKGROUND_STYLE
IMAGE_EXTS = set(IMAGE_EXTENSIONS)


# -- Run ID: auto-increment so we never overwrite previous results --------------

def get_next_run_id():
    """Finds the highest existing run number and returns next one."""
    existing = glob.glob(os.path.join(BASE_OUTPUT, "results_*.csv"))
    if not existing:
        return 1
    nums = []
    for f in existing:
        try:
            nums.append(int(os.path.basename(f).replace("results_", "").replace(".csv", "")))
        except ValueError:
            pass
    return max(nums) + 1 if nums else 1


def parse_args():
    parser = argparse.ArgumentParser(description="CV2 Project Batch Pipeline")
    parser.add_argument(
        "--run-id", type=int, default=None,
        help="Run ID for versioned outputs. Auto-increments if not set."
    )
    return parser.parse_args()


# -- Helpers -------------------------------------------------------------------

def get_image_paths():
    """Returns sorted list of all product image paths."""
    paths = []
    for fname in sorted(os.listdir(RAW_IMAGES_DIR)):
        # Skip test.jpg (that's our module test file)
        if fname == TEST_IMAGE_NAME:
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTS:
            paths.append(os.path.join(RAW_IMAGES_DIR, fname))
    return paths


def save_mask(mask, image_rgb, stem):
    """Save both binary mask AND colored product."""
    # 1. Binary mask for deterministic compositing and evaluation.
    mask_path = os.path.join(MASKS_DIR, f"{stem}_mask.png")
    cv2.imwrite(mask_path, (mask * 255).astype(np.uint8))

    # 2. Colored product (what you actually see - product isolated from background)
    product_path = os.path.join(MASKS_DIR, f"{stem}_product.png")
    colorized_product = image_rgb.copy()
    colorized_product[~mask] = [255, 255, 255]  # White background for non-product
    colorized_product_bgr = cv2.cvtColor(colorized_product, cv2.COLOR_RGB2BGR)
    cv2.imwrite(product_path, colorized_product_bgr)

    return mask_path, product_path


def _lpips_tensor(image_rgb, device):
    image = cv2.resize(image_rgb, (LPIPS_IMAGE_SIZE, LPIPS_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
    tensor = tensor / LPIPS_NORMALIZATION_SCALE - LPIPS_NORMALIZATION_OFFSET
    return tensor.to(device)


def compute_lpips_score(reference_rgb, composite_rgb):
    """
    LPIPS perceptual distance. Lower is more similar to the original photo.
    This is a report metric, separate from SDCS shadow consistency.
    """
    global _lpips_model
    try:
        import lpips

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if _lpips_model is None:
            print("  LPIPS model: loading AlexNet backbone.")
            with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
                _lpips_model = lpips.LPIPS(net=LPIPS_BACKBONE).to(device)
            _lpips_model.eval()

        ref = _lpips_tensor(reference_rgb, device)
        comp = _lpips_tensor(composite_rgb, device)
        with torch.no_grad():
            score = _lpips_model(ref, comp)
        return float(score.item())
    except Exception as e:
        print(f"\n  [WARN] LPIPS failed: {e}")
        return None


def _quiet_call(func, *args, **kwargs):
    with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
        return func(*args, **kwargs)


def _format_score(value):
    return "N/A" if value is None else f"{value:.4f}" if isinstance(value, float) else str(value)


def process_single_image(image_path, run_output_dir, index, total):
    """
    Runs the full pipeline on one image.
    Returns a dict with all angles, scores, timing info, and output paths.
    """
    stem   = os.path.splitext(os.path.basename(image_path))[0]
    result = {"image": stem, "status": "ok"}

    print(f"\nImage {index}/{total}: {stem}")

    # -- Step 1: Segmentation -------------------------------------------------
    try:
        print("  1. Segmenting product with SAM...")
        mask, image_rgb = _quiet_call(segment_product, image_path)
        save_mask(mask, image_rgb, stem)  # Now saves both binary mask AND colored product
        print("     Mask saved.")
    except Exception as e:
        print(f"\n  [ERROR] Segmentation failed for {stem}: {e}")
        result["status"] = f"segmentation_failed"
        return result

    # -- Step 2A: Sobel light estimation --------------------------------------
    print("  2. Estimating lighting angles...")
    t0 = time.time()
    theta_sobel = estimate_light_direction_sobel(image_rgb, mask)
    sobel_time  = time.time() - t0

    result["theta_sobel"]  = round(theta_sobel, 2)
    result["sobel_time_s"] = round(sobel_time, 4)

    # -- Step 2B: DL light estimation -----------------------------------------
    theta_dl = None
    try:
        t0 = time.time()
        theta_dl = _quiet_call(estimate_light_direction_dl, image_rgb, mask)
        dl_time  = time.time() - t0
        result["theta_dl"]    = round(theta_dl, 2)
        result["dl_time_s"]   = round(dl_time, 4)
    except Exception as e:
        print(f"\n  [WARN] DL estimation failed for {stem}: {e}")
        result["theta_dl"]  = None
        result["dl_time_s"] = None
    print(
        f"     Sobel: {result['theta_sobel']} deg ({result['sobel_time_s']}s) | "
        f"DL: {_format_score(result['theta_dl'])} deg ({_format_score(result['dl_time_s'])}s)"
    )

    # -- Step 3: Generate 3 composite versions ---------------------------------
    print("  3. Compositing background and synthetic shadow variants: naive, sobel, dl.")

    configs = [
        ("naive",  None,        "Naive (no lighting hint)"),
        ("sobel",  theta_sobel, "Sobel-conditioned"),
        ("dl",     theta_dl,    "DL-conditioned"),
    ]

    composites = {}
    for key, theta, desc in configs:
        if key == "dl" and theta_dl is None:
            composites[key] = None
            result[f"sdcs_{key}"] = None
            result[f"output_{key}"] = None
            result[f"prompt_{key}"] = None
            result[f"light_direction_{key}"] = None
            result[f"shadow_direction_{key}"] = None
            result[f"shadow_angle_{key}"] = None
            result[f"shadow_dx_{key}"] = None
            result[f"shadow_dy_{key}"] = None
            continue

        out_path = os.path.join(run_output_dir, f"{stem}_{key}.png")
        prompt, negative_prompt, light_direction, shadow_direction = build_generation_prompt(theta=theta, bg_style=BG_STYLE)
        shadow_dx, shadow_dy, shadow_angle = shadow_vector_from_light_angle(theta, image_rgb.shape)
        result["negative_prompt"] = negative_prompt
        result[f"prompt_{key}"] = prompt
        result[f"light_direction_{key}"] = light_direction
        result[f"shadow_direction_{key}"] = shadow_direction
        result[f"shadow_angle_{key}"] = round(shadow_angle, 2) if shadow_angle is not None else None
        result[f"shadow_dx_{key}"] = shadow_dx
        result[f"shadow_dy_{key}"] = shadow_dy
        try:
            composite_pil = _quiet_call(
                generate_background,
                image_rgb,
                mask,
                theta=theta,
                bg_style=BG_STYLE,
            )
            composite_pil.save(out_path)
            composites[key]           = np.array(composite_pil)
            result[f"output_{key}"]   = out_path
        except Exception as e:
            print(f"\n  [ERROR] Generation ({key}) failed for {stem}: {e}")
            composites[key]           = None
            result[f"sdcs_{key}"]     = None
            result[f"output_{key}"]   = None

    # -- Step 4: SDCS evaluation -----------------------------------------------
    print("  4. Evaluating SDCS and LPIPS.")
    # SDCS is an evaluation reference, not the generation-conditioning input.
    # The naive image is unconditioned, but we still score whether its shadow
    # agrees with the product's estimated foreground lighting.
    theta_by_variant = {
        "naive": theta_sobel,
        "sobel": theta_sobel,
        "dl": theta_dl,
    }
    for key in GENERATED_VARIANTS:
        arr = composites.get(key)
        if arr is None:
            result[f"sdcs_{key}"] = None
            result[f"lpips_{key}"] = None
            continue
        score = compute_sdcs(arr, mask, theta_by_variant[key])
        result[f"sdcs_{key}"] = round(score, 4) if score is not None else None
        lpips_score = compute_lpips_score(image_rgb, arr)
        result[f"lpips_{key}"] = round(lpips_score, 4) if lpips_score is not None else None

    print(
        "     SDCS  "
        f"naive={_format_score(result.get('sdcs_naive'))} | "
        f"sobel={_format_score(result.get('sdcs_sobel'))} | "
        f"dl={_format_score(result.get('sdcs_dl'))}"
    )
    print(
        "     LPIPS "
        f"naive={_format_score(result.get('lpips_naive'))} | "
        f"sobel={_format_score(result.get('lpips_sobel'))} | "
        f"dl={_format_score(result.get('lpips_dl'))}"
    )
    print("     Status: complete.")

    return result


# -- Main entry point ----------------------------------------------------------

def run_batch(run_id):
    # Create output directories
    os.makedirs(MASKS_DIR, exist_ok=True)
    os.makedirs(BASE_OUTPUT, exist_ok=True)

    run_output_dir = os.path.join(BASE_OUTPUT, f"run_{run_id:03d}")
    results_csv    = os.path.join(BASE_OUTPUT, f"results_{run_id:03d}.csv")
    os.makedirs(run_output_dir, exist_ok=True)

    print("")
    print("=" * 72)
    print(f"Run {run_id:03d}: illumination-aware product photography batch")
    print(f"Images:  {RAW_IMAGES_DIR}/")
    print(f"Outputs: {run_output_dir}/")
    print(f"CSV:     {results_csv}")
    print("=" * 72)

    image_paths = get_image_paths()
    if not image_paths:
        print(f"ERROR: No images found in {RAW_IMAGES_DIR}/")
        print("Add product photos there and try again.")
        sys.exit(1)

    print(f"Found {len(image_paths)} images. Processing starts now.")
    all_results = []

    for index, image_path in enumerate(image_paths, start=1):
        result = process_single_image(image_path, run_output_dir, index, len(image_paths))
        all_results.append(result)

        # Save CSV after EVERY image - crash-safe
        _save_csv(all_results, results_csv)
        print(f"  Progress saved to {results_csv}.")

    print("")
    print("=" * 72)
    print(f"Done. Processed {len(all_results)} images.")
    print(f"Results CSV: {results_csv}")
    print(f"Generated images: {run_output_dir}/")
    _print_summary(all_results)


def _save_csv(results, path):
    if not results:
        return
    preferred = [
        "image", "status",
        "theta_sobel", "theta_dl", "sobel_time_s", "dl_time_s",
        "sdcs_naive", "sdcs_sobel", "sdcs_dl",
        "lpips_naive", "lpips_sobel", "lpips_dl",
        "light_direction_naive", "light_direction_sobel", "light_direction_dl",
        "shadow_direction_naive", "shadow_direction_sobel", "shadow_direction_dl",
        "shadow_angle_naive", "shadow_angle_sobel", "shadow_angle_dl",
        "shadow_dx_naive", "shadow_dx_sobel", "shadow_dx_dl",
        "shadow_dy_naive", "shadow_dy_sobel", "shadow_dy_dl",
        "negative_prompt",
        "prompt_naive", "prompt_sobel", "prompt_dl",
        "output_naive", "output_sobel", "output_dl",
    ]
    seen = set()
    fieldnames = []
    for key in preferred:
        if any(key in r for r in results):
            fieldnames.append(key)
            seen.add(key)
    for result in results:
        for key in result.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def _print_summary(results):
    def avg(key):
        vals = [r[key] for r in results if isinstance(r.get(key), (int, float))]
        return f"{sum(vals)/len(vals):.4f}" if vals else "N/A"

    print("")
    print("Average Metrics")
    print("-" * 72)
    print(f"{'Method':<10} {'SDCS':>10} {'LPIPS':>10} {'Notes'}")
    print(f"{'Naive':<10} {avg('sdcs_naive'):>10} {avg('lpips_naive'):>10} {'ambient contact shadow'}")
    print(f"{'Sobel':<10} {avg('sdcs_sobel'):>10} {avg('lpips_sobel'):>10} {'proposed method'}")
    print(f"{'DL':<10} {avg('sdcs_dl'):>10} {avg('lpips_dl'):>10} {'DPT baseline'}")
    print("")
    print("Average Light Estimation Time")
    print("-" * 72)
    print(f"Sobel: {avg('sobel_time_s')} sec/image")
    print(f"DL:    {avg('dl_time_s')} sec/image")
    print("-" * 72)


if __name__ == "__main__":
    args   = parse_args()
    run_id = args.run_id if args.run_id is not None else get_next_run_id()
    run_batch(run_id)
