import cv2
import numpy as np
import torch
import time
from PIL import Image
import os
import sys
import shutil
import logging
import warnings
from io import StringIO
import contextlib

from constants import DPT_CACHE_DIRNAME, DPT_MODEL_ID, DRIVE_PROJECT_DIR, LOCAL_MODELS_CACHE_DIR

# Suppress all warnings
warnings.filterwarnings("ignore")

# Suppress verbose HuggingFace + transformers logging to reduce duplicate output
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("diffusers").setLevel(logging.ERROR)
logging.getLogger("transformers.feature_extraction_utils").setLevel(logging.ERROR)

# Disable tqdm progress bars (they print 1000s of lines in Jupyter)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TQDM_DISABLE"] = "1"

def configure_hf_cache():
    """
    Keep HuggingFace caches in Google Drive on Colab, falling back to local cache
    only when Drive is not mounted. Notebook env vars take precedence.
    """
    drive_base = os.environ.get(
        "CV2_DRIVE_WEIGHTS",
        DRIVE_PROJECT_DIR,
    )
    if os.path.exists(drive_base):
        hf_home = os.environ.get("HF_HOME", os.path.join(drive_base, "hf_cache"))
    else:
        hf_home = os.environ.get("HF_HOME", str(LOCAL_MODELS_CACHE_DIR))

    hub_cache = os.environ.get("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.makedirs(hub_cache, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = os.environ.get("TRANSFORMERS_CACHE", hub_cache)
    os.environ["DIFFUSERS_CACHE"] = os.environ.get("DIFFUSERS_CACHE", hub_cache)
    os.environ["XDG_CACHE_HOME"] = os.environ.get("XDG_CACHE_HOME", os.path.join(hf_home, "xdg"))
    return hf_home, hub_cache

HF_HOME, HF_HUB_CACHE = configure_hf_cache()

# Global variable to hold the DL model so we don't load it multiple times
_depth_estimator = None

def _cache_contains_model(model_cache_name):
    for base in [os.environ.get("HF_HUB_CACHE"), os.environ.get("TRANSFORMERS_CACHE"), os.environ.get("HF_HOME")]:
        if base and os.path.exists(os.path.join(base, model_cache_name)):
            return True
    return False

def estimate_light_direction_sobel(image_rgb, mask):
    """
    Module 2A: The Sobel-Gradient Method (Your original proposal)
    Calculates the direction of the light source using pure computer vision math.
    It looks for the sharpest transitions from bright to dark.
    
    Args:
        image_rgb (np.array): Original image.
        mask (np.array): Boolean mask of the product.
    Returns:
        float: Angle of the light source in degrees (-180 to 180).
    """
    # 1. Isolate the product using the mask
    masked = image_rgb.copy()
    masked[~mask] = 0
    gray = cv2.cvtColor(masked, cv2.COLOR_RGB2GRAY).astype(float)

    # 2. Apply Sobel operators (Detects horizontal and vertical brightness changes)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # 3. Calculate the magnitude (strength) and angle of the changes
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    angle = np.arctan2(sobel_y, sobel_x) * 180 / np.pi

    # 4. Filter only the strong edges inside the mask
    valid = mask & (magnitude > magnitude.mean())
    angles_valid = angle[valid]
    weights_valid = magnitude[valid]

    if len(angles_valid) == 0:
        return 0.0

    # 5. Build a histogram to find the most common light angle
    hist, bin_edges = np.histogram(
        angles_valid, bins=36, range=(-180, 180), weights=weights_valid
    )

    dominant_bin = np.argmax(hist)
    theta_fg = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2

    return theta_fg

def estimate_light_direction_dl(image_rgb, mask):
    """
    Module 2B: The Deep Learning Baseline (TA's Requirement)
    Uses a 340 Million parameter Vision Transformer (DPT-Large) to estimate the 3D depth
    of the product, then combines that 3D structure with pixel brightness to find the light source.

    Args:
        image_rgb (np.array): Original image.
        mask (np.array): Boolean mask of the product.
    Returns:
        float: Angle of the light source in degrees (-180 to 180).
    """
    global _depth_estimator

    # 1. Lazy load the Deep Learning model (only loads the first time this function is called)
    if _depth_estimator is None:
        from transformers import pipeline
        device = 0 if torch.cuda.is_available() else -1
        configure_hf_cache()
        cache_dir = os.environ["HF_HUB_CACHE"]
        model_cache_name = DPT_CACHE_DIRNAME
        cache_status = "Drive/local cache hit" if _cache_contains_model(model_cache_name) else "cache miss, will download once"
        print(f"Loading DPT-Large depth estimator ({cache_status}).")
        print(f"  HF_HOME: {os.environ['HF_HOME']}")
        print(f"  HF_HUB_CACHE: {os.environ['HF_HUB_CACHE']}")

        # SUPPRESS STDOUT during model loading (JAX/Flax + transformers write directly to STDOUT)
        with contextlib.redirect_stdout(StringIO()):
            _depth_estimator = pipeline(
                "depth-estimation",
                model=DPT_MODEL_ID,
                device=device,
                model_kwargs={"cache_dir": cache_dir},
            )

    # 2. Convert numpy array to PIL Image for the HuggingFace pipeline
    image_pil = Image.fromarray(image_rgb)
    
    # 3. Run the Deep Learning inference to get the 3D depth map
    depth_output = _depth_estimator(image_pil)
    depth_map = np.array(depth_output["depth"], dtype=float)
    
    # Normalize depth map to [0, 1]
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-5)
    depth_map[~mask] = 0 # Ignore background
    
    # 4. Get Grayscale intensity
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(float)
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-5)
    
    # 5. DL Logic: Light hits the most protruding (highest depth) and brightest areas.
    # We multiply depth by brightness to find the 3D peaks that are catching the light.
    combined_activation = gray * depth_map
    
    # Find the geometric center of the product
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0: 
        return 0.0
    center_y, center_x = np.mean(y_indices), np.mean(x_indices)
    
    # Find the center of mass of the "light catching" areas (top 5% brightest/closest points)
    valid_activations = combined_activation[mask]
    threshold = np.percentile(valid_activations, 95) 
    
    bright_y, bright_x = np.where((combined_activation >= threshold) & mask)
    if len(bright_y) == 0: 
        return 0.0
    
    light_y, light_x = np.mean(bright_y), np.mean(bright_x)
    
    # Calculate angle from the geometric center of the object pointing towards the light area
    dy = light_y - center_y
    dx = light_x - center_x
    
    angle = np.arctan2(dy, dx) * 180 / np.pi
    return angle

def benchmark_estimators(image_path, mask_path):
    """
    Utility function to run both estimators on an image and compare their speed and output.
    """
    image_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    
    # Load mask (assume it's saved as 0/255 from our segmentation script)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = mask_img > 127
    
    print("-" * 40)
    print("Testing Sobel Method (CPU Math)...")
    start_time = time.time()
    angle_sobel = estimate_light_direction_sobel(image_rgb, mask)
    sobel_time = time.time() - start_time
    print(f"theta_sobel : {angle_sobel:.1f} degrees")
    print(f"Time taken: {sobel_time:.4f} seconds")
    
    print("-" * 40)
    print("Testing Deep Learning Baseline (DPT-Large)...")
    start_time = time.time()
    angle_dl = estimate_light_direction_dl(image_rgb, mask)
    dl_time = time.time() - start_time
    print(f"theta_dl (computed DL baseline): {angle_dl:.1f} degrees")
    print(f"Time taken: {dl_time:.4f} seconds")
    print("-" * 40)
    
    return angle_sobel, angle_dl

# --- TEST BLOCK FOR COLAB ---
# Run via: !python core/2_illumination.py
if __name__ == "__main__":
    import os
    test_image_path = "data/raw_images/test.jpg"
    test_mask_path = "data/masks/test_mask.png"
    
    if not os.path.exists(test_image_path) or not os.path.exists(test_mask_path):
        print("ACTION REQUIRED: Make sure you have run '1_segmentation.py' first so that test.jpg and test_mask.png exist!")
    else:
        benchmark_estimators(test_image_path, test_mask_path)
