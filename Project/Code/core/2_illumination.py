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

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from constants import (
    DPT_CACHE_DIRNAME,
    DPT_MODEL_ID,
    DRIVE_PROJECT_DIR,
    LOCAL_MODELS_CACHE_DIR,
    MASKS_DIR,
    RAW_IMAGES_DIR,
    TEST_IMAGE_NAME,
    TEST_MASK_NAME,
)

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


def _normalize_angle(angle_degrees):
    return ((angle_degrees + 180.0) % 360.0) - 180.0


def _angle_from_vector(dx, dy):
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return None
    return _normalize_angle(np.degrees(np.arctan2(dy, dx)))


def _blend_angles(angle_a, weight_a, angle_b, weight_b):
    angles = []
    weights = []
    if angle_a is not None and weight_a > 0:
        angles.append(np.radians(angle_a))
        weights.append(weight_a)
    if angle_b is not None and weight_b > 0:
        angles.append(np.radians(angle_b))
        weights.append(weight_b)
    if not angles:
        return 0.0

    x = sum(w * np.cos(a) for a, w in zip(angles, weights))
    y = sum(w * np.sin(a) for a, w in zip(angles, weights))
    return _angle_from_vector(x, y) or 0.0


def _interior_mask(mask):
    """Remove object-boundary pixels so the estimator focuses on shading, not silhouette edges."""
    mask_uint8 = mask.astype(np.uint8)
    y_indices, x_indices = np.where(mask)
    if len(y_indices) == 0:
        return mask

    object_size = max(y_indices.max() - y_indices.min() + 1, x_indices.max() - x_indices.min() + 1)
    erosion_size = max(3, int(object_size * 0.035))
    if erosion_size % 2 == 0:
        erosion_size += 1
    kernel = np.ones((erosion_size, erosion_size), dtype=np.uint8)
    eroded = cv2.erode(mask_uint8, kernel, iterations=1).astype(bool)

    # Very thin objects can disappear after erosion; fall back to the original mask.
    return eroded if eroded.sum() > max(50, mask.sum() * 0.15) else mask

def estimate_light_direction_sobel(image_rgb, mask):
    """
    Module 2A: Proposed training-free illumination estimator.

    The first implementation used a magnitude-weighted histogram of all Sobel
    edge orientations. That was too sensitive to product silhouettes and printed
    texture. This version follows the same training-free spirit but estimates
    illumination from low-frequency shading:

    1. erode the mask to remove boundary edges,
    2. blur the grayscale image to suppress texture/albedo detail,
    3. estimate the dark-to-bright direction inside the product,
    4. blend that with a weak Sobel gradient cue from the smoothed shading image.
    
    Args:
        image_rgb (np.array): Original image.
        mask (np.array): Boolean mask of the product.
    Returns:
        float: Angle of the light source in degrees (-180 to 180).
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return 0.0

    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    interior = _interior_mask(mask)

    product_values = gray[mask]
    fill_value = float(np.median(product_values)) if len(product_values) else 0.0
    gray_filled = gray.copy()
    gray_filled[~mask] = fill_value

    y_indices, x_indices = np.where(mask)
    object_size = max(y_indices.max() - y_indices.min() + 1, x_indices.max() - x_indices.min() + 1)
    blur_size = max(7, int(object_size * 0.11))
    if blur_size % 2 == 0:
        blur_size += 1
    shading = cv2.GaussianBlur(gray_filled, (blur_size, blur_size), 0)

    valid_values = shading[interior]
    if len(valid_values) == 0:
        return 0.0

    # Centroid cue: illumination direction tends to point from darker product
    # regions toward brighter product regions after texture is smoothed away.
    low_threshold = np.percentile(valid_values, 25)
    high_threshold = np.percentile(valid_values, 75)
    dark_region = interior & (shading <= low_threshold)
    bright_region = interior & (shading >= high_threshold)

    centroid_angle = None
    centroid_confidence = 0.0
    if dark_region.sum() > 0 and bright_region.sum() > 0:
        dark_y, dark_x = np.where(dark_region)
        bright_y, bright_x = np.where(bright_region)
        dx = float(np.mean(bright_x) - np.mean(dark_x))
        dy = float(np.mean(bright_y) - np.mean(dark_y))
        centroid_angle = _angle_from_vector(dx, dy)
        centroid_confidence = min(1.0, np.hypot(dx, dy) / max(1.0, object_size * 0.25))

    # Sobel cue: use gradients only on the smoothed shading image and only in
    # the mask interior. This keeps the original Sobel idea but suppresses
    # boundary and albedo-texture dominance.
    sobel_x = cv2.Sobel(shading, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(shading, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    valid = interior & (magnitude > np.percentile(magnitude[interior], 65))

    gradient_angle = None
    gradient_confidence = 0.0
    if valid.sum() > 0:
        weights = magnitude[valid]
        vx = float(np.sum(sobel_x[valid] * weights))
        vy = float(np.sum(sobel_y[valid] * weights))
        gradient_angle = _angle_from_vector(vx, vy)
        resultant = np.hypot(vx, vy)
        total = float(np.sum(weights * np.sqrt(sobel_x[valid] ** 2 + sobel_y[valid] ** 2))) + 1e-6
        gradient_confidence = min(1.0, resultant / total)

    # The centroid cue is more stable on textured products; the gradient cue is
    # retained as a secondary signal when the smoothed shading has a coherent
    # direction.
    return _blend_angles(
        centroid_angle,
        max(0.15, centroid_confidence),
        gradient_angle,
        0.35 * gradient_confidence,
    )

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

        # Suppress model-loader internals that otherwise flood notebook output.
        with contextlib.redirect_stdout(StringIO()), contextlib.redirect_stderr(StringIO()):
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
    print(f"theta_sobel (computed, not hardcoded): {angle_sobel:.1f} degrees")
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
    test_image_path = str(RAW_IMAGES_DIR / TEST_IMAGE_NAME)
    test_mask_path = str(MASKS_DIR / TEST_MASK_NAME)
    
    if not os.path.exists(test_image_path) or not os.path.exists(test_mask_path):
        print("ACTION REQUIRED: Make sure you have run '1_segmentation.py' first so that test.jpg and test_mask.png exist!")
    else:
        benchmark_estimators(test_image_path, test_mask_path)
