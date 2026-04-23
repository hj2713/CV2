import os
import sys
from pathlib import Path
import cv2
import numpy as np
import urllib.request
import torch
from segment_anything import sam_model_registry, SamPredictor

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from constants import (
    MASKS_DIR,
    RAW_IMAGES_DIR,
    SAM_MODEL_TYPE,
    SAM_WEIGHTS_FILENAME,
    SAM_WEIGHTS_URL,
    TEST_IMAGE_NAME,
    TEST_MASK_NAME,
    TEST_PRODUCT_NAME,
)

_sam_predictor_cache = {}

def download_sam_weights(model_type=SAM_MODEL_TYPE):
    """
    Downloads the SAM model weights if they are not already present in the current directory.
    This saves you from having to run !wget manually in Colab.
    """
    urls = {
        SAM_MODEL_TYPE: SAM_WEIGHTS_URL,
        # We can add other models like vit_b if vit_h is too heavy, but we'll stick to vit_h as proposed.
    }
    
    if model_type not in urls:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    env_checkpoint = os.environ.get("SAM_CHECKPOINT_PATH")
    if env_checkpoint and os.path.exists(env_checkpoint):
        print(f"Found SAM checkpoint from SAM_CHECKPOINT_PATH: {env_checkpoint}")
        return env_checkpoint

    url = urls[model_type]
    filename = SAM_WEIGHTS_FILENAME if model_type == SAM_MODEL_TYPE else url.split("/")[-1]
    
    if not os.path.exists(filename):
        print(f"Downloading {filename} (this is ~2.4GB and might take a minute)...")
        urllib.request.urlretrieve(url, filename)
        print(f"Downloaded {filename} successfully!")
    else:
        print(f"Found {filename} locally. No need to download.")
        
    return filename

def get_sam_predictor(model_type=SAM_MODEL_TYPE):
    """
    Loads SAM once per process and reuses it across all images in a batch run.
    """
    checkpoint_path = download_sam_weights(model_type)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_key = (model_type, checkpoint_path, device)

    if cache_key in _sam_predictor_cache:
        print(f"Using cached SAM model on {device}.")
        return _sam_predictor_cache[cache_key], device

    print(f"Using device: {device}")
    print("Loading SAM model into memory...")
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    _sam_predictor_cache[cache_key] = predictor
    return predictor, device

def segment_product(image_path, box=None, model_type=SAM_MODEL_TYPE):
    """
    Loads an image, runs Meta's SAM (Segment Anything Model), and extracts the product mask.
    
    Args:
        image_path (str): Path to the messy product photo.
        box (list or np.array): [x_min, y_min, x_max, y_max]. If None, we assume the product is centered.
        model_type (str): Type of SAM model to use.
        
    Returns:
        mask (np.array): A boolean 2D array where True indicates the product pixels.
        image_rgb (np.array): The original image in RGB format.
    """
    print(f"Processing image: {image_path}")
    
    # 1. Load or reuse SAM model
    predictor, _ = get_sam_predictor(model_type)
    
    # 2. Load Image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not load image at {image_path}")
    
    # OpenCV loads in BGR format, but SAM expects RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    predictor.set_image(image_rgb)
    
    # 3. Determine Bounding Box
    if box is None:
        # If the user didn't provide a box, assume the product takes up the central 80% of the image.
        # This works well for most casual product photos.
        h, w, _ = image_rgb.shape
        x_min = int(w * 0.1)
        y_min = int(h * 0.1)
        x_max = int(w * 0.9)
        y_max = int(h * 0.9)
        input_box = np.array([x_min, y_min, x_max, y_max])
    else:
        input_box = np.array(box)
        
    # 4. Predict the Mask
    print("Running segmentation prediction...")
    masks, scores, logits = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False # We only want the single best mask
    )
    
    # The output is a boolean array (True/False). 
    # masks[0] is our primary segmented product.
    best_mask = masks[0]
    
    print("Segmentation complete!")
    return best_mask, image_rgb

# --- TEST BLOCK FOR COLAB ---
# You can run this block in Colab to test it via: !python core/1_segmentation.py
if __name__ == "__main__":
    # For testing, place an image named 'test.jpg' inside data/raw_images/
    test_image_path = str(RAW_IMAGES_DIR / TEST_IMAGE_NAME)
    test_mask_path = str(MASKS_DIR / TEST_MASK_NAME)
    test_product_path = str(MASKS_DIR / TEST_PRODUCT_NAME)

    if not os.path.exists(test_image_path):
        print(f"ACTION REQUIRED: Upload a product photo and name it 'test.jpg' inside {test_image_path} to run this test!")
    else:
        print("Starting SAM Segmentation Test...")
        # Run segmentation
        mask, rgb_img = segment_product(test_image_path)

        # 1. Save binary mask for compositing and evaluation.
        mask_image = (mask * 255).astype(np.uint8)
        cv2.imwrite(test_mask_path, mask_image)
        print(f"Binary mask saved: {test_mask_path}")

        # 2. Save COLORED product (product isolated from background)
        #    This is what you actually see - the real product!
        colorized_product = rgb_img.copy()
        # Make background transparent/white (outside mask)
        colorized_product[~mask] = [255, 255, 255]  # White background for non-product areas

        # Convert RGB to BGR for cv2.imwrite
        colorized_product_bgr = cv2.cvtColor(colorized_product, cv2.COLOR_RGB2BGR)
        cv2.imwrite(test_product_path, colorized_product_bgr)
        print(f"Colored product saved: {test_product_path}")

        print(f"\nResults:")
        print(f"  - Binary mask (white/black): {test_mask_path}")
        print(f"    Used by the compositor to place the product and synthesize shadows")
        print(f"  - Colored product: {test_product_path}")
        print(f"    Actual product segmented from background")
