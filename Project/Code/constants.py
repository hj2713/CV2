"""Shared configuration for the CV2 illumination-aware photography project."""

from pathlib import Path


# Repository configuration used by the Colab notebook.
REPOSITORY_URL = "https://github.com/hj2713/CV2.git"
COLAB_REPOSITORY_DIR = "/content/cv2_repo"
SPARSE_CHECKOUT_PATH = "Project"
COLAB_CODE_DIR = "/content/cv2_repo/Project/Code"
COLAB_DRIVE_MOUNT_POINT = "/content/drive"


# Data directories, relative to Code/.
DATA_DIR = Path("data")
RAW_IMAGES_DIR = DATA_DIR / "raw_images"
MASKS_DIR = DATA_DIR / "masks"
OUTPUTS_DIR = DATA_DIR / "outputs"
LOCAL_MODELS_CACHE_DIR = DATA_DIR / "models_cache"


# Image and run configuration.
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
TEST_IMAGE_NAME = "juice.jpg"
TEST_MASK_NAME = "juice_mask.png"
TEST_PRODUCT_NAME = "juice_product.png"
TEST_GENERATED_NAME = "juice_generated.png"
GRADIO_TEMP_INPUT_NAME = "_tmp_input.jpg"
DEFAULT_BACKGROUND_STYLE = "marble surface table"
BACKGROUND_STYLE_OPTIONS = (
    "marble surface table",
    "wooden table",
    "white fabric background",
    "concrete floor",
    "dark slate surface",
    "outdoor grass",
)
GENERATED_VARIANTS = ("naive", "sobel", "dl")
MAX_VISUALIZATION_IMAGES = 6
OUTPUTS_PER_IMAGE = 3


# Google Drive and model cache configuration.
DRIVE_PROJECT_DIR = "/content/drive/MyDrive/Study/Second Sem/CV2/Project"
HF_CACHE_DIRNAME = "hf_cache"
HF_HUB_CACHE_DIRNAME = "hub"
HF_XDG_CACHE_DIRNAME = "xdg"


# Model identifiers and model files.
SAM_MODEL_TYPE = "vit_h"
SAM_WEIGHTS_FILENAME = "sam_vit_h_4b8939.pth"
SAM_WEIGHTS_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
DPT_MODEL_ID = "Intel/dpt-large"
DPT_CACHE_DIRNAME = "models--Intel--dpt-large"
STABLE_DIFFUSION_MODEL_ID = "sd2-community/stable-diffusion-2-inpainting"
STABLE_DIFFUSION_CACHE_DIRNAME = "models--sd2-community--stable-diffusion-2-inpainting"


# Deterministic compositing and evaluation settings.
SHADOW_OFFSET_FRACTION = 0.08
SHADOW_BLUR_FRACTION = 0.045
SHADOW_OPACITY = 0.48
AMBIENT_SHADOW_OPACITY = 0.22
BACKGROUND_NOISE_STRENGTH = 5.0
LPIPS_IMAGE_SIZE = 256
LPIPS_BACKBONE = "alex"
LPIPS_NORMALIZATION_SCALE = 127.5
LPIPS_NORMALIZATION_OFFSET = 1.0


# Legacy diffusion settings kept only for reference with older experiments.
GENERATION_IMAGE_SIZE = 512
GENERATION_STEPS = 30
GENERATION_GUIDANCE_SCALE = 7.5


# GitHub push safety.
GITHUB_FILE_SIZE_LIMIT_BYTES = 100 * 1024 * 1024
GIT_IDENTITY_NAME = "Himanshu Jhawar"
GIT_IDENTITY_EMAIL = "hj2713@columbia.edu"
