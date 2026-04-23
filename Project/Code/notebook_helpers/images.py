"""Image discovery and notebook visualization helpers."""

from __future__ import annotations

import shutil
from pathlib import Path

from constants import IMAGE_EXTENSIONS, MASKS_DIR, RAW_IMAGES_DIR, TEST_IMAGE_NAME


def discover_product_images() -> tuple[list[str], list[str], str]:
    """Discover all product images and prepare test.jpg for module cells."""
    raw_images_dir = Path(RAW_IMAGES_DIR)
    raw_images_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in raw_images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and path.name != TEST_IMAGE_NAME
    )

    if not image_paths:
        supported = ", ".join(IMAGE_EXTENSIONS)
        raise FileNotFoundError(
            f"No product images found in {raw_images_dir}. "
            f"Add one or more supported images ({supported}) and rerun this cell."
        )

    print(f"Image discovery complete: {len(image_paths)} image(s) found in {raw_images_dir}.")
    for index, image_path in enumerate(image_paths, start=1):
        size_kb = image_path.stat().st_size / 1024
        print(f"  {index:02d}. {image_path.name} ({size_kb:.0f} KB)")

    selected_test_image = image_paths[0]
    shutil.copy2(selected_test_image, raw_images_dir / TEST_IMAGE_NAME)

    pipeline_images = [path.name for path in image_paths]
    pipeline_image_paths = [str(path) for path in image_paths]
    test_image_source = selected_test_image.name

    print(f"Module test image: {TEST_IMAGE_NAME} copied from {test_image_source}.")
    print(f"Pipeline image count: {len(pipeline_images)}")
    return pipeline_images, pipeline_image_paths, test_image_source


def visualize_sam_segmentation() -> None:
    """Display the test input and SAM product cutout."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    test_image_path = Path(RAW_IMAGES_DIR) / TEST_IMAGE_NAME
    test_product_path = Path(MASKS_DIR) / "test_product.png"

    if not test_image_path.exists() or not test_product_path.exists():
        raise FileNotFoundError("Missing test image or segmentation output. Run Cell 2.1 first.")

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    original_image = mpimg.imread(test_image_path)
    segmented_product = mpimg.imread(test_product_path)

    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(segmented_product)
    axes[1].set_title("SAM Segmentation Result", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

    print(f"Original image shape: {original_image.shape}")
    print(f"Segmented product shape: {segmented_product.shape}")
