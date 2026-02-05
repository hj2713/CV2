# Prokudin-Gorskii Collection Alignment

This project implements an automated image alignment algorithm to reconstruct color images from the digitized glass plate negatives of the Prokudin-Gorskii collection. The program aligns three separate color channels (Blue, Green, Red) to produce a single high-quality color image.

## Project Structure

- `CV2_HW1.ipynb`: The main Jupyter Notebook containing the alignment algorithm and processing pipeline.
- `generate_report.py`: A Python script to generate a visual HTML report of the results.
- `input_images/`: Directory containing the raw glass plate negative images (.jpg for single-scale, .tif for multi-scale).
- `output_images/`: Directory where the aligned color images are saved.
- `report.html`: The generated HTML report displaying the aligned images and processing details.

## Setup

1.  **Clone the repository**:

    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Dependencies**:
    Ensure you have Python installed. Install the required libraries:

    ```bash
    pip install numpy matplotlib scikit-image jupyter
    ```

3.  **Prepare Data**:
    - Place your input images (JPEG or TIFF) in the `input_images/` directory.
    - Ensure the `output_images/` directory exists (it will be created automatically by the notebook if missing).

## How to Run

### 1. Run the Alignment Code

Open the Jupyter Notebook and execute all cells to process the images:

```bash
jupyter notebook CV2_HW1.ipynb
```

The notebook will:

- Load images from `input_images/`.
- Perform single-scale alignment for small JPEGs.
- Perform multi-scale pyramid alignment for large TIFFs.
- Save the aligned results to `output_images/`.

### 2. Generate the HTML Report

After processing the images, you can generate a styled HTML report to view the results:

```bash
python3 generate_report.py
```

This will create (or update) `report.html` in the project root. Open this file in any web browser to view the aligned collection with offset details.

## Algorithm Details

The core alignment utilizes an image pyramid approach to handle high-resolution images efficiently:

- **Metric**: Sum of Squared Differences (SSD / L2 Norm) is used to measure alignment quality.
- **Pyramid Search**: The algorithm recursively aligns smaller versions of the image to find a coarse offset, then refines it at higher resolutions.
- **Features**: Edges are extracted using Sobel filters to improved alignment robustness against intensity variations between channels.
