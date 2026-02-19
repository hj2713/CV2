# HW2: Automatic Feature Matching

**Author:** Himanshu Jhawar (hj2713)  
**Course:** COMS4732W Computer Vision 2  
**Based on:** "Multi-Image Matching using Multi-Scale Oriented Patches" (Brown et al.)

---

## Overview

This project implements an automatic feature matching pipeline between images:

1. **Harris Corner Detection** — using provided `harris.py` (single scale)
2. **Non-Maximal Suppression (NMS)** — local max filtering with `scipy.ndimage.maximum_filter`
3. **Feature Descriptor Extraction** — 40×40 RGB patches → 8×8×3 (192-dim), bias/gain normalized
4. **Feature Matching** — Lowe's NNDR ratio test with L2 (Euclidean) distance
5. **Extra Credit: Image Stitching** — RANSAC homography estimation + warping + blending (3 images)

---

## Project Structure

```
cv2/
├── main.ipynb                  # Main notebook with all steps & visualizations
├── harris.py                   # Provided Harris corner detector
├── feature_matching.py         # Plotting utilities for Step 4
├── create_panorama.py          # Extra credit panorama stitching code
├── images/                     # Input images
│   ├── imgA1.jpg, imgA2.jpg        # Scene 1: HW staff example pair
│   ├── imgD1.jpg, imgD2.jpg        # Scene 2: Own photo pair
│   └── imgC1.jpg, imgC2.jpg,       # Scene 3: 3 images for panorama
│       imgC3.jpg
├── results/
│   ├── image_present_HW/           # Scene 1 results (staff example)
│   │   ├── step0_originals.png
│   │   ├── step1_harris.png
│   │   ├── step2_nms.png
│   │   ├── step4_nndr_histogram.png
│   │   ├── step4_matches.png
│   │   └── step4_top5_matches.png
│   ├── running_on_my_image/        # Scene 2 results (own images)
│   │   ├── step0_originals.png
│   │   ├── step1_harris.png
│   │   ├── step2_nms.png
│   │   ├── step4_nndr_histogram.png
│   │   ├── step4_matches.png
│   │   └── step4_top5_matches.png
│   └── panoramic_3_images_combined/ # Extra credit panorama
│       ├── extra_credit_panorama.png
│       └── step4_matches.png
├── index.html                  # Results webpage
└── README.md                   # This file
```

---

## Libraries Used

| Library        | Purpose                                                                              |
| -------------- | ------------------------------------------------------------------------------------ |
| `numpy`        | Array operations                                                                     |
| `matplotlib`   | Plotting & visualization                                                             |
| `scikit-image` | `resize`, `corner_harris`, `peak_local_max`, `ransac`, `ProjectiveTransform`, `warp` |
| `scipy`        | `maximum_filter` (NMS), `cdist` (L2 distance)                                        |
| `Pillow`       | Image loading with EXIF orientation fix                                              |

---

## How to Run

### Prerequisites

```bash
python -m venv venv
source venv/bin/activate
pip install numpy matplotlib scikit-image scipy Pillow jupyter
```

### Run the Notebook

```bash
jupyter notebook main.ipynb
```

Run all cells sequentially (Kernel → Restart & Run All).

---

## Hyperparameters

| Parameter                   | Value           |
| --------------------------- | --------------- |
| Harris `edge_discard`       | 20 px           |
| NMS window size             | 20 px           |
| Descriptor patch size       | 40×40 → 8×8     |
| Descriptor dimensions       | 192 (8×8×3 RGB) |
| Similarity metric           | L2 (Euclidean)  |
| NNDR threshold              | 0.5             |


---

## Results

Results for **2 scenes + extra credit panorama** are displayed in `index.html`:

1. **Scene 1 (HW Staff Example):** `results/image_present_HW/` — Steps 0–4 on staff-provided images
2. **Scene 2 (Own Images):** `results/running_on_my_image/` — Steps 0–4 on personal photos
3. **Extra Credit (Panorama):** `results/panoramic_3_images_combined/` — 3-image stitching with RANSAC
