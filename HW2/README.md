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
5. **Extra Credit: Image Stitching** — RANSAC homography estimation + warping + blending

---

## Project Structure

```
cv2/
├── main.ipynb              # Main notebook with all steps & visualizations
├── harris.py               # Provided Harris corner detector
├── feature_matching.py     # Plotting utilities for Step 4
├── create_panorama.py      # Extra credit panorama stitching code
├── images/                 # Input images
│   ├── imgA1.jpg, imgA2.jpg    # Scene A (pair)
│   ├── imgC1-3.jpg             # Scene C (3 images for panorama)
│   └── imgD1-3.jpg             # Scene D (3 images for panorama)
├── results/                # Output visualizations
│   ├── step0_originals.png
│   ├── step1_harris.png
│   ├── step2_nms.png
│   ├── step4_nndr_histogram.png
│   ├── step4_matches.png
│   ├── step4_top5_matches.png
│   └── extra_credit_panorama.png
├── index.html              # Results webpage
└── README.md               # This file
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

| Parameter                   | Value                          |
| --------------------------- | ------------------------------ |
| Harris `edge_discard`       | 20 px                          |
| NMS window size             | 20 px                          |
| Descriptor patch size       | 40×40 → 8×8                    |
| Descriptor dimensions       | 192 (8×8×3 RGB)                |
| Similarity metric           | L2 (Euclidean)                 |
| NNDR threshold              | 0.5 (Scene A), 0.7 (Scene C/D) |
| RANSAC `min_samples`        | 4                              |
| RANSAC `residual_threshold` | 4 px                           |
| RANSAC `max_trials`         | 2000                           |

---

## Results

Results for **2 scenes** are displayed in `index.html`. Open it in a browser to see all visualizations.
