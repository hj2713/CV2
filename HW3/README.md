# HW3: Simple Structure from Motion (SfM)

**Himanshu Jhawar (hj2713)**
**COMS4732W Computer Vision 2**

## Overview

This project implements a simple Structure from Motion pipeline that estimates correspondences, camera position/motion, and 3D scene points from a pair of images.

### Pipeline Steps:

1. **Camera Intrinsics**: Computes the intrinsic matrix `K` from the provided physical camera specifications.
2. **Feature Detection**: Detects SIFT features in both images to handle rotation and scale changes.
3. **Feature Matching**: Matches SIFT descriptors using euclidean distance with Lowe's Nearest Neighbor Distance Ratio (NNDR) test.
4. **Pose Estimation**: Uses the 8-point algorithm inside a RANSAC loop to robustly estimate the Essential Matrix `E`, which is then decomposed to find the Rotation `R` and translation `t`.
5. **Triangulation**: Reconstructs the 3D points from the verified inliers.

## How to Run

1. **Environment Setup**:
   Ensure you have the required dependencies installed. You can install them in a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Execute the Pipeline**:
   The entire pipeline can be run using the pre-configured parameters by executing the main script:

   ```bash
   python main.py
   ```

   This will create a timestamped folder inside the `outputs/` directory containing all the visualizations, numerical results, and the 3D scene data (`step5_scene_data.npz`).

3. **Visualize 3D Reconstruction**:
   To view the triangulated sparse point cloud in 3D:
   ```bash
   python visualize_viser.py outputs/<your_specific_run_folder>/step5_scene_data.npz
   ```
   _Replace `<your_specific_run_folder>` with the specific output folder generated._

## Configuration Parameters Used

These are the final parameters chosen for the submitted results:

- **Random Seed**: `42`
- **SIFT Max Features**: `1200` (Detected exactly `1197` valid SIFT features in both images)
- **NNDR Threshold**: `0.8`
- **RANSAC Epsilon (Sampson distance threshold)**: `0.01`
- **RANSAC Iterations**: `2000`

### Calculated Camera Intrinsics `K`:

```
[[ 887.47,    0.00,  640.00],
 [   0.00,  887.47,  480.00],
 [   0.00,    0.00,    1.00]]
```
