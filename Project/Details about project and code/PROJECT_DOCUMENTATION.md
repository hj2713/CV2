# CV2 Project Documentation

## Project Summary

This project builds an illumination-aware product photography pipeline for casual product images. The current architecture is no longer text-only diffusion generation. It is a computer-vision compositing pipeline:

1. Segment the product using Segment Anything Model (SAM).
2. Estimate foreground lighting direction with the proposed Sobel-shading method.
3. Estimate a comparison lighting direction with a DPT-Large depth-based deep-learning baseline.
4. Create clean studio backgrounds procedurally.
5. Synthesize a cast shadow from the product mask opposite the estimated light direction.
6. Paste the original product pixels back exactly.
7. Evaluate with SDCS, LPIPS, and timing.

The key project idea is to preserve the product as photographed and adapt the generated/composited scene to match the product's existing lighting.

Primary entry point:

```text
Code/CV2_Pipeline.ipynb
```

Run this notebook in Google Colab. Local execution is only for static checks and editing.

## Original Proposal Context

The submitted proposal described an illumination-aware product photography system:

- Segment a product from a casual photo.
- Estimate the foreground light direction.
- Create a background whose lighting and shadow direction match the product.
- Evaluate physical consistency using a custom SDCS metric.

The proposal also mentioned possible deliverables such as a 30-image benchmark, LPIPS/CLIP-IQA, and a human study. These were proposal examples and goals, not all mandatory final implementation items. The final report should only claim what is actually implemented and measured.

## TA Feedback

The TA's comment was:

```text
This is a well-scoped project, though I do want to note there are many methods which do deep-learning-based illumination estimation (some diffusion-based), and I would like to see how your Sobel-gradient-based method compares to these methods. You might also want to take a look at IC-Light (given background image, change the lighting on the foreground subject), because what you are doing seems to be the reverse of that (given foreground subject, generate/change the lighting of the background). Also, your last table is a bit confusing---is this the table you want to get? Or have you already done this project and thus you have your numbers? (If you have already done parts of the project for another purpose, check edstem pinned post on our double dipping policy.)
```

How the implementation addresses this:

| TA point | Current response |
| --- | --- |
| Compare Sobel to deep-learning illumination methods | Added DPT-Large depth-based DL baseline. It estimates depth, then derives an angle from depth-weighted brightness. |
| Mention IC-Light | Discuss IC-Light as related work. IC-Light relights foreground to match a target background. This project does the reverse: preserve foreground lighting and adapt the background/shadow. |
| Clarify table numbers | Final report must clearly state that proposal numbers were expected values, while final tables are measured experimental results. |
| Avoid double-dipping confusion | State that the implementation and results were produced for this CV2 project. |

## Current Architecture

### Module 1: Segmentation

File:

```text
Code/core/1_segmentation.py
```

Purpose:

- Uses SAM ViT-H to segment the product.
- Saves a binary mask and a white-background product cutout.
- Caches SAM globally during batch runs so it is not reloaded for every image.

Main function:

```python
segment_product(image_path, box=None, model_type="vit_h")
```

Output:

- `mask`: boolean product mask.
- `image_rgb`: original RGB image.

### Module 2A: Proposed Sobel-Shading Illumination Estimator

File:

```text
Code/core/2_illumination.py
```

Main function:

```python
estimate_light_direction_sobel(image_rgb, mask)
```

Method:

- Convert the product region to grayscale.
- Erode the mask so silhouette edges do not dominate the estimate.
- Blur the grayscale product to suppress texture, labels, and high-frequency reflections.
- Estimate the dominant shading direction from the centroid of darker interior pixels to the centroid of brighter interior pixels.
- Compute a secondary Sobel cue only on the smoothed shading image.
- Blend the centroid cue and Sobel cue with circular averaging.

This is the proposed method. It is fast, explainable, CPU-only, and more robust than the earlier raw Sobel orientation histogram.

### Module 2B: DPT-Large Deep-Learning Baseline

File:

```text
Code/core/2_illumination.py
```

Model:

```text
Intel/dpt-large
```

Main function:

```python
estimate_light_direction_dl(image_rgb, mask)
```

Important wording:

- DPT-Large does not directly estimate illumination.
- It estimates monocular depth.
- The pipeline derives a lighting direction from depth-weighted brightness over the segmented product.

This should be described as a depth-based deep-learning baseline, not a dedicated lighting-estimation model.

Paper:

```text
Vision Transformers for Dense Prediction
Ranftl, Bochkovskiy, Koltun
ICCV 2021
https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html
```

### Module 3: Deterministic Illumination-Aware Compositing

File:

```text
Code/core/3_generation.py
```

Main function:

```python
generate_background(image_rgb, product_mask, theta=None, bg_style="marble surface table")
```

The function name is preserved for compatibility with the notebook, app, and batch pipeline. The implementation is now deterministic compositing, not diffusion.

Current process:

1. Create a clean studio/table background procedurally.
2. Build an ambient contact shadow for the naive baseline.
3. For Sobel/DPT variants, translate and blur the product mask opposite the estimated light direction.
4. Darken the background under that synthetic shadow.
5. Paste the original product pixels back exactly.

Variants:

| Variant | Input angle | Behavior |
| --- | --- | --- |
| Naive | `None` | Ambient contact shadow only. |
| Sobel | `theta_sobel` | Cast shadow conditioned on proposed Sobel angle. |
| DPT | `theta_dl` | Cast shadow conditioned on DPT-derived angle. |

Why this pivot was made:

- Stable Diffusion inpainting was too uncontrolled for this project.
- It frequently hallucinated text, props, clutter, and product distortions.
- The deterministic compositor makes the computer vision contribution visible and controllable.
- It also preserves product identity exactly, which is important for product photography.

### Module 4: SDCS Evaluation

File:

```text
Code/core/4_evaluation.py
```

Main function:

```python
compute_sdcs(composite_rgb, product_mask, theta_fg_deg)
```

Concept:

- Find likely shadow pixels near/below the product base.
- Estimate the generated shadow direction.
- Compare it against the expected shadow direction.

Correct physical definition:

```text
expected shadow angle = foreground light angle + pi
SDCS = cos(theta_shadow - expected_shadow_angle)
```

Higher is better:

- `+1`: shadow direction matches expected physics.
- `0`: orthogonal / weak consistency.
- `-1`: shadow points in the opposite direction.

SDCS is a heuristic. It is useful for comparison, but can still be affected by background texture, transparent products, bad masks, and unusual object geometry.

## Notebook Workflow

Use:

```text
Code/CV2_Pipeline.ipynb
```

The notebook is now a thin Colab entry point. Large notebook logic has been moved to:

```text
Code/notebook_helpers/
```

This helper package contains Colab setup, image discovery, run preparation, result summaries, visualization, and GitHub push logic.

Recommended Colab flow:

1. Cell 1.1: clone or refresh from GitHub.
2. Cell 1.2: set working directory and import configuration.
3. Cell 1.5: configure Drive-backed model cache.
4. Cell 2.1: validate SAM segmentation.
5. Cell 2.2: validate Sobel and DPT illumination estimation.
6. Cell 2.3: validate deterministic compositing.
7. Cell 2.4: validate SDCS.
8. Cell 3.1: choose/prepare run ID.
9. Cell 3.2: run full batch over all images in `data/raw_images/`.
10. Cell 4.1: inspect product-by-product visual diagnostics.
11. Cell 5.1: commit and push notebook, masks, CSVs, and outputs.

Production mode:

- The pipeline automatically uses all supported images in:

```text
Code/data/raw_images/
```

- Supported extensions:

```text
.jpg, .jpeg, .png, .webp
```

- `test.jpg` is skipped during batch mode because it is reserved for module tests.

## Outputs

Batch outputs are versioned and never overwrite previous runs:

```text
Code/data/outputs/results_001.csv
Code/data/outputs/results_002.csv
Code/data/outputs/run_001/
Code/data/outputs/run_002/
```

Per image, the run folder stores:

```text
{image}_naive.png
{image}_sobel.png
{image}_dl.png
```

Masks are stored in:

```text
Code/data/masks/
```

The visualization cell saves product-level comparison rows:

```text
Code/data/outputs/comparison_run_NNN_{image}.png
```

## CSV Columns

The batch CSV includes:

- `image`
- `status`
- `theta_sobel`
- `theta_dl`
- `sobel_time_s`
- `dl_time_s`
- `sdcs_naive`
- `sdcs_sobel`
- `sdcs_dl`
- `lpips_naive`
- `lpips_sobel`
- `lpips_dl`
- `light_direction_naive`
- `light_direction_sobel`
- `light_direction_dl`
- `shadow_direction_naive`
- `shadow_direction_sobel`
- `shadow_direction_dl`
- `negative_prompt` or compositing constraints
- `prompt_naive` / compositing instruction
- `prompt_sobel` / compositing instruction
- `prompt_dl` / compositing instruction
- output paths

The word `prompt` remains in column names for compatibility with older runs, but the current implementation stores deterministic compositing instructions.

## GitHub Push From Colab

The final notebook push cell stages only:

```text
Project/Code/CV2_Pipeline.ipynb
Project/Code/data/masks
Project/Code/data/outputs
```

It does not stage:

- SAM weights.
- HuggingFace caches.
- model cache folders.
- unrelated local files.

It rejects staged files over GitHub's 100 MB file limit.

## Report Guidance

Use the final report to tell the story honestly:

1. The original proposal explored text-conditioned diffusion background generation.
2. Experiments showed text-only diffusion was too unstable for physically controlled shadows.
3. The final architecture moved to deterministic illumination-aware compositing.
4. The contribution is estimating foreground light direction and enforcing physically consistent shadows through mask-based geometry.
5. Sobel is compared against a DPT-Large depth-based DL baseline.
6. IC-Light is discussed as related work, not as the same task.

Recommended method section:

```text
Segmentation -> Illumination Estimation -> Shadow Synthesis -> Compositing -> Evaluation
```

Recommended comparison table:

| Method | Description |
| --- | --- |
| Naive | Clean background with ambient contact shadow only. |
| Sobel | Proposed Sobel angle controls cast-shadow direction. |
| DPT | DPT-derived angle controls cast-shadow direction. |

Recommended metrics:

- SDCS: shadow direction consistency.
- LPIPS: perceptual distance from original product photo.
- Timing: light-estimation speed for Sobel vs DPT.

Important report statements:

- DPT is a depth-based baseline, not a direct lighting estimator.
- The proposal table contained target/expected values, not completed results.
- The final table uses actual measured results from the run CSV.
- Transparent, reflective, and multi-light products remain hard.
- Procedural shadow synthesis is controllable but approximate.

## IC-Light Related Work

IC-Light should be mentioned because the TA asked for it.

IC-Light task:

```text
foreground subject + target lighting/background -> relight foreground
```

This project task:

```text
foreground product with existing lighting -> adapt background/shadow to match foreground
```

Suggested report wording:

```text
IC-Light addresses illumination harmonization by relighting the foreground subject to match a target lighting condition or background. Our project studies the inverse direction: preserving the product's observed lighting and adapting the generated/composited background and shadow to match it.
```

Reference:

```text
IC-Light GitHub: https://github.com/lllyasviel/IC-Light
```

## Known Limitations

- The Sobel-shading estimator can fail on transparent, reflective, glossy, very dark, or multi-light products.
- DPT can produce depth maps that are not reliable for small product details.
- SAM masks may fail on transparent objects or low-contrast boundaries.
- Synthetic shadows assume a simplified planar surface.
- Multiple light sources are not modeled.
- The SDCS shadow detector is heuristic and can be confused by textured backgrounds.

## Current Best Next Steps

1. Run a fresh Colab batch after pulling the deterministic-compositing code.
2. Inspect Cell 4.1 product-by-product.
3. Tune shadow constants in `Code/constants.py` if the shadows are too weak, too strong, or too displaced.
4. Add more clean product images to `Code/data/raw_images/`.
5. Use only actual CSV values in the final report.

## References

SAM:

```text
Kirillov et al. Segment Anything. ICCV 2023.
https://arxiv.org/abs/2304.02643
```

DPT:

```text
Ranftl, Bochkovskiy, Koltun. Vision Transformers for Dense Prediction. ICCV 2021.
https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html
```

LPIPS:

```text
Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.
https://arxiv.org/abs/1801.03924
```

IC-Light:

```text
https://github.com/lllyasviel/IC-Light
```

The original proposal PDF and older planning notes have been consolidated into this file. The proposal is historical context only. The current implementation and final report should follow this consolidated documentation.
