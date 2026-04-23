# Illumination-Aware Product Photography via Foreground Light Estimation and Deterministic Shadow Compositing

Himanshu Jhawar  
COMS 4732 Computer Vision II  
Columbia University

## Abstract

Consumer product photographs often contain cluttered backgrounds, weak composition, and inconsistent lighting. Existing background replacement tools can produce visually sharp images, but they frequently ignore a physical constraint that is immediately visible to humans: the generated scene should agree with the illumination already present on the product. This project studies illumination-aware product compositing from a single casual product image. The system segments the product using Segment Anything Model (SAM), estimates the product's foreground light direction using a proposed Sobel-shading method, compares that estimate against a DPT-Large depth-based deep-learning baseline, and creates a clean studio composite with a synthetic cast shadow placed opposite the estimated light direction. Unlike text-only diffusion background generation, the final pipeline preserves the original product pixels exactly and enforces lighting consistency through deterministic mask-based shadow synthesis. Evaluation uses Shadow Direction Consistency Score (SDCS), LPIPS, and light-estimation runtime. Results will be reported for a benchmark set of product images using three variants: a naive ambient-shadow composite, the proposed Sobel-conditioned composite, and a DPT-conditioned baseline composite.

## 1. Introduction

Product images are central to e-commerce, catalogs, and social media marketing. A product photo may have acceptable foreground detail but still look unprofessional because the background is cluttered or the shadow direction is physically inconsistent. Many modern tools can segment a product and replace its background, but they generally optimize for visual appearance rather than physical agreement between foreground lighting and background shadows.

This project focuses on a specific compositing question:

> Given a single product image, can we estimate the foreground lighting direction and use that estimate to create a more physically consistent studio composite?

The original plan explored text-conditioned diffusion inpainting for background generation. During implementation, this proved unreliable: the model often hallucinated text, props, clutter, or changed the product itself. The final system therefore uses a more controlled computer vision architecture. The foreground product is segmented and preserved exactly. A clean studio background is created procedurally. A cast shadow is synthesized directly from the product mask and placed in the direction implied by the estimated foreground illumination.

The main contributions are:

1. A training-free Sobel-shading method for estimating product foreground light direction.
2. A DPT-Large depth-based deep-learning baseline for comparison.
3. A deterministic shadow compositing pipeline that converts estimated light angle into an explicit cast shadow.
4. A shadow-direction metric, SDCS, for evaluating physical consistency.
5. A Colab-based benchmark workflow that logs angles, outputs, metrics, and timing for each product.

## 2. Related Work

### Product Background Replacement

Commercial tools such as Adobe Firefly, Canva, and remove.bg can segment objects and replace backgrounds. These systems generally emphasize visual plausibility, style, and convenience. However, a visually attractive generated background can still look fake if its shadow direction contradicts the observed lighting on the product. This motivates a more physics-aware treatment of product compositing.

### Segment Anything

The pipeline uses Segment Anything Model (SAM) for foreground extraction. SAM provides a strong zero-shot segmentation capability and is suitable for product photos because it can produce detailed masks from minimal prompting. In this project, SAM is used to obtain the binary product mask that drives both product preservation and synthetic shadow generation.

### Dense Prediction Transformers

The deep-learning comparison method uses DPT-Large (`Intel/dpt-large`), based on Vision Transformers for Dense Prediction. DPT-Large is not a direct illumination estimator. It predicts monocular depth. The project uses this depth map as a geometric cue and derives a light direction from depth-weighted product brightness. This provides a modern deep-learning baseline against which the proposed Sobel method can be compared.

### IC-Light

IC-Light is relevant because it addresses illumination harmonization with generative relighting. In its common setup, a foreground subject is relit to match a target lighting condition or background. This project studies the inverse direction: the product's existing foreground lighting is preserved, and the background/shadow is adapted to match that lighting. IC-Light is therefore related but not identical to the target task. This distinction is important because product photography often requires preserving the actual product appearance rather than relighting or altering it.

### Perceptual Metrics

LPIPS measures perceptual similarity using deep feature distances. It is useful here because the product should remain visually close to the original input. However, LPIPS alone does not measure physical lighting consistency, which motivates the additional SDCS metric.

## 3. Problem Formulation

Input:

- A single RGB product image `I`.
- A product mask `M` estimated by SAM.

Output:

- A studio-style composite image `C`.
- A foreground light angle estimate `theta_fg`.
- A synthetic shadow whose direction is consistent with `theta_fg`.

The physical goal is:

```text
shadow direction should be opposite the estimated foreground light direction
```

If the estimated light direction is `theta_fg`, the expected cast-shadow direction is:

```text
theta_expected_shadow = theta_fg + pi
```

The project compares three compositing variants:

| Variant | Angle used    | Description                                             |
| ------- | ------------- | ------------------------------------------------------- |
| Naive   | None          | Clean background with ambient contact shadow only.      |
| Sobel   | `theta_sobel` | Proposed Sobel estimate controls cast-shadow direction. |
| DPT     | `theta_dl`    | DPT-derived estimate controls cast-shadow direction.    |

## 4. Method

The full pipeline is:

```text
Input image
-> SAM segmentation
-> Sobel illumination estimation
-> DPT depth-based baseline estimation
-> background creation
-> mask-based shadow synthesis
-> product compositing
-> SDCS / LPIPS / timing evaluation
```

### 4.1 Product Segmentation

The first stage segments the product using SAM ViT-H. Given an input product photo, SAM returns a binary mask:

```text
M(x, y) = 1 for product pixels
M(x, y) = 0 for background pixels
```

The mask is used in three places:

1. To restrict illumination estimation to the product.
2. To synthesize a product-shaped cast shadow.
3. To paste the original product pixels back into the final composite.

This third use is important. The final system does not ask a generative model to redraw the product. It preserves the product pixels exactly:

```text
C[M = 1] = I[M = 1]
```

This avoids product identity drift, text hallucination, and geometry changes that occurred during earlier text-only diffusion experiments.

### 4.2 Sobel-Shading Illumination Estimation

The proposed method estimates the foreground light direction using a training-free combination of low-frequency product shading and a secondary Sobel gradient cue. The initial implementation used a magnitude-weighted Sobel orientation histogram, but this was too sensitive to product silhouettes, printed texture, and specular highlights. The current method therefore treats raw edges as unreliable and first tries to recover the dominant shading trend inside the object.

First, the product mask is eroded to remove boundary pixels. This prevents the object outline from dominating the angle estimate. The RGB image is converted to grayscale, background pixels are filled with the product median intensity, and a Gaussian blur is applied:

```text
I_gray = grayscale(I)
I_smooth = GaussianBlur(I_gray inside mask)
```

The method then identifies relatively dark and bright interior regions using intensity percentiles:

```text
R_dark = pixels below the 25th percentile
R_bright = pixels above the 75th percentile
```

The main angle cue is the vector from the dark-region centroid to the bright-region centroid:

```text
theta_centroid = atan2(y_bright - y_dark, x_bright - x_dark)
```

This expresses the assumption that, after texture has been smoothed away, the brighter side of the product is closer to the dominant light source. A weak Sobel cue is still computed, but only on the smoothed shading image and only inside the eroded mask:

```text
Gx = dI_smooth / dx
Gy = dI_smooth / dy
```

The Sobel cue is converted into a weighted gradient direction:

```text
magnitude = sqrt(Gx^2 + Gy^2)
angle = atan2(Gy, Gx)
```

Finally, the centroid angle and Sobel angle are blended using circular averaging:

```text
theta_sobel = circular_blend(theta_centroid, theta_gradient)
```

This method remains simple, fast, and interpretable, but it is more robust than a raw Sobel histogram because it suppresses high-frequency texture and mask-boundary edges. It is still most reliable for opaque products with visible single-source shading and less reliable for transparent, reflective, glossy, or multi-light scenes.

### 4.3 DPT-Large Depth-Based Baseline

To address the need for a deep-learning comparison, the project includes a DPT-Large baseline. DPT-Large estimates a dense monocular depth map:

```text
D = DPT(I)
```

The method then combines normalized depth and grayscale brightness:

```text
A(x, y) = D(x, y) * brightness(x, y)
```

Inside the product mask, pixels in the top activation percentile are treated as likely light-catching regions. The baseline computes the vector from the product center to the center of these activated pixels:

```text
theta_dl = atan2(y_light - y_center, x_light - x_center)
```

This is a depth-based lighting proxy, not a direct learned illumination estimator. Its role is to compare the proposed Sobel method against a modern learned visual representation.

### 4.4 Deterministic Studio Background

Instead of relying on text-only diffusion, the current pipeline creates a clean studio-style background procedurally. The background includes:

- a wall-to-table gradient,
- a horizon/surface transition,
- subtle texture based on the selected style,
- optional directional brightness modulation based on the estimated angle.

This design is intentionally conservative. The goal is not to generate artistic scenes, but to create a stable background where shadow direction can be controlled and evaluated.

### 4.5 Mask-Based Shadow Synthesis

The synthetic shadow is generated from the product mask. The naive baseline receives only an ambient contact shadow. For Sobel and DPT variants, the mask is dilated, translated, blurred, and darkened.

Given estimated light angle `theta`, the shadow direction is:

```text
theta_shadow = theta + 180 degrees
```

The mask is translated by:

```text
dx = cos(theta_shadow) * offset
dy = sin(theta_shadow) * offset
```

The translated mask is blurred to approximate a soft cast shadow. It is then composited over the background as a dark alpha layer. Finally, the original product pixels are pasted back using the SAM mask.

This makes the shadow direction explicit and controllable. It also makes the method more computer-vision-centered than text-conditioned generation.

## 5. Evaluation

The report will evaluate the system along three axes:

1. Physical shadow direction consistency using SDCS.
2. Perceptual similarity using LPIPS.
3. Runtime efficiency of light estimation.

### 5.1 Shadow Direction Consistency Score

SDCS measures whether the detected composite shadow points in the expected direction. The metric estimates a shadow centroid in the background near the product base. It then computes the angle from the product base to that shadow centroid:

```text
theta_shadow = atan2(y_shadow - y_base, x_shadow - x_product_center)
```

The expected shadow direction is opposite the estimated foreground light:

```text
theta_expected = theta_fg + pi
```

The score is:

```text
SDCS = cos(theta_shadow - theta_expected)
```

Interpretation:

| SDCS | Meaning                                             |
| ---- | --------------------------------------------------- |
| `+1` | Shadow direction perfectly matches expectation.     |
| `0`  | Shadow direction is orthogonal / weakly consistent. |
| `-1` | Shadow direction is physically inverted.            |

SDCS is a heuristic and should be interpreted alongside qualitative examples.

### 5.2 LPIPS

LPIPS is computed between the original product image and each composite at a fixed size. Lower LPIPS means the composite is perceptually closer to the original image. Because the final pipeline preserves product pixels exactly, LPIPS mainly captures background and overall compositing changes rather than product identity drift.

### 5.3 Runtime

The report compares:

- Sobel runtime per image.
- DPT runtime per image.

This comparison is important because Sobel is CPU-only and lightweight, while DPT requires loading and running a large learned model.

## 6. Experimental Setup

### Dataset

The benchmark images are stored in:

```text
Code/data/raw_images/
```

The final report should state the exact number of images used:

```text
TODO: Insert final image count here.
```

The dataset should include a mix of product types, for example:

- bottles,
- cans,
- glass objects,
- dark reflective objects,
- metallic objects,
- standard opaque products.

Transparent and reflective objects should be discussed as stress tests because they are difficult for both segmentation and illumination estimation.

### Compared Methods

| Method | Description                                                    |
| ------ | -------------------------------------------------------------- |
| Naive  | Procedural studio background plus ambient contact shadow only. |
| Sobel  | Proposed Sobel angle controls synthetic cast-shadow direction. |
| DPT    | DPT-derived angle controls synthetic cast-shadow direction.    |

### Implementation

The full workflow is implemented in Python and run through:

```text
Code/CV2_Pipeline.ipynb
```

The notebook is intentionally thin. Larger Colab orchestration logic, including setup, image discovery, run summaries, visualization, and GitHub result pushing, is kept in:

```text
Code/notebook_helpers/
```

Core modules:

| Module                 | File                          | Role                                             |
| ---------------------- | ----------------------------- | ------------------------------------------------ |
| Notebook orchestration | `Code/notebook_helpers/`      | Colab helper package used by the notebook cells. |
| Segmentation           | `Code/core/1_segmentation.py` | SAM product mask.                                |
| Illumination           | `Code/core/2_illumination.py` | Sobel and DPT light estimation.                  |
| Compositing            | `Code/core/3_compositing.py`  | Background and synthetic shadow creation.        |
| Evaluation             | `Code/core/4_evaluation.py`   | SDCS metric.                                     |
| Batch run              | `Code/main_pipeline.py`       | Runs all images and writes CSV/results.          |

## 7. Results

This section should be completed after the final Colab run. Do not use proposal target numbers here. Use actual numbers from:

```text
Code/data/outputs/results_NNN.csv
```

### 7.1 Quantitative Results

TODO: Replace `TBD` values with final measured results.

| Method                   | Mean SDCS up | Mean LPIPS down | Mean light-estimation time down |
| ------------------------ | -----------: | --------------: | ------------------------------: |
| Naive ambient shadow     |          TBD |             TBD |                             N/A |
| Sobel-conditioned shadow |          TBD |             TBD |                             TBD |
| DPT-conditioned shadow   |          TBD |             TBD |                             TBD |

Suggested additional table:

| Method                   | Median SDCS | Std SDCS | Failed / no-shadow cases |
| ------------------------ | ----------: | -------: | -----------------------: |
| Naive ambient shadow     |         TBD |      TBD |                      TBD |
| Sobel-conditioned shadow |         TBD |      TBD |                      TBD |
| DPT-conditioned shadow   |         TBD |      TBD |                      TBD |

### 7.2 Runtime Results

TODO: Fill from final CSV.

| Estimator | Mean time per image | Notes                         |
| --------- | ------------------: | ----------------------------- |
| Sobel     |                 TBD | CPU-only, no model loading.   |
| DPT-Large |                 TBD | Requires DPT model inference. |

### 7.3 Qualitative Results

Include product-by-product comparison rows generated by Cell 4.1:

```text
Code/data/outputs/comparison_run_NNN_{image}.png
```

Recommended figure layout:

```text
Original | SAM Product | Naive | Sobel-conditioned | DPT-conditioned
```

For each figure, mention:

- Sobel angle,
- DPT angle,
- shadow direction used,
- visible success or failure case.

TODO: Insert 3-5 representative qualitative examples.

## 8. Discussion

### Sobel vs DPT

The Sobel-shading method is fast and transparent. It relies on low-frequency product shading and a lightweight gradient cue, so it has almost no computational overhead. DPT is more expensive but uses a learned global representation of scene geometry. The final results should compare whether DPT's additional computation leads to better SDCS than the proposed training-free method.

Important interpretation:

- If Sobel achieves similar SDCS to DPT, it is a strong result because Sobel is cheaper and easier to explain.
- If DPT performs better on some images, analyze whether those images have geometry cues, transparent material, or specular highlights that the training-free method could not capture.
- If both fail, inspect segmentation quality and material type.

### Why Deterministic Compositing Was Chosen

The project initially explored diffusion inpainting. However, text-only diffusion was not reliable enough for controlled physical shadow placement. It often generated visually busy backgrounds, extra objects, text-like artifacts, or changed the product. This is problematic for product photography because the product must remain accurate.

The final deterministic compositor makes the project more focused:

- product pixels are preserved,
- shadow direction is explicit,
- failure modes are easier to analyze,
- SDCS becomes more meaningful,
- the pipeline is reproducible.

### IC-Light Relationship

IC-Light relights a foreground subject to match a target background or lighting condition. This project studies the reverse problem: the product's foreground lighting is kept fixed, and the background/shadow is adapted to match it. This difference is important for consumer product photography, where changing the product appearance can be undesirable.

## 9. Limitations

The method is intentionally controlled, but it has limitations:

1. **Simplified shadow model:** The synthetic shadow assumes a planar surface and a single dominant light source.
2. **Reflective and transparent products:** Sobel-shading cues and DPT depth can be unreliable on glass, metallic, glossy, or transparent objects.
3. **Segmentation dependency:** Bad SAM masks directly harm compositing and SDCS.
4. **No true 3D geometry:** The method uses a 2D mask shadow, not a physically rendered object model.
5. **SDCS heuristic:** The shadow detector can be confused by textured backgrounds or unusual object bases.
6. **Dataset size:** Final claims should be limited to the actual number and type of tested images.

## 10. Future Work

Future work could improve the pipeline in several directions:

- Add a stronger confidence estimate for Sobel-shading angles using dark-bright centroid separation, gradient coherence, and material cues.
- Use a matting model for transparent products.
- Fit a ground plane and perspective-aware shadow transform.
- Estimate multiple light sources instead of one dominant angle.
- Use a learned illumination estimator trained directly for lighting direction.
- Compare against IC-Light or a dedicated relighting model as an additional baseline.
- Add a small human preference study for visual realism.

## 11. Conclusion

This project presents an illumination-aware product compositing pipeline that estimates product lighting and uses it to synthesize physically consistent cast shadows. The proposed Sobel-shading method is simple, training-free, and computationally efficient. A DPT-Large depth-based baseline provides a learned comparison point. The final deterministic compositor avoids the instability of text-only diffusion by preserving product pixels exactly and controlling shadow direction explicitly through the product mask. The result is a reproducible computer vision pipeline for studying whether foreground illumination cues can improve the physical consistency of generated product photography composites.

## References

[1] Alexander Kirillov et al. Segment Anything. ICCV 2023.  
https://arxiv.org/abs/2304.02643

[2] Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision Transformers for Dense Prediction. ICCV 2021.  
https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html

[3] Richard Zhang et al. The Unreasonable Effectiveness of Deep Features as a Perceptual Metric. CVPR 2018.  
https://arxiv.org/abs/1801.03924

[4] Lvmin Zhang et al. IC-Light.  
https://github.com/lllyasviel/IC-Light

[5] R. J. Woodham. Photometric Method for Determining Surface Orientation from Multiple Images. Optical Engineering, 1980.

[6] Jizheng Yi et al. Illuminant direction estimation for a single image based on local region complexity analysis and average gray value. Applied Optics, 2014.  
https://doi.org/10.1364/AO.53.000226

[7] Peter Kocsis et al. LightIt: Illumination Modeling and Control for Diffusion Models. CVPR 2024.  
https://openaccess.thecvf.com/content/CVPR2024/html/Kocsis_LightIt_Illumination_Modeling_and_Control_for_Diffusion_Models_CVPR_2024_paper.html

[8] Haian Jin et al. Neural Gaffer: Relighting Any Object via Diffusion. NeurIPS 2024.  
https://openreview.net/forum?id=zV2GDsZb5a
