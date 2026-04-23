"""
Gradio Demo App: Illumination-Aware Product Photography
Run this on Colab with: !python app.py
A public shareable link will be printed in the output.
"""
import os
import sys
import importlib.util
import numpy as np
import cv2
import gradio as gr
from PIL import Image

from constants import BACKGROUND_STYLE_OPTIONS, DEFAULT_BACKGROUND_STYLE, GRADIO_TEMP_INPUT_NAME, OUTPUTS_DIR

# ---- Load our modules (importlib handles filenames starting with digits) ----

def _load_module(filename, module_name):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join("core", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

seg_mod  = _load_module("1_segmentation.py", "segmentation")
illu_mod = _load_module("2_illumination.py", "illumination")
gen_mod  = _load_module("3_generation.py",   "generation")
eval_mod = _load_module("4_evaluation.py",   "evaluation")

segment_product               = seg_mod.segment_product
estimate_light_direction_sobel = illu_mod.estimate_light_direction_sobel
generate_background            = gen_mod.generate_background
compute_sdcs                   = eval_mod.compute_sdcs


# ---- Gradio pipeline function ----

def run_pipeline(input_image_pil, bg_style, use_dl_baseline):
    """
    Called by Gradio when the user clicks Generate.

    Args:
        input_image_pil: PIL Image from the Gradio upload widget
        bg_style: background style string chosen by user
        use_dl_baseline: bool - also show the DL baseline comparison

    Returns:
        naive_img, sobel_img, dl_img (or None), label_text
    """
    # Save temp image so SAM can load it from disk
    tmp_path = str(OUTPUTS_DIR / GRADIO_TEMP_INPUT_NAME)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    input_image_pil.save(tmp_path)

    # 1. Segment the product
    print("Step 1: Segmenting product with SAM...")
    mask, image_rgb = segment_product(tmp_path)

    # 2. Estimate light direction
    print("Step 2: Estimating light direction...")
    theta_sobel = estimate_light_direction_sobel(image_rgb, mask)
    print(f"  Sobel angle: {theta_sobel:.1f} degrees")

    theta_dl = None
    if use_dl_baseline:
        theta_dl = illu_mod.estimate_light_direction_dl(image_rgb, mask)
        print(f"  DL angle:    {theta_dl:.1f} degrees")

    # 3. Composite clean backgrounds and synthetic shadows
    print("Step 3: Compositing backgrounds and synthetic shadows...")
    naive_pil = generate_background(image_rgb, mask, theta=None,        bg_style=bg_style)
    sobel_pil = generate_background(image_rgb, mask, theta=theta_sobel, bg_style=bg_style)
    dl_pil    = generate_background(image_rgb, mask, theta=theta_dl,    bg_style=bg_style) if use_dl_baseline else None

    # 4. Score with SDCS
    naive_arr = np.array(naive_pil)
    sobel_arr = np.array(sobel_pil)

    sdcs_naive = compute_sdcs(naive_arr, mask, theta_sobel)
    sdcs_sobel = compute_sdcs(sobel_arr, mask, theta_sobel)

    sdcs_dl_str = ""
    if dl_pil is not None:
        dl_arr  = np.array(dl_pil)
        sdcs_dl = compute_sdcs(dl_arr, mask, theta_dl)
        sdcs_dl_str = f"  DL SDCS:    {sdcs_dl:.3f}" if sdcs_dl is not None else "  DL SDCS:    no shadow detected"

    label = (
        f"Estimated light angle (Sobel): {theta_sobel:.1f} degrees\n"
        f"Estimated light angle (DL):    {theta_dl:.1f} degrees\n" if theta_dl is not None else ""
        f"\n"
        f"SDCS Scores (higher = more physically realistic shadow):\n"
        f"  Naive SDCS: {sdcs_naive:.3f}" if sdcs_naive is not None else "  Naive SDCS: no shadow detected"
    )
    label = (
        f"Estimated light angle (Sobel): {theta_sobel:.1f} degrees"
        + (f"\nEstimated light angle (DL): {theta_dl:.1f} degrees" if theta_dl is not None else "")
        + f"\n\nSDCS Scores (higher = more physically realistic shadow):"
        + (f"\n  Naive SDCS:  {sdcs_naive:.3f}" if sdcs_naive is not None else "\n  Naive SDCS:  no shadow detected")
        + (f"\n  Sobel SDCS:  {sdcs_sobel:.3f}" if sdcs_sobel is not None else "\n  Sobel SDCS:  no shadow detected")
        + sdcs_dl_str
    )

    return naive_pil, sobel_pil, dl_pil, label


# ---- Build Gradio Interface ----

with gr.Blocks(title="Illumination-Aware Product Photography") as demo:
    gr.Markdown(
        """
        # Illumination-Aware Product Photography
        Upload a casual product photo -> the pipeline estimates the lighting angle -> composites a clean studio background with a matching synthetic shadow.

        **How it works:**
        1. SAM segments the product from the background
        2. Sobel gradients estimate the light direction angle
        3. A deterministic compositor creates a clean studio background and a shadow conditioned on that angle
        4. SDCS score measures how well the generated shadow matches physics
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload your product photo")
            bg_style = gr.Dropdown(
                choices=list(BACKGROUND_STYLE_OPTIONS),
                value=DEFAULT_BACKGROUND_STYLE,
                label="Background style"
            )
            use_dl = gr.Checkbox(label="Also run Deep Learning baseline (slower, requires ~1GB extra RAM)", value=False)
            generate_btn = gr.Button("Generate Studio Photos", variant="primary")

        with gr.Column(scale=2):
            out_naive = gr.Image(label="Naive AI (no lighting info)")
            out_sobel = gr.Image(label="Sobel Pipeline (our method)")
            out_dl    = gr.Image(label="Deep Learning Baseline (if enabled)")
            out_label = gr.Textbox(label="Results & Scores", lines=8)

    generate_btn.click(
        fn=run_pipeline,
        inputs=[input_image, bg_style, use_dl],
        outputs=[out_naive, out_sobel, out_dl, out_label]
    )

if __name__ == "__main__":
    print("Starting Gradio with share=True. The public URL will appear below after Gradio finishes setup.", flush=True)
    demo.launch(share=True, show_error=True)   # share=True gives a public link on Colab
