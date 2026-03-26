# My NeRF Journey: Concepts & Code

## 🎯 Big Picture Objective
My goal for this assignment is to take multi-view 2D images of a Lego model (and eventually a video I shoot on my phone), and train a Neural Network to figure out its 3D geometry and colors. When the network finishes training, I should be able to fly a virtual camera completely around the object and render "novel views" from angles that I never even captured!

## 🧠 Core Concepts Cheat Sheet
* **NeRF (Neural Radiance Field):** A technique storing an entire 3D scene inside the weights of a deep neural network rather than using polygons or voxels.
* **Volume Rendering:** The physics math used to shoot virtual rays through a scene, determining the final color of a pixel based on the density and color of the semi-transparent particles the ray passes through.
* **Structure from Motion (SfM):** The algorithm (like COLMAP) that calculates *exactly where* the camera was standing when a photo was taken, by matching feature points across overlapping images.
* **Camera Intrinsics ($K$):** The internal physical properties of a camera lens (focal length, sensor center) that dictate how 3D light projects onto a 2D digital image.
* **Camera Extrinsics ($c2w$):** A 4x4 matrix telling me exactly where my camera is located and where it is pointing in the 3D world (Camera-to-World matrix).
* **Positional Encoding:** A math trick using sine/cosine waves to turn basic inputs like `(x, y)` into long, high-frequency vectors so a neural network can learn tiny, sharp details instead of producing blurry blobs.
* **Multilayer Perceptron (MLP):** The actual deep learning brain (built out of sequentially stacked linear layers and ReLUs) that learns to associate 3D coordinates with RGB colors.

---

## Phase 1: 2D Neural Fields (The Sandbox)
**Concept:** 
Before jumping into 3D, I built a network that memorizes a single 2D picture. Instead of storing an image as a 2D array of pixels, a "Neural Field" acts as a function `f(x, y) = (r, g, b)`. If I pass in a pixel coordinate, the network spits out the color.

**Key Code Pieces:**
- **Positional Encoding (`PositionalEncoding`):** Standard neural networks (MLPs) are terrible at learning sharp edges or high-frequency details. I had to map my basic `(x, y)` coordinates to a much higher dimension using sine and cosine waves. My `L=10` parameter created a 42-dimensional vector.
  - *Wait, what is `(x, y)` in this context?* They are the 2D coordinates of a single pixel on my screen, normalized between 0 and 1. The formula is simply: "If I hand you this 2D coordinate, predict the exact RGB color of the pixel located there."
  - *Why are MLPs "terrible at learning sharp edges"?* Neural networks suffer from "spectral bias", meaning they prefer learning smooth, blurry gradients. A tiny input change from `0.499` to `0.501` means practically nothing to a normal MLP, so it outputs a blurry gray smear instead of a sharp black-to-white edge.
  - *How does mapping using $L=10$ create exactly 42 dimensions?* The formula keeps the original `(x, y)`, and generates a new `sin` and `cos` wave for *every* frequency step up to $L$. For $L=10$: 10 sine + 10 cosine for `x` (= 20), plus 10 sine + 10 cosine for `y` (= 20), plus the original `(x,y)` (= 2) totals **42 dimensions**. Instead of feeding the network `[0.5, 0.5]`, I give it a massive array of 42 high-frequency numbers.
  - *How does this actually help the model learn sharp edges?* A massive sine wave like `sin(2^9 * pi * x)` swings violently from 1 to -1 even if `x` only changes by `0.001`. That violent mathematical swing acts as a screaming alarm to the network saying: *"Hey! The physical location just changed!"* allowing it to instantly detect sharp edges.
- **The MLP (`NeuralField2D`):** A deep network with 4 hidden layers of size 256. It takes my 42-dimensional encoded coordinate and processes it through `ReLU` activations. The final layer predicts 3 values (R, G, B) and uses a `Sigmoid` to make sure the color is clamped between 0 and 1.
- **Training Loop:** I sampled 10,000 random pixels per step, calculated the Mean Squared Error (MSE) loss against the real pixel colors, and ran an Adam optimizer. I also tracked Peak Signal-to-Noise Ratio (PSNR) to measure image reconstruction quality.
  - *Wait, what exactly is PSNR?* PSNR (Peak Signal-to-Noise Ratio) is a Computer Vision metric that converts math error (MSE) into a logarithmic decibel (dB) scale. Because the human eye perceives light logarithmically, a minor decimal drop in MSE might look like a massive visual leap in clarity. A score over 30 dB is generally visually indistinguishable from the original.
  - *Why exactly a batch of 10,000 random pixels?* In Phase 1 (2D), my GPU could fit the whole solid image. But NeRF is built for 3D! In 3D, for every single pixel ray on an 800x800 image, my network will sample 64 distinct coordinates bouncing through space. Forward-passing 40.9 million 3D coordinates (and logging all their gradients) demands hundreds of gigabytes of VRAM, which produces an instant `OutOfMemory` crash. Shuffling and grabbing exactly 10,000 random rays is the sweet spot that teaches the overall geometry efficiently without exceeding standard 8-16GB memory buffers.
  - *Why specifically MSE Loss and the Adam optimizer?* We use **MSE Loss** because we are forcing the network to output the exact continuous RGB floats of the target pixel (regression), heavily penalizing giant color outliers. We use the **Adam Optimizer** instead of standard SGD because our massive positional encoding waves create a violently bumpy "loss landscape". Adam tracks momentum dynamically, surfing completely over the chaotic high-frequency bumps rather than getting hopelessly stuck in a mathematical rut.

---

## Phase 2: 3D Camera Geometry & Ray Casting
**The Big Objective for `dataset_3d.py`:**
The overarching intention of Phase 2 is to handle all the complex 3D geometry of our cameras *before* we even touch a Neural Network. If I have 100 images in my training set, I have millions of pixels. Because a NeRF network can only process 3D space, I need a way to mathematically convert every single 2D pixel into a physical 3D light ray (a Ray Origin vector and a Ray Direction vector) shooting through the scene. Once all millions of rays are calculated, my goal is to toss them into a massive, randomized "bucket" (a PyTorch `Dataset` called `RaysData`) so the network can pull random batches of 10,000 rays and train without blowing up my GPU's memory limit.

**Concept:**
In 3D, my input isn't a pixel. Instead, for every pixel on my screen, a virtual camera "shoots a ray" into the 3D scene.
- **Intrinsics ($K$ matrix):** Deals with my camera's focal length and optical center. Maps 3D points inside the camera to my 2D screen.
- **Extrinsics ($c2w$ / $w2c$ matrix):** Tells me exactly where my camera is physically sitting and rotating in the actual 3D world (World-to-Camera or Camera-to-World).
  - *Wait, where do these $K$ and $c2w$ matrices actually come from?* In Phase 2, they are literally handed to me on a silver platter! The TAs pre-calculated them and saved them inside the `lego_200x200.npz` dataset. My `load_data()` function just reads them directly.
  - *If I shot my own video, how would I get them?* That is what Phase 3 is for! I will feed all my video frames into a Structure from Motion (SfM) software called **COLMAP**. COLMAP finds overlapping features (like the corner of a table visible in 5 photos) and uses complex trigonometry to reverse-engineer my exact $K$ lens focal length, and the 3D $c2w$ coordinates of exactly where I was standing when I snapped every photo.

**Key Code Pieces (`dataset_3d.py`):**
- `image_to_rays()`: Given one image, I generate an `(H, W)` grid of pixel coordinates `(u, v)`. **Important gotcha I shouldn't forget:** I have to add 0.5 to these coordinates to shoot the ray through the *center* of the pixel instead of the top-left corner! Then I flatten them and run physics math to get a Ray Origin (`r_o`) and Ray Direction (`r_d`) for each pixel.
- `images_to_rays()`: Same as above, but looping through $N$ images to create a massive `(N, H, W, 6)` tensor holding the rays for all my cameras.
- `RaysData` Class:
  - This acts as my massive bucket of training data. I can't feed an entire image through the network at once without crashing my GPU.
  - `__init__()`: I pre-calculate every single ray across my dataset. I flatten all the image colors (`gt_rgbs`) and flatten all the rays (`rays_o`, `rays_d`) into lists. I also save the integer `(x, y)` coordinates of each ray inside `self.uvs` just for debugging visualizers.
  - `sample_rays()`: When my training loop asks for data, I randomly pick a batch of 10,000 integer indices and pull out the matching rays from my massive bucket.

---

## Phase 3: 3D Sampling & Volume Rendering
**The Big Objective for `rendering.py`:**
In Phase 2 I calculated 10,000 continuous Ray lines. Neural Networks cannot process infinite lines—they need discrete 3D $(X,Y,Z)$ points. In Phase 3, my job is to slice those continuous lines into exactly 64 evenly-spaced 3D dots using the physical Ray Equation $P(t) = O + t \times D$. Once I have those dots, I feed them into the NeRF network to get their Color and Density ($\sigma$). Finally, `volrend` squashes all 64 semi-transparent dots together to calculate the final pixel color on my screen.

**Key Code Pieces & Concepts:**
- **Ray Marching (`sample_along_rays`):** We define a `near` plane (e.g., 2.0m) and a `far` plane to create a box around the actual Lego model, throwing away the math for empty sky. We slice this line into 64 even distance steps ($t$).
  - *Wait, how do we select those 64 distances?* We use `torch.linspace(near, far, 64)` to generate perfectly even steps.
- **Perturbation (Jittering):** Adding a mathematically random amount of noise to the 64 distances during training.
  - *Why do we perturb the steps?* If we always tested the exact same 64 distances (like exactly 2.0, 2.1, 2.2), the network would overfit and just memorize those specific 64 depths instead of learning the actual shape of the model. By randomly jittering the dots back and forth every step, the network learns a perfectly smooth, continuous solid 3D geometry!
- **The NeRF Output ($\sigma$ and RGB):** 
  - *What exactly are the `sigmas` ($\sigma$) predicted by the model?* Instead of treating space like hard, solid video game polygons, NeRF treats everything like semi-transparent colored fog. $\sigma$ is simply the "Density" or "Thickness" of the fog! If $\sigma \approx 0$, it's empty air. If $\sigma$ is huge, it's a solid piece of yellow plastic on the Lego model.
- **Volume Rendering (`volrend`):**
  - *What exactly does `volrend()` do?* It marches down the 64 points calculating the **Physics of Light**. If point 11 is highly dense yellow, `volrend` calculates that a ton of yellow light hits your camera lens. Crucially, the math suppresses light from points 12 through 64 because they are physically blocked by that dense yellow layer in front! The algorithm mathematically integrals/squashes all 64 semi-transparent fog layers into a single crisp RGB pixel value.

---

## Phase 4: Training & Novel View Generation
**The Big Objective for `model.py` and `train_3d.ipynb`:**
Now that we have 3D coordinates and camera rays, we need to build the actual deep neural network architecture. Unlike Phase 1 where we mapped 2D pixels to colors, here we map **3D Coordinates + Viewing Angle** to **RGB Color + Density**. After training, we generate the final "Orbital Video" showing the network has learned the complete 3D structure.

**Key Code Pieces & Concepts:**
- **The NeRF MLP (`NeuralRadianceField`):**
  - The model is 8 layers deep to process the 3D space (`xyz` coordinates). 
  - *Why do we split Density and Color?* Density ($\sigma$) relies *only* on the 3D coordinate (a piece of Lego plastic is always solid no matter where you look from). But Color (RGB) is **view-dependent** (plastic is shiny and reflects light differently depending on your viewing angle). So, we calculate Density first, then concatenate the `r_d` (Viewing Direction) vector near the very end of the network to calculate Color!
  - *What is the Skip Connection?* Because 8 layers is very deep, the network starts "forgetting" the original 3D coordinate. We manually concatenate the original `xyz` Positional Encoding back into the 5th layer to refresh its memory.
- **The Dead ReLU Trap:** 
  - *Why did the model initially output pitch black space?* We originally included a `torch.pi` multiplier in our Positional Encoding like we did in Phase 1. However, since 3D space coordinates are up to 6.0 (instead of 1.0), multiplying by $\pi$ caused the sine wave frequencies to hit $\approx 10,000$. This caused the gradients to explode on Step 1, launching all weights into the negatives. Since Density relies on a `ReLU` activation, all negative inputs returned exactly `0.0` (empty vacuum), generating a completely black image! Removing `torch.pi` solved this.

**Generating Novel Views (`vis_orbit.py`):**
- **Concept:** Because the model actually learned the full physical 3D geometry of the object (rather than just memorizing 2D training pictures), we can literally put our virtual camera *anywhere in space* to create fully novel images not seen in the dataset!
- **How `vis_orbit.py` works:**
  1. We pick a starting distance (e.g., $3.9$ meters away from the `[0,0,0]` origin).
  2. We use Python math to build a fake Camera-to-World ($c2w$) matrix looking squarely at the origin.
  3. We set up an angle loop (`phi` from $0$ to $360$ degrees), mathematically forcing the fake camera matrix to perfectly orbit the origin in 60 smooth 6-degree steps.
  4. For every angle step, we pretend it's a real camera: We shoot exactly 40,000 rays using the trained `model.pth` weights we fetched from Colab.
  5. The model accurately hallucinates the entire front, side, and back of the Lego structure from these entirely unique angles.
  6. We stitch the 60 newly generated images together using `imageio` to produce `lego_orbit.gif`!
