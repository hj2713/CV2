import math
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import imageio
import os

from dataset_3d import load_data, image_to_rays
from rendering import sample_along_rays, volrend
from model import NeuralRadianceField

def look_at_origin(pos):
    forward = -pos / np.linalg.norm(pos)  
    up = np.array([0, 1, 0])
    right = np.cross(up, forward)
    right = right / np.linalg.norm(right)
    up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w

def rot_x(phi):
    return np.array([
        [math.cos(phi), -math.sin(phi), 0, 0],
        [math.sin(phi), math.cos(phi), 0, 0],
        [0,0,1,0],
        [0,0,0,1],
    ])


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Executing 360 Orbital Rendering on {device}...")
    
    # LOAD WEIGHTS
    model = NeuralRadianceField().to(device)
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model.load_state_dict(torch.load(os.path.join(BASE_DIR, "output/phase3/nerf_model_weights.pth"), map_location=device))
    model.eval()
    
    near = 2.0
    far = 6.0
    num_samples = 64
    chunk_size = 16384
    H, W = 200, 200
    
    # Grab training coordinates to find a good starting radius
    _, c2ws_train, _, _, _, K_np = load_data(os.path.join(BASE_DIR, "data/lego_200x200.npz"))
    START_POS = c2ws_train[0][:3, 3].cpu().numpy() if torch.is_tensor(c2ws_train) else c2ws_train[0][:3, 3]
    K = K_np.float().to(device) if torch.is_tensor(K_np) else torch.from_numpy(K_np).float().to(device)
    
    NUM_SAMPLES_FRAMES = 60
    frames = []
    
    print("Generating novel views...")
    for phi in tqdm(np.linspace(360., 0., NUM_SAMPLES_FRAMES, endpoint=False)):
        c2w_np = look_at_origin(START_POS)
        extrinsic_np = rot_x(phi/180.*np.pi) @ c2w_np
        extrinsic = torch.from_numpy(extrinsic_np).float().to(device)
        
        fake_img_shape = torch.zeros((H, W, 3), device=device)
        rays = image_to_rays(fake_img_shape, extrinsic, K, device=device)
        
        rays_o_val = rays[..., :3].reshape(-1, 3)
        rays_d_val = rays[..., 3:].reshape(-1, 3)
        
        rendered_val = []
        with torch.no_grad():
            for chunk_idx in range(0, rays_o_val.shape[0], chunk_size):
                ro_batch = rays_o_val[chunk_idx:chunk_idx+chunk_size]
                rd_batch = rays_d_val[chunk_idx:chunk_idx+chunk_size]
                
                xyzs_val = sample_along_rays(ro_batch, rd_batch, near, far, num_samples, perturb=False, device=device)
                rgb_b, sig_b = model(xyzs_val, rd_batch)
                chunk_rendered = volrend(sig_b, rgb_b, near, far, num_samples, device=device)
                
                rendered_val.append(chunk_rendered)
                
        rendered_val = torch.cat(rendered_val, dim=0).reshape(H, W, 3)
        rendered_np = rendered_val.cpu().numpy().clip(0, 1)
        
        frame = (rendered_np * 255.0).astype(np.uint8)
        frames.append(frame)
        
    os.makedirs(os.path.join(BASE_DIR, "output/phase3"), exist_ok=True)
    imageio.mimsave(os.path.join(BASE_DIR, "output/phase3/lego_orbit.gif"), frames, fps=15)
    print("SUCCESS! Orbital GIF rendered perfectly and saved to output/phase3/lego_orbit.gif!")
